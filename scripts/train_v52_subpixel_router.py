#!/usr/bin/env python3
"""Train only the V52 RGB classifier for the audited subpixel specialist."""

import argparse
import hashlib
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_v41_visual_recovery_unseen_ood import apply_visual_perturbation
from train_visual_recovery_dual_teacher_ppo import atomic_save, env_kwargs, extract_observation, file_sha256, observation_contract, privileged_aux_dim, select_task
from train_v37_dense_canonical import vector_env
from v52_subpixel_specialist_agent import SubpixelSpecialistAgent


NEGATIVE_MODES = ("rotation_counterclockwise_4deg", "scale_108", "combined_similarity_v1")


def equal_state(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--output",default="results/visual_recovery_ppo"); parser.add_argument("--task-index",type=int,default=0); parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args(); config=json.loads(Path(args.config).read_text()); task,count=select_task(config,args.task_index)
    if args.preflight: print(json.dumps({"task_count":count,**task},indent=2)); return
    if not torch.cuda.is_available(): raise RuntimeError("V52 training requires CUDA")
    registration=importlib.import_module(task["registration_module"]); seed=int(task["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.backends.cudnn.deterministic=True
    run_dir=Path(args.output)/config["name"]/task["method"]/f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()): raise FileExistsError(f"refusing to overwrite V52 run: {run_dir}")
    run_dir.mkdir(parents=True); (run_dir/"task.json").write_text(json.dumps(task,indent=2)+"\n")
    n=int(task["num_envs"]); environment=vector_env(task["env_id"],n,env_kwargs(task)); observation,_=environment.reset(seed=seed)
    rgb,proprio,critic=extract_observation(observation,task["asymmetric_critic"],task.get("actor_tcp_pose",False),False); action_dim=int(np.prod(environment.single_action_space.shape))
    agent=SubpixelSpecialistAgent(task["image_size"],proprio.shape[1],critic.shape[1],action_dim,task["asymmetric_critic"],0,privileged_aux_dim(task),task.get("actor_learned_goal_progress",False)).cuda()
    v51_path=Path(task["v51_checkpoint"].format(seed=seed)); v43_path=Path(task["v43_checkpoint"].format(seed=seed)); v51=torch.load(v51_path,map_location="cuda",weights_only=False); v43=torch.load(v43_path,map_location="cuda",weights_only=False)
    if v51.get("training_protocol")!="hierarchical_renderer_experts_v19" or v43.get("training_protocol")!="broad_render_dense_repair_v19": raise ValueError("V52 source protocol mismatch")
    if v51.get("observation_contract")!=observation_contract(task) or v43.get("observation_contract")!=observation_contract(task): raise ValueError("V52 observation contract mismatch")
    agent.initialize_sources(v51["agent"],v43["agent"])
    if not equal_state(agent.v51.actor.state_dict(),agent.subpixel.actor.state_dict()): raise ValueError("V52 actor heads differ")
    if not equal_state(agent.v51.goal_progress_predictor.state_dict(),agent.subpixel.goal_progress_predictor.state_dict()): raise ValueError("V52 progress heads differ")
    parameters=list(agent.subpixel_router.parameters()); optimizer=torch.optim.AdamW(parameters,lr=config["learning_rate"],weight_decay=config["weight_decay"],eps=1e-5)
    updates=int(task["router_updates"]); transitions=updates*n
    if transitions!=int(task["total_timesteps"]): raise ValueError("V52 transition budget mismatch")
    history=[]; resets=0; agent.train(); agent.v51.eval(); agent.subpixel.eval()
    for update in range(updates):
        rgb,proprio,_=extract_observation(observation,task["asymmetric_critic"],task.get("actor_tcp_pose",False),False); positive=apply_visual_perturbation(rgb,"subpixel_shift_right_2_25"); negatives=[rgb]+[apply_visual_perturbation(rgb,mode) for mode in NEGATIVE_MODES]
        positive_logits=agent.subpixel_logits(positive); negative_logits=[agent.subpixel_logits(image) for image in negatives]
        loss=F.binary_cross_entropy_with_logits(positive_logits,torch.ones_like(positive_logits))+torch.stack([F.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits)) for logits in negative_logits]).mean()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,1.0); optimizer.step()
        with torch.no_grad():
            positive_accuracy=(torch.sigmoid(positive_logits)>=0.5).float().mean(); negative_accuracy=torch.stack([(torch.sigmoid(logits)<0.5).float().mean() for logits in negative_logits]).mean(); executed=agent.v51.v47.v41.base.get_action(rgb,proprio,deterministic=True); observation,_,term,trunc,_=environment.step(executed)
            if bool(torch.logical_or(term,trunc).any()): resets+=1; observation,_=environment.reset(seed=seed+90_000_000+resets)
        item={"loss":float(loss.detach()),"positive_accuracy":float(positive_accuracy.detach()),"negative_accuracy":float(negative_accuracy.detach())}; history.append(item)
        if (update+1)%int(config["log_freq"])==0:
            recent=history[-int(config["log_freq"]):]; payload={"update":update+1,"simulator_transitions":(update+1)*n}; payload.update({key:float(np.mean([record[key] for record in recent])) for key in item});
            with (run_dir/"metrics.jsonl").open("a") as stream: stream.write(json.dumps(payload)+"\n")
    environment.close(); recent=history[-100:]; metrics={f"mean_last_100_{key}":float(np.mean([record[key] for record in recent])) for key in history[-1]}
    hashes={"trainer":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"agent":hashlib.sha256(Path(__file__).with_name("v52_subpixel_specialist_agent.py").read_bytes()).hexdigest(),"environment":hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest()}
    checkpoint={"schema_version":1,"training_protocol":"subpixel_specialist_router_v19","observation_contract":observation_contract(task),"source_sha256":hashes,"task":task,"agent":agent.state_dict(),"optimizer":optimizer.state_dict(),"iteration":updates,"global_step":transitions,"best_score":-metrics["mean_last_100_loss"],"best_metrics":metrics}; atomic_save(checkpoint,run_dir/"best.pt"); atomic_save(checkpoint,run_dir/"latest.pt")
    v51_complete=json.loads((v51_path.parent/"TRAINING_COMPLETE.json").read_text()); v43_complete=json.loads((v43_path.parent/"TRAINING_COMPLETE.json").read_text()); initialization=int(v51_complete["total_simulator_transitions"])+int(v43_complete["dense_finetune_transitions"])
    completion={"schema_version":1,"training_protocol":"subpixel_specialist_router_v19","global_step":transitions,"feature_adapter_transitions":transitions,"simulator_transitions":transitions,"ppo_environment_steps":0,"initialization_simulator_transitions":initialization,"total_simulator_transitions":initialization+transitions,"deployment_actor_inputs":"rgb_qpos_qvel_tcp_instruction_learned_progress","evaluation_domain_label_available":False,"v51_checkpoint":str(v51_path),"v51_checkpoint_sha256":file_sha256(v51_path),"v43_checkpoint":str(v43_path),"v43_checkpoint_sha256":file_sha256(v43_path),**metrics,"source_sha256":hashes}; (run_dir/"TRAINING_COMPLETE.json").write_text(json.dumps(completion,indent=2,sort_keys=True)+"\n"); print(json.dumps(completion,indent=2,sort_keys=True))


if __name__=="__main__": main()
