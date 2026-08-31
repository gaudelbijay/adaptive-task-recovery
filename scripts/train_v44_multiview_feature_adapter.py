#!/usr/bin/env python3
"""Train a routed feature adapter from synchronized renderer views."""

import argparse, hashlib, importlib, json, os, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_v41_visual_recovery_unseen_ood import install_extensions
from train_v19_factorized_canonical import assert_synchronized
from train_visual_recovery_dual_teacher_ppo import atomic_save, env_kwargs, extract_observation, file_sha256, observation_contract, privileged_aux_dim, select_task
from train_v37_dense_canonical import vector_env
from train_v38_cardinality_aligned_canonical import cumulative_source_interactions
from v44_multiview_feature_agent import MultiViewFeatureV41Agent, AlwaysMultiViewFeatureV41Agent


def action_from_latent(agent, latent, proprio):
    parts = [latent, proprio]
    if agent.goal_progress_predictor is not None:
        parts.append(torch.sigmoid(agent.goal_progress_predictor(latent)))
    return torch.tanh(agent.actor(torch.cat(parts, dim=1)))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--output",default="results/visual_recovery_ppo"); parser.add_argument("--task-index",type=int,default=0); parser.add_argument("--preflight",action="store_true"); args=parser.parse_args()
    config=json.loads(Path(args.config).read_text()); task,count_tasks=select_task(config,args.task_index)
    if args.preflight: print(json.dumps({"task_count":count_tasks,**task},indent=2)); return
    if not torch.cuda.is_available(): raise RuntimeError("V44 training requires CUDA")
    registration=importlib.import_module(task["registration_module"]); install_extensions()
    seed=int(task["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.backends.cudnn.deterministic=True
    run_dir=Path(args.output)/config["name"]/task["method"]/f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()): raise FileExistsError(f"refusing to overwrite V44 run: {run_dir}")
    run_dir.mkdir(parents=True); (run_dir/"task.json").write_text(json.dumps(task,indent=2)+"\n")
    n=int(task["num_envs"]); reference=vector_env(task["reference_env_id"],n,env_kwargs(task)); profiles={}
    for name in task["paired_environment_profiles"]:
        kw=env_kwargs(task); kw["visual_domain_profile"]=name; profiles[name]=vector_env("LearnedRecovery-v3-OOD",n,kw)
    reference_obs,_=reference.reset(seed=seed); shifted={name:env.reset(seed=seed)[0] for name,env in profiles.items()}
    for obs in shifted.values(): assert_synchronized(reference_obs,obs,task)
    rgb,proprio,critic=extract_observation(reference_obs,task["asymmetric_critic"],task.get("actor_tcp_pose",False),False)
    action_dim=int(np.prod(reference.single_action_space.shape)); agent_class=AlwaysMultiViewFeatureV41Agent if task.get("deployment_mode")=="always_feature" else MultiViewFeatureV41Agent; agent=agent_class(task["image_size"],proprio.shape[1],critic.shape[1],action_dim,task["asymmetric_critic"],0,privileged_aux_dim(task),task.get("actor_learned_goal_progress",False)).cuda()
    source_path=Path(task["source_visual_checkpoint"].format(seed=seed)); source=torch.load(source_path,map_location="cuda",weights_only=False)
    if source.get("training_protocol")!="backkey_targeted_dense_repair_v19": raise ValueError("V44 source protocol mismatch")
    if source.get("observation_contract")!=observation_contract(task): raise ValueError("V44 observation contract mismatch")
    agent.initialize_from_v40(source["agent"])
    params=list(agent.renderer_encoder.parameters())+(list(agent.router.parameters()) if task.get("deployment_mode")!="always_feature" else []); optimizer=torch.optim.AdamW(params,lr=config["learning_rate"],weight_decay=config["weight_decay"],eps=1e-5)
    updates=int(task["feature_updates"]); transitions=updates*n*(1+len(profiles))
    if transitions!=task["total_timesteps"]: raise ValueError("V44 transition budget mismatch")
    cycle=task["profile_sampling_cycle"]; history=[]; resets=0; agent.train(); agent.v41.eval()
    for update in range(updates):
        rgb,proprio,_=extract_observation(reference_obs,task["asymmetric_critic"],task.get("actor_tcp_pose",False),False); name=cycle[update%len(cycle)]; shifted_rgb=shifted[name]["sensor_data"]["base_camera"]["rgb"]
        with torch.no_grad(): target=agent.v41.encode(rgb); target_action=action_from_latent(agent,target,proprio)
        adapted=agent.renderer_latent(shifted_rgb); adapted_action=action_from_latent(agent,adapted,proprio); clean_adapted=agent.renderer_latent(rgb); clean_action=action_from_latent(agent,clean_adapted,proprio)
        feature_loss=F.smooth_l1_loss(adapted,target,beta=0.1); cosine_loss=(1-F.cosine_similarity(adapted,target,dim=1)).mean(); action_loss=F.mse_loss(adapted_action,target_action)
        clean_logits=agent.route_logits(rgb); shifted_logits=agent.route_logits(shifted_rgb); route_loss=F.binary_cross_entropy_with_logits(clean_logits,torch.zeros_like(clean_logits))+F.binary_cross_entropy_with_logits(shifted_logits,torch.ones_like(shifted_logits))
        clean_feature_loss=F.smooth_l1_loss(clean_adapted,target,beta=0.1); clean_action_loss=F.mse_loss(clean_action,target_action)
        loss=task["feature_weight"]*feature_loss+task["cosine_weight"]*cosine_loss+task["action_weight"]*action_loss+task["route_weight"]*route_loss+task.get("clean_feature_weight",0.0)*clean_feature_loss+task.get("clean_action_weight",0.0)*clean_action_loss
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); optimizer.step()
        with torch.no_grad():
            executed=agent.v41.base.get_action(rgb,proprio,deterministic=True); reference_obs,_,term,trunc,_=reference.step(executed); done=torch.logical_or(term,trunc)
            for key,env in profiles.items(): shifted[key],_,t,u,_=env.step(executed); done=torch.logical_or(done,torch.logical_or(t,u))
            if bool(done.any()):
                resets+=1; paired_seed=seed+40_000_000+resets; reference_obs,_=reference.reset(seed=paired_seed); shifted={key:env.reset(seed=paired_seed)[0] for key,env in profiles.items()}
        if (update+1)%task["synchronization_check_frequency"]==0:
            for obs in shifted.values(): assert_synchronized(reference_obs,obs,task)
        item={"loss":float(loss.detach()),"feature_loss":float(feature_loss.detach()),"cosine_loss":float(cosine_loss.detach()),"action_loss":float(action_loss.detach()),"route_loss":float(route_loss.detach()),"clean_route_probability":float(torch.sigmoid(clean_logits).mean().detach()),"shifted_route_probability":float(torch.sigmoid(shifted_logits).mean().detach())}; history.append(item)
        if (update+1)%config["log_freq"]==0:
            recent=history[-config["log_freq"]:]; payload={"update":update+1,"simulator_transitions":(update+1)*n*(1+len(profiles))}; payload.update({k:float(np.mean([x[k] for x in recent])) for k in item}); (run_dir/"metrics.jsonl").open("a").write(json.dumps(payload)+"\n")
    reference.close(); [env.close() for env in profiles.values()]; recent=history[-100:]; metrics={f"mean_last_100_{k}":float(np.mean([x[k] for x in recent])) for k in history[-1]}
    hashes={"trainer":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"agent":hashlib.sha256(Path(__file__).with_name("v44_multiview_feature_agent.py").read_bytes()).hexdigest(),"environment":hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest()}
    checkpoint={"schema_version":1,"training_protocol":"routed_multiview_feature_adapter_v19","observation_contract":observation_contract(task),"source_sha256":hashes,"task":task,"agent":agent.state_dict(),"optimizer":optimizer.state_dict(),"iteration":updates,"global_step":transitions,"best_score":-metrics["mean_last_100_loss"],"best_metrics":metrics}; atomic_save(checkpoint,run_dir/"best.pt"); atomic_save(checkpoint,run_dir/"latest.pt")
    init=cumulative_source_interactions(json.loads((source_path.parent/"TRAINING_COMPLETE.json").read_text())); completion={"schema_version":1,"training_protocol":"routed_multiview_feature_adapter_v19","global_step":transitions,"feature_adapter_transitions":transitions,"simulator_transitions":transitions,"ppo_environment_steps":0,"initialization_simulator_transitions":init,"total_simulator_transitions":init+transitions,"deployment_actor_inputs":"rgb_qpos_qvel_tcp_instruction_learned_progress","evaluation_domain_label_available":False,"source_visual_checkpoint":str(source_path),"source_visual_checkpoint_sha256":file_sha256(source_path),**metrics,"source_sha256":hashes}; (run_dir/"TRAINING_COMPLETE.json").write_text(json.dumps(completion,indent=2,sort_keys=True)+"\n"); print(json.dumps(completion,indent=2,sort_keys=True))


if __name__=="__main__": main()
