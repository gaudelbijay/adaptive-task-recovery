#!/usr/bin/env python3
"""Train a balanced altered-geometry versus default RGB router."""

import argparse, hashlib, importlib, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from evaluate_v52_visual_recovery_unseen_ood import install_extensions
from train_v19_factorized_canonical import assert_synchronized
from train_v37_dense_canonical import vector_env
from train_visual_recovery_dual_teacher_ppo import atomic_save, env_kwargs, extract_observation, file_sha256, observation_contract, privileged_aux_dim, select_task
from train_v54_continuous_geometry import sample_parameters
from v36_continuous_canonical_agent import synthesize_corruption
from v55_binary_geometry_agent import BinaryGeometryCompositionAgent


def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",default="results/visual_recovery_ppo");p.add_argument("--task-index",type=int,default=0);p.add_argument("--preflight",action="store_true");a=p.parse_args();c=json.loads(Path(a.config).read_text());t,count=select_task(c,a.task_index)
 if a.preflight:print(json.dumps({"task_count":count,**t},indent=2));return
 if not torch.cuda.is_available():raise RuntimeError("V55 training requires CUDA")
 registration=importlib.import_module(t["registration_module"]);install_extensions();seed=int(t["seed"]);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.backends.cudnn.deterministic=True
 run=Path(a.output)/c["name"]/t["method"]/f"seed_{seed}";
 if run.exists() and any(run.iterdir()):raise FileExistsError(f"refusing to overwrite V55 run: {run}")
 run.mkdir(parents=True);(run/"task.json").write_text(json.dumps(t,indent=2)+"\n");n=int(t["num_envs"]);reference=vector_env(t["reference_env_id"],n,env_kwargs(t));profiles={}
 for name in t["opened_profiles"]:kw=env_kwargs(t);kw["visual_domain_profile"]=name;profiles[name]=vector_env("LearnedRecovery-v3-OOD",n,kw)
 ref,_=reference.reset(seed=seed);shifted={name:env.reset(seed=seed)[0] for name,env in profiles.items()};[assert_synchronized(ref,obs,t) for obs in shifted.values()];rgb,proprio,critic=extract_observation(ref,t["asymmetric_critic"],t.get("actor_tcp_pose",False),False);action_dim=int(np.prod(reference.single_action_space.shape));agent=BinaryGeometryCompositionAgent(t["image_size"],proprio.shape[1],critic.shape[1],action_dim,t["asymmetric_critic"],0,privileged_aux_dim(t),t.get("actor_learned_goal_progress",False)).cuda();v53_path=Path(t["source_v53_checkpoint"].format(seed=seed));v39_path=Path(t["source_v39_checkpoint"].format(seed=seed));v53=torch.load(v53_path,map_location="cuda",weights_only=False);v39=torch.load(v39_path,map_location="cuda",weights_only=False);agent.initialize(v53["agent"],v39["agent"])
 for parameter in agent.parameters():parameter.requires_grad_(False)
 for parameter in agent.router.parameters():parameter.requires_grad_(True)
 optimizer=torch.optim.AdamW(agent.router.parameters(),lr=c["router_learning_rate"],weight_decay=c["weight_decay"],eps=1e-5);updates=int(t["router_updates"]);transitions=updates*n*(1+len(profiles));
 if transitions!=int(t["total_timesteps"]):raise ValueError("V55 budget mismatch")
 history=[];resets=0;names=list(t["opened_profiles"]);ones=torch.ones((n,3),device="cuda");zeros=torch.zeros((n,3),device="cuda");agent.train();agent.base.eval();agent.geometry_encoder.eval()
 for update in range(updates):
  rgb,proprio,_=extract_observation(ref,t["asymmetric_critic"],t.get("actor_tcp_pose",False),False);corruptions=[synthesize_corruption(rgb,sample_parameters(n,label,rgb.device),ones,zeros) for label in range(1,5)];defaults=[rgb]+[shifted[name]["sensor_data"]["base_camera"]["rgb"] for name in names];images=torch.cat(defaults+corruptions);targets=torch.cat([torch.zeros(n*len(defaults),device=rgb.device,dtype=torch.long),torch.ones(n*len(corruptions),device=rgb.device,dtype=torch.long)]);logits=agent.router_logits(images);loss=F.cross_entropy(logits,targets);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(agent.router.parameters(),1);optimizer.step()
  with torch.no_grad():accuracy=(logits.argmax(1)==targets).float().mean();executed=agent.base.base.v51.v47.v41.base.get_action(rgb,proprio,deterministic=True);ref,_,term,trunc,_=reference.step(executed);done=torch.logical_or(term,trunc)
  for name,env in profiles.items():shifted[name],_,term,trunc,_=env.step(executed);done=torch.logical_or(done,torch.logical_or(term,trunc))
  if bool(done.any()):resets+=1;paired=seed+120_000_000+resets;ref,_=reference.reset(seed=paired);shifted={name:env.reset(seed=paired)[0] for name,env in profiles.items()}
  if (update+1)%int(t["synchronization_check_frequency"])==0:[assert_synchronized(ref,obs,t) for obs in shifted.values()]
  history.append({"loss":float(loss.detach()),"accuracy":float(accuracy.detach())})
  if (update+1)%int(c["log_freq"])==0:
   recent=history[-int(c["log_freq"]):]
   with (run/"metrics.jsonl").open("a") as stream:stream.write(json.dumps({"update":update+1,"simulator_transitions":(update+1)*n*(1+len(profiles)),"router_loss":float(np.mean([x["loss"] for x in recent])),"router_accuracy":float(np.mean([x["accuracy"] for x in recent]))})+"\n")
 reference.close();[env.close() for env in profiles.values()];metrics={"mean_last_100_router_loss":float(np.mean([x["loss"] for x in history[-100:]])),"mean_last_100_router_accuracy":float(np.mean([x["accuracy"] for x in history[-100:]]))};hashes={"trainer":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"agent":hashlib.sha256(Path(__file__).with_name("v55_binary_geometry_agent.py").read_bytes()).hexdigest(),"environment":hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest()};checkpoint={"schema_version":1,"training_protocol":"binary_geometry_router_v19","observation_contract":observation_contract(t),"source_sha256":hashes,"task":t,"agent":agent.state_dict(),"optimizer":optimizer.state_dict(),"iteration":updates,"global_step":transitions,"best_score":-metrics["mean_last_100_router_loss"],"best_metrics":metrics};atomic_save(checkpoint,run/"best.pt");atomic_save(checkpoint,run/"latest.pt");completion={"schema_version":1,"training_protocol":"binary_geometry_router_v19","global_step":transitions,"router_training_transitions":transitions,"simulator_transitions":transitions,"ppo_environment_steps":0,"initialization_simulator_transitions":0,"total_simulator_transitions":transitions,"evaluation_domain_label_available":False,"source_v53_checkpoint_sha256":file_sha256(v53_path),"source_v39_checkpoint_sha256":file_sha256(v39_path),**metrics,"source_sha256":hashes};(run/"TRAINING_COMPLETE.json").write_text(json.dumps(completion,indent=2,sort_keys=True)+"\n");print(json.dumps(completion,indent=2,sort_keys=True))

if __name__=="__main__":main()
