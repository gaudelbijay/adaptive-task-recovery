#!/usr/bin/env python3
"""Train four confidence-gated experts on the opened seed-127M renderers."""

import argparse,hashlib,importlib,json,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from evaluate_v52_visual_recovery_unseen_ood import apply_visual_perturbation, install_extensions
from train_v19_factorized_canonical import assert_synchronized
from train_v44_multiview_feature_adapter import action_from_latent
from train_visual_recovery_dual_teacher_ppo import atomic_save,env_kwargs,extract_observation,file_sha256,observation_contract,privileged_aux_dim,select_task
from train_v37_dense_canonical import vector_env
from train_v38_cardinality_aligned_canonical import cumulative_source_interactions
from v53_opened_renderer_agent import OpenedRendererExpertAgent

GEOMETRY=("subpixel_shift_left_3_5","rotation_clockwise_6deg","scale_90","combined_similarity_v2")

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",default="results/visual_recovery_ppo");p.add_argument("--task-index",type=int,default=0);p.add_argument("--preflight",action="store_true");a=p.parse_args();c=json.loads(Path(a.config).read_text());t,count=select_task(c,a.task_index)
 if a.preflight: print(json.dumps({"task_count":count,**t},indent=2));return
 if not torch.cuda.is_available():raise RuntimeError("V53 training requires CUDA")
 registration=importlib.import_module(t["registration_module"]);install_extensions();seed=int(t["seed"]);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.backends.cudnn.deterministic=True
 run=Path(a.output)/c["name"]/t["method"]/f"seed_{seed}";
 if run.exists() and any(run.iterdir()):raise FileExistsError(f"refusing to overwrite V53 run: {run}")
 run.mkdir(parents=True);(run/"task.json").write_text(json.dumps(t,indent=2)+"\n");n=int(t["num_envs"]);reference=vector_env(t["reference_env_id"],n,env_kwargs(t));profiles={}
 for name in t["opened_profiles"]: kw=env_kwargs(t);kw["visual_domain_profile"]=name;profiles[name]=vector_env("LearnedRecovery-v3-OOD",n,kw)
 ref,_=reference.reset(seed=seed);shifted={name:env.reset(seed=seed)[0] for name,env in profiles.items()}
 for obs in shifted.values():assert_synchronized(ref,obs,t)
 rgb,proprio,critic=extract_observation(ref,t["asymmetric_critic"],t.get("actor_tcp_pose",False),False);action_dim=int(np.prod(reference.single_action_space.shape));agent=OpenedRendererExpertAgent(t["image_size"],proprio.shape[1],critic.shape[1],action_dim,t["asymmetric_critic"],0,privileged_aux_dim(t),t.get("actor_learned_goal_progress",False)).cuda();source_path=Path(t["source_visual_checkpoint"].format(seed=seed));source=torch.load(source_path,map_location="cuda",weights_only=False)
 if source.get("training_protocol")!="subpixel_specialist_router_v19" or source.get("observation_contract")!=observation_contract(t):raise ValueError("V53 source mismatch")
 agent.initialize_from_v52(source["agent"]);expert_params=[p for expert in agent.experts for p in expert.parameters()];router_params=list(agent.router.parameters());optimizer=torch.optim.AdamW([{"params":expert_params,"lr":c["expert_learning_rate"]},{"params":router_params,"lr":c["router_learning_rate"]}],weight_decay=c["weight_decay"],eps=1e-5);updates=int(t["expert_updates"]);transitions=updates*n*(1+len(profiles))
 if transitions!=int(t["total_timesteps"]):raise ValueError("V53 budget mismatch")
 names=list(t["opened_profiles"]);history=[];resets=0;agent.train();agent.base.eval()
 for update in range(updates):
  rgb,proprio,_=extract_observation(ref,t["asymmetric_critic"],t.get("actor_tcp_pose",False),False);profile_rgbs=[shifted[name]["sensor_data"]["base_camera"]["rgb"] for name in names]
  with torch.no_grad():target=agent.base.v51.v47.v41.encode(rgb);target_action=action_from_latent(agent,target,proprio)
  latents=[agent.expert_latent(i,image) for i,image in enumerate(profile_rgbs)];feature_loss=torch.stack([F.smooth_l1_loss(latent,target,beta=.1) for latent in latents]).mean();cosine_loss=torch.stack([(1-F.cosine_similarity(latent,target,dim=1)).mean() for latent in latents]).mean();action_loss=torch.stack([F.mse_loss(action_from_latent(agent,latent,proprio),target_action) for latent in latents]).mean();images=[rgb]+[apply_visual_perturbation(rgb,mode) for mode in GEOMETRY]+profile_rgbs;labels=[0]*5+list(range(1,5));logits=agent.router_logits(torch.cat(images));targets=torch.cat([torch.full((n,),label,device=logits.device,dtype=torch.long) for label in labels]);router_loss=F.cross_entropy(logits,targets);loss=t["feature_weight"]*feature_loss+t["cosine_weight"]*cosine_loss+t["action_weight"]*action_loss+t["router_weight"]*router_loss;optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(expert_params+router_params,1);optimizer.step()
  with torch.no_grad():accuracy=(logits.argmax(1)==targets).float().mean();executed=agent.base.v51.v47.v41.base.get_action(rgb,proprio,deterministic=True);ref,_,term,trunc,_=reference.step(executed);done=torch.logical_or(term,trunc)
  for name,env in profiles.items():shifted[name],_,term,trunc,_=env.step(executed);done=torch.logical_or(done,torch.logical_or(term,trunc))
  if bool(done.any()):resets+=1;paired=seed+100_000_000+resets;ref,_=reference.reset(seed=paired);shifted={name:env.reset(seed=paired)[0] for name,env in profiles.items()}
  if (update+1)%int(t["synchronization_check_frequency"])==0:
   for obs in shifted.values():assert_synchronized(ref,obs,t)
  item={"loss":float(loss.detach()),"feature_loss":float(feature_loss.detach()),"cosine_loss":float(cosine_loss.detach()),"action_loss":float(action_loss.detach()),"router_loss":float(router_loss.detach()),"router_accuracy":float(accuracy.detach())};history.append(item)
  if (update+1)%int(c["log_freq"])==0:
   recent=history[-int(c["log_freq"]):];payload={"update":update+1,"simulator_transitions":(update+1)*n*(1+len(profiles))};payload.update({key:float(np.mean([x[key] for x in recent])) for key in item});
   with (run/"metrics.jsonl").open("a") as stream:stream.write(json.dumps(payload)+"\n")
 reference.close();[env.close() for env in profiles.values()];recent=history[-100:];metrics={f"mean_last_100_{key}":float(np.mean([x[key] for x in recent])) for key in history[-1]};hashes={"trainer":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"agent":hashlib.sha256(Path(__file__).with_name("v53_opened_renderer_agent.py").read_bytes()).hexdigest(),"environment":hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest()};checkpoint={"schema_version":1,"training_protocol":"opened_renderer_experts_v19","observation_contract":observation_contract(t),"source_sha256":hashes,"task":t,"agent":agent.state_dict(),"optimizer":optimizer.state_dict(),"iteration":updates,"global_step":transitions,"best_score":-metrics["mean_last_100_loss"],"best_metrics":metrics};atomic_save(checkpoint,run/"best.pt");atomic_save(checkpoint,run/"latest.pt");initialization=cumulative_source_interactions(json.loads((source_path.parent/"TRAINING_COMPLETE.json").read_text()));completion={"schema_version":1,"training_protocol":"opened_renderer_experts_v19","global_step":transitions,"feature_adapter_transitions":transitions,"simulator_transitions":transitions,"ppo_environment_steps":0,"initialization_simulator_transitions":initialization,"total_simulator_transitions":initialization+transitions,"deployment_actor_inputs":"rgb_qpos_qvel_tcp_instruction_learned_progress","evaluation_domain_label_available":False,"source_visual_checkpoint":str(source_path),"source_visual_checkpoint_sha256":file_sha256(source_path),**metrics,"source_sha256":hashes};(run/"TRAINING_COMPLETE.json").write_text(json.dumps(completion,indent=2,sort_keys=True)+"\n");print(json.dumps(completion,indent=2,sort_keys=True))

if __name__=="__main__":main()
