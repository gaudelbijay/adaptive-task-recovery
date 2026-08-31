#!/usr/bin/env python3
"""Train confidence-gated continuous geometry experts above frozen V53."""

import argparse, hashlib, importlib, json, math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_v52_visual_recovery_unseen_ood import install_extensions
from train_v19_factorized_canonical import assert_synchronized
from train_v37_dense_canonical import vector_env
from train_v38_cardinality_aligned_canonical import cumulative_source_interactions
from train_v44_multiview_feature_adapter import action_from_latent
from train_visual_recovery_dual_teacher_ppo import atomic_save, env_kwargs, extract_observation, file_sha256, observation_contract, privileged_aux_dim, select_task
from v36_continuous_canonical_agent import synthesize_corruption
from v54_continuous_geometry_agent import ContinuousGeometryCompositionAgent


def sample_parameters(count, label, device):
    result = torch.zeros((count, 4), device=device)
    if label in (1, 4):
        magnitude = torch.empty(count, device=device).uniform_(2.0, 5.0)
        result[:, 0] = magnitude * torch.where(torch.rand(count, device=device) < .5, -1, 1)
    if label in (2, 4):
        magnitude = torch.empty(count, device=device).uniform_(math.radians(3), math.radians(8))
        result[:, 2] = magnitude * torch.where(torch.rand(count, device=device) < .5, -1, 1)
    if label in (3, 4):
        low = torch.empty(count, device=device).uniform_(math.log(.86), math.log(.95))
        high = torch.empty(count, device=device).uniform_(math.log(1.05), math.log(1.18))
        result[:, 3] = torch.where(torch.rand(count, device=device) < .5, low, high)
    if label == 4:
        result[:, 1] = torch.empty(count, device=device).uniform_(-2.0, 2.0)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--output", default="results/visual_recovery_ppo"); p.add_argument("--task-index", type=int, default=0); p.add_argument("--preflight", action="store_true"); a = p.parse_args()
    c = json.loads(Path(a.config).read_text()); t, count = select_task(c, a.task_index)
    if a.preflight: print(json.dumps({"task_count": count, **t}, indent=2)); return
    if not torch.cuda.is_available(): raise RuntimeError("V54 training requires CUDA")
    registration = importlib.import_module(t["registration_module"]); install_extensions(); seed = int(t["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.backends.cudnn.deterministic = True
    run = Path(a.output) / c["name"] / t["method"] / f"seed_{seed}"
    if run.exists() and any(run.iterdir()): raise FileExistsError(f"refusing to overwrite V54 run: {run}")
    run.mkdir(parents=True); (run / "task.json").write_text(json.dumps(t, indent=2) + "\n")
    n = int(t["num_envs"]); reference = vector_env(t["reference_env_id"], n, env_kwargs(t)); profiles = {}
    for name in t["opened_profiles"]:
        kw = env_kwargs(t); kw["visual_domain_profile"] = name; profiles[name] = vector_env("LearnedRecovery-v3-OOD", n, kw)
    ref, _ = reference.reset(seed=seed); shifted = {name: env.reset(seed=seed)[0] for name, env in profiles.items()}
    for obs in shifted.values(): assert_synchronized(ref, obs, t)
    rgb, proprio, critic = extract_observation(ref, t["asymmetric_critic"], t.get("actor_tcp_pose", False), False)
    action_dim = int(np.prod(reference.single_action_space.shape))
    agent = ContinuousGeometryCompositionAgent(t["image_size"], proprio.shape[1], critic.shape[1], action_dim, t["asymmetric_critic"], 0, privileged_aux_dim(t), t.get("actor_learned_goal_progress", False)).cuda()
    v53_path = Path(t["source_v53_checkpoint"].format(seed=seed)); v39_path = Path(t["source_v39_checkpoint"].format(seed=seed))
    v53 = torch.load(v53_path, map_location="cuda", weights_only=False); v39 = torch.load(v39_path, map_location="cuda", weights_only=False)
    if v53.get("training_protocol") != "opened_renderer_experts_v19" or v39.get("training_protocol") != "backkey_targeted_dense_repair_v19": raise ValueError("V54 source protocol mismatch")
    if v53.get("observation_contract") != observation_contract(t) or v39.get("observation_contract") != observation_contract(t): raise ValueError("V54 source contract mismatch")
    agent.initialize(v53["agent"], v39["agent"])
    expert_params = list(agent.global_correctors.parameters()) + list(agent.dense_correctors.parameters()); router_params = list(agent.router.parameters())
    optimizer = torch.optim.AdamW([{"params": expert_params, "lr": c["expert_learning_rate"]}, {"params": router_params, "lr": c["router_learning_rate"]}], weight_decay=c["weight_decay"], eps=1e-5)
    updates = int(t["expert_updates"]); transitions = updates * n * (1 + len(profiles))
    if transitions != int(t["total_timesteps"]): raise ValueError("V54 budget mismatch")
    history = []; resets = 0; names = list(t["opened_profiles"]); agent.train(); agent.base.eval(); agent.geometry_encoder.eval()
    ones = torch.ones((n, 3), device="cuda"); zeros = torch.zeros((n, 3), device="cuda")
    for update in range(updates):
        rgb, proprio, _ = extract_observation(ref, t["asymmetric_critic"], t.get("actor_tcp_pose", False), False)
        parameters = [sample_parameters(n, label, rgb.device) for label in range(1, 5)]
        corruptions = [synthesize_corruption(rgb, item, ones, zeros) for item in parameters]
        with torch.no_grad():
            target_latent = agent.geometry_encoder.encode(rgb); target_action = action_from_latent(agent, target_latent, proprio)
        corrected = [agent.correct(i, image) for i, image in enumerate(corruptions)]
        latents = [agent.geometry_encoder.encode(image) for image in corrected]
        actions = [action_from_latent(agent, latent, proprio) for latent in latents]
        image_loss = torch.stack([F.l1_loss(image.div(255.0), rgb.float().div(255.0)) for image in corrected]).mean()
        feature_loss = torch.stack([F.smooth_l1_loss(z, target_latent, beta=.1) for z in latents]).mean()
        action_loss = torch.stack([F.mse_loss(action, target_action) for action in actions]).mean()
        geometry_loss = torch.stack([F.smooth_l1_loss(agent.global_correctors[i].estimate(corruptions[i])[1] / torch.tensor([5., 2., math.radians(8), abs(math.log(.86))], device=rgb.device), parameters[i] / torch.tensor([5., 2., math.radians(8), abs(math.log(.86))], device=rgb.device), beta=.05) for i in range(4)]).mean()
        default = [rgb] + [shifted[name]["sensor_data"]["base_camera"]["rgb"] for name in names]
        router_images = default + corruptions; logits = agent.router_logits(torch.cat(router_images)); labels = [0] * len(default) + list(range(1, 5)); targets = torch.cat([torch.full((n,), label, device=rgb.device, dtype=torch.long) for label in labels])
        router_loss = F.cross_entropy(logits, targets)
        loss = t["image_weight"] * image_loss + t["feature_weight"] * feature_loss + t["action_weight"] * action_loss + t["geometry_weight"] * geometry_loss + t["router_weight"] * router_loss
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(expert_params + router_params, 1); optimizer.step()
        with torch.no_grad():
            accuracy = (logits.argmax(1) == targets).float().mean(); executed = agent.base.base.v51.v47.v41.base.get_action(rgb, proprio, deterministic=True)
            ref, _, term, trunc, _ = reference.step(executed); done = torch.logical_or(term, trunc)
            for name, env in profiles.items(): shifted[name], _, term, trunc, _ = env.step(executed); done = torch.logical_or(done, torch.logical_or(term, trunc))
        if bool(done.any()):
            resets += 1; paired = seed + 110_000_000 + resets; ref, _ = reference.reset(seed=paired); shifted = {name: env.reset(seed=paired)[0] for name, env in profiles.items()}
        if (update + 1) % int(t["synchronization_check_frequency"]) == 0:
            for obs in shifted.values(): assert_synchronized(ref, obs, t)
        item = {"loss": float(loss.detach()), "image_loss": float(image_loss.detach()), "feature_loss": float(feature_loss.detach()), "action_loss": float(action_loss.detach()), "geometry_loss": float(geometry_loss.detach()), "router_loss": float(router_loss.detach()), "router_accuracy": float(accuracy.detach())}; history.append(item)
        if (update + 1) % int(c["log_freq"]) == 0:
            recent = history[-int(c["log_freq"]):]; payload = {"update": update + 1, "simulator_transitions": (update + 1) * n * (1 + len(profiles))}; payload.update({key: float(np.mean([x[key] for x in recent])) for key in item});
            with (run / "metrics.jsonl").open("a") as stream: stream.write(json.dumps(payload) + "\n")
    reference.close(); [env.close() for env in profiles.values()]; recent = history[-100:]; metrics = {f"mean_last_100_{key}": float(np.mean([x[key] for x in recent])) for key in history[-1]}
    hashes = {"trainer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "agent": hashlib.sha256(Path(__file__).with_name("v54_continuous_geometry_agent.py").read_bytes()).hexdigest(), "environment": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest()}
    checkpoint = {"schema_version": 1, "training_protocol": "continuous_geometry_composition_v19", "observation_contract": observation_contract(t), "source_sha256": hashes, "task": t, "agent": agent.state_dict(), "optimizer": optimizer.state_dict(), "iteration": updates, "global_step": transitions, "best_score": -metrics["mean_last_100_loss"], "best_metrics": metrics}; atomic_save(checkpoint, run / "best.pt"); atomic_save(checkpoint, run / "latest.pt")
    v53_completion = json.loads((v53_path.parent / "TRAINING_COMPLETE.json").read_text()); v39_completion = json.loads((v39_path.parent / "TRAINING_COMPLETE.json").read_text()); initialization = cumulative_source_interactions(v53_completion) + cumulative_source_interactions(v39_completion)
    completion = {"schema_version": 1, "training_protocol": "continuous_geometry_composition_v19", "global_step": transitions, "geometry_training_transitions": transitions, "simulator_transitions": transitions, "ppo_environment_steps": 0, "initialization_simulator_transitions": initialization, "total_simulator_transitions": initialization + transitions, "deployment_actor_inputs": "rgb_qpos_qvel_tcp_instruction_learned_progress", "evaluation_domain_label_available": False, "source_v53_checkpoint": str(v53_path), "source_v53_checkpoint_sha256": file_sha256(v53_path), "source_v39_checkpoint": str(v39_path), "source_v39_checkpoint_sha256": file_sha256(v39_path), **metrics, "source_sha256": hashes}; (run / "TRAINING_COMPLETE.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n"); print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__": main()
