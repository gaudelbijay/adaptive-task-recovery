---
title: References
status: draft
last_updated: 2026-07-24
---

# References

A curated reading list by topic, referenced from [02-background-and-related-work.md](02-background-and-related-work.md). Entries are recorded by author/title/venue as best known; **verify exact citation details (year, venue, arXiv ID) directly when you actually read each one** — treat this list as pointers to go find and confirm, not a bibliography ready to paste into a paper.

Fill in the "Notes" column as you read — a reading list with no notes six months from now is just a list of titles you'll have forgotten the content of.

## Simulation platforms

| Reference | Topic | Notes |
|---|---|---|
| ManiSkill3 project/paper (SAPIEN-based GPU-parallel manipulation benchmark) | Primary simulator for this project | |
| Isaac Lab / Isaac Gym documentation and associated papers | Comparison platform, widely used for legged/humanoid RL | |
| MuJoCo / MJX documentation | Comparison platform, accurate contact, JAX vectorization | |

## Failure detection / anomaly detection

| Reference | Topic | Notes |
|---|---|---|
| Ensemble-based epistemic uncertainty / model disagreement literature (model-based RL safety) | Dynamics-ensemble failure detection | |
| VAE / autoencoder-based anomaly detection literature | Reconstruction-error OOD detection | |
| Classic force/torque and tactile sensing papers on slip detection | Contact-based failure signal | |

## Recovery and safe RL

| Reference | Topic | Notes |
|---|---|---|
| Thananjeyan, Balakrishna, et al. — "Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones" | Risk-aware recovery-policy architecture template | Confirm exact venue/year when reading |
| Control barrier function / Lyapunov-based safe RL literature | Safety-layer wrapping for learned policies | |
| Sutton, Precup, Singh — "Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning" | Options framework | Foundational hierarchical RL reference |
| Reset-free / autonomous RL literature | Training without external episode resets | |

## Humanoid/bipedal control and locomotion

| Reference | Topic | Notes |
|---|---|---|
| Capture point / ZMP-based push-recovery papers (classical bipedal balance control) | Balance-recovery controller design | Read at least one before implementing `step-recovery` |
| Whole-body control / QP-based controller papers | Low-level controller under RL recovery skills | |
| GPU-parallel sim-to-real humanoid/legged locomotion papers (e.g., Isaac Gym-trained locomotion policies) | Domain randomization and reward-shaping methodology | |

## Vision-language(-action) models and high-level planning

| Reference | Topic | Notes |
|---|---|---|
| Ahn et al. — "SayCan" (Do As I Can, Not As I Say) | LLM-grounded affordance/skill selection | Future-work context |
| RT-2 (Google DeepMind) | Vision-language-action model | Future-work context |
| "Eureka" (Nvidia) — LLM-designed reward functions | Reward-shaping accelerant | Optional engineering tool, not core method |

## How to extend this list

Add a row whenever [02-background-and-related-work.md](02-background-and-related-work.md) references something new. Prefer adding the *exact* citation (authors, year, venue/arXiv ID) once you've actually located and skimmed the paper, rather than guessing details up front.
