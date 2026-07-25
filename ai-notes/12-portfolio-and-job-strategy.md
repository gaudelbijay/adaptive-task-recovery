---
title: Portfolio and Job Strategy
status: draft
last_updated: 2026-07-24
---

# Portfolio and Job Strategy

## 1. Why this project is a strong portfolio piece

It touches, in one coherent system, most of what humanoid/robot-learning job postings actually list: RL, simulation, sim-to-real, systems design, safety-critical thinking, and rigorous evaluation — and it's built around a **real open problem** (failure recovery) rather than a re-implementation of a known benchmark result. The differentiator versus "I trained a policy to do X in a tutorial sim" is (a) the explicit, honest evaluation methodology in [10-evaluation-and-benchmarks.md](10-evaluation-and-benchmarks.md), and (b) the systems/architecture reasoning in [03-system-architecture.md](03-system-architecture.md) — both of which are exactly what technical interviewers probe past the "does it work" surface level.

## 2. Deliverables to produce (in priority order)

1. **A genuinely clean public repo**: good top-level README (problem, approach, results, GIFs/video, how to reproduce), `ai-notes/` kept as living design docs (this folder), one command to reproduce the headline result ([08-training-pipeline.md](08-training-pipeline.md) §5), tests, CI badge.
2. **A demo video** (even sim-only): 60–90 seconds showing (a) task succeeding normally, (b) a failure happening, (c) the system detecting and recovering, (d) the no-recovery baseline failing on the same scenario side-by-side. This side-by-side comparison clip is the single highest-leverage artifact for a portfolio — it makes the contribution legible in 10 seconds to someone who will never read the code.
3. **A short blog-post series** (one post per major phase, or one comprehensive post): written for a technical-but-not-necessarily-robotics-expert audience. Include the failures/debugging stories, not just the final numbers — "here's a dead end I hit and how I diagnosed it" is more convincing of real engineering skill than a clean success narrative.
4. **Optional: a workshop paper draft** (e.g., a robot-learning workshop at a venue like CoRL/ICRA/RSS) if the results are strong enough — even an unpublished draft PDF linked from the repo signals a research-caliber effort.

## 3. Target roles this demonstrates fit for

- **Humanoid robotics companies** (e.g., Figure, 1X, Apptronik, Sanctuary AI, Unitree itself, Boston Dynamics): direct domain match — you used their kind of hardware/sim stack on their kind of problem.
- **Robot learning research labs / research engineer roles**: the RL + evaluation rigor + related-work grounding maps directly onto research-engineer expectations.
- **General robotics/ML engineer roles**: the systems architecture, reproducibility tooling, and Docker/CI hygiene demonstrate production-adjacent engineering competence beyond "can train a model."

## 4. Resume bullet draft (edit once real results exist — don't lock in numbers before you have them)

> Designed and built a modular failure-detection and recovery system for humanoid manipulation/locomotion tasks in ManiSkill3 (targeting Unitree G1); implemented a learned failure monitor and an options-based recovery-skill library (RL + classical control), improving task success rate under injected failures by **[X] percentage points** over a no-recovery baseline across **[N]** tasks and **[M]** failure types, with a reproducible open-source benchmark suite.

## 5. Interview talking points to prepare

Be ready to go deep on, and defend, each of these — they are exactly the questions a strong interviewer will ask:

- **"Why modular instead of end-to-end?"** → [03-system-architecture.md](03-system-architecture.md) §5, and be honest about where the monolithic baseline ([10](10-evaluation-and-benchmarks.md) §2 item 4) actually won or came close, if it did.
- **"Why ManiSkill3 over Isaac Lab/MuJoCo?"** → [02-background-and-related-work.md](02-background-and-related-work.md) §5, with real friction points you actually hit, not just the a-priori justification.
- **"How do you know your detector isn't just overfit to your specific injected failures?"** → the held-out failure-type generalization ablation ([10](10-evaluation-and-benchmarks.md) §3) — have the actual number ready.
- **"What breaks on real hardware that doesn't in sim?"** → [09-sim-to-real-transfer.md](09-sim-to-real-transfer.md) §1, with specifics from your own system-identification findings if you got that far.
- **"What would you do differently / what's the biggest limitation?"** → have a real, specific answer, not a deflection — this question is a credibility test, and a thoughtful limitation is more convincing than pretending there isn't one.

## 6. Timing relative to job search

If there's a job-search deadline, **ship the Phase 4 (sim-only, end-to-end) deliverable as a complete, presentable project on its own** rather than waiting for hardware phases — per [11-roadmap-and-milestones.md](11-roadmap-and-milestones.md), the sim-only result is already a full, defensible piece of work, and a finished simulation project beats an unfinished hardware project every time in an interview setting.
