# Todo

Last updated: 2026-07-26

## Ownership

- **Shared:** benchmark, definitions, interfaces, data splits, integration,
  evaluation, and research claims.
- **Person A:** vision/language model selection, self-supervised representation,
  goal graphs, change/feasibility inference, calibration, and abstention.
- **Person B:** simulator/humanoid integration, low-level skills, RL policies,
  intent guard, and policy baselines.

## Now — shared foundation

- [ ] **Shared:** Freeze v0 schemas for goals, constraints, priorities,
  interventions, feasibility beliefs, humanoid skills, and episode logs.
- [ ] **Shared:** Create `src/atr/`, `configs/`, `scripts/`, `tests/`, and data manifests.
- [ ] **Shared:** Define the first multi-goal task, controlled instruction grammar,
  and irreversible/reversible/neutral interventions.
- [ ] **Shared:** Define benchmark splits, leakage checks, primary metrics, and
  integration acceptance tests.
- [ ] **Shared:** Choose dependency management, experiment tracking, and compute budgets.
- [ ] **Shared:** Redraw the architecture diagram with owners and API boundaries.

## Now — parallel spikes

- [ ] **Person A:** Shortlist visual encoders, self-supervised objectives, and
  language/goal parsing approaches.
- [ ] **Person A:** Build a model-selection matrix covering downstream feasibility,
  calibration, paraphrase/generalization, latency, memory, licensing, and integration cost.
- [ ] **Person A:** Prototype the `ObservationWindow + instruction -> AgentBelief` API
  against recorded or synthetic trajectories.
- [ ] **Person B:** Compare humanoid-capable simulators for visual access,
  interventions, deterministic replay, oracle planning, and controller availability.
- [ ] **Person B:** Select a humanoid asset and validate navigate, reach, grasp,
  place, inspect, and safe-stop skills.
- [ ] **Person B:** Prototype the `AgentBelief + available skills -> SkillCall` API
  using oracle feasibility beliefs.

## Next — benchmark and baselines

- [ ] **Shared:** Implement and unit-test the benchmark generator, interventions,
  oracle feasibility planner, and constraint checker.
- [ ] **Shared:** Produce one deterministic end-to-end humanoid episode and a
  versioned trajectory dataset for both workstreams.
- [ ] **Person A:** Collect/finalize unlabeled visual data splits and train initial
  frozen, fine-tuned, image-SSL, and temporal-SSL baselines.
- [ ] **Person A:** Train calibrated per-goal change and feasibility models.
- [ ] **Person B:** Train the static language-conditioned and oracle-feasibility policies.
- [ ] **Person B:** Implement the intent guard and policy evaluation harness.
- [ ] **Shared:** Replace oracle feasibility with Person A's learned beliefs and
  pass contract, latency, shape, calibration, and end-to-end smoke tests.

## Later — integrated research

- [ ] **Person B:** Train feasibility-conditioned and matched monolithic adaptive policies.
- [ ] **Person A:** Run representation, history, factorization, and uncertainty ablations.
- [ ] **Person B:** Run policy, guard, oracle-skill, and humanoid-execution ablations.
- [ ] **Shared:** Run held-out composition, intervention, object, layout, and paraphrase tests.
- [ ] **Shared:** Diagnose failures by perception, feasibility, strategy, guard,
  and humanoid skill execution.
- [ ] **Shared:** Publish multi-seed results, configs, checkpoints where permitted,
  failure cases, benchmark generator, and demos.

## Completed

- [x] Replace the old humanoid physical-recovery research direction.
- [x] Rewrite stable design documents around the new research question.
- [x] Make simulated-humanoid evaluation an explicit requirement.
- [x] Define the two-person ownership model and shared-integration principle.
