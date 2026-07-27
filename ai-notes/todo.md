# Todo

Last updated: 2026-07-26

## Now

- [ ] Redraw the architecture diagram and retire superseded media.
- [ ] Create `src/atr/`, `configs/`, `scripts/`, `tests/`, and data manifests.
- [ ] Define typed schemas for goals, constraints, priorities, and interventions.
- [ ] Compare candidate environments on visual access, intervention control,
  deterministic replay, language support, oracle planning, and humanoid support.
- [ ] Select a humanoid asset and validate navigation, reach, grasp, place,
  inspect, and safe-stop skill interfaces.
- [ ] Choose dependency management and experiment tracking.

## Next

- [ ] Implement one multi-goal task and controlled instruction grammar.
- [ ] Implement irreversible, reversible, and neutral interventions.
- [ ] Build and unit-test the oracle feasibility/constraint checker.
- [ ] Train a static language-conditioned baseline.
- [ ] Collect unlabeled visual trajectories with frozen dataset splits.
- [ ] Select and train initial self-supervised visual baselines.

## Later

- [ ] Train calibrated per-goal feasibility models.
- [ ] Train feasibility-conditioned and monolithic adaptive policies.
- [ ] Add intent shielding and invalid-substitution checks.
- [ ] Run held-out composition, intervention, object, and paraphrase tests.
- [ ] Publish multi-seed results, failure cases, configs, and benchmark generator.

## Completed

- [x] Replace the old humanoid recovery research direction.
- [x] Rewrite stable design documents around the new research question.
- [x] Preserve simulation-only scope and separate stable/live documentation.
- [x] Make simulated-humanoid evaluation an explicit project requirement.
