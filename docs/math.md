---
title: Mathematical Specification of Adaptive Task Recovery
status: active
last_updated: 2026-08-25
---

# Mathematical specification of adaptive task recovery

## 1. Purpose and claim boundary

This document defines the mathematical objects implemented by the repository,
connects each equation to executable code, and separates exact properties from
assumptions. The central problem is:

> Given a multi-goal instruction, a changing world, and hard instruction
> constraints, choose which remaining goals to attempt, skip, or defer and find
> an admissible execution route that maximizes legitimate completion.

The implementation is modular. Goal feasibility, selective decisions, symbolic
goal ordering, route-effect screening, and constrained navigation are executable
components, but there is not yet one joint optimizer that activates every
component simultaneously. Section 9 gives the composed execution semantics;
Section 10 gives a generalized objective that the modules approximate.

The mathematical scope is intentionally narrower than unrestricted language
understanding, general physical safety, or continuous-space optimal control.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $\mathcal O$ | finite set of named objects |
| $\mathcal G=\{g_1,\ldots,g_n\}$ | atomic instruction goals |
| $\mathcal C=\{c_1,\ldots,c_m\}$ | hard instruction constraints |
| $x_t$ | privileged world state at control step $t$ |
| $o_t$ | agent observation at step $t$ |
| $h_t=(o_0,a_0,\ldots,o_t)$ | observation/action history |
| $\kappa$ | intervention kind |
| $\tau$ | intervention onset step |
| $Y_i\in\{0,1\}$ | indicator that goal $g_i$ is achieved |
| $F_i(x_t)\in\{0,1\}$ | current oracle feasibility indicator |
| $p_{i,\kappa}$ | probability that an attempted goal survives through completion |
| $\pi$ | geometric route or sequence of semantic skills |
| $E(\pi,x_t)$ | objects predicted to be affected by $\pi$ |
| $A_t$ | set of goals already achieved at step $t$ |

For each object $o\in\mathcal O$, the privileged object state is

$$
x_t(o)=\bigl(e_t(o),\,p_t(o),\,u_t(o)\bigr),
$$

where $e_t(o)\in\{0,1\}$ is existence, $p_t(o)\in\mathbb R^3$ is
position when the object exists, and $u_t(o)\in\mathbb R^3$ is its optional
world-frame up vector. This is `ObjectState` in
`src/atr/feasibility/oracle.py`.

## 3. Instruction and goal graph

### 3.1 Atomic goals

Each goal is a tuple

$$
g_i=(\operatorname{id}_i,\rho_i,o_i,w_i,D_i,q_i),
$$

where:

- $\rho_i$ is the goal predicate; the implemented predicate is `on_tray`;
- $o_i\in\mathcal O$ is the target object;
- $w_i\in\mathbb Z_{\ge 0}$ is the stored priority;
- $D_i\subseteq\{\operatorname{id}_1,\ldots,\operatorname{id}_n\}$ is the
  dependency set;
- $q_i$ is either absent or an existence condition $(o,b)$, with
  $b\in\{0,1\}$.

The repository stores this structure in `Goal` and `GoalGraph` in
`src/atr/language/goal_graph.py`.

### 3.2 Current oracle feasibility

Let $q_i=\bot$ denote no condition. The implemented feasibility predicate is

$$
F_i(x_t)
=
\mathbf 1[e_t(o_i)=1]
\begin{cases}
1, & q_i=\bot,\\
\mathbf 1[e_t(o_q)=b_q], & q_i=(o_q,b_q).
\end{cases}
$$

This definition answers whether the required object exists and whether an
optional conditional goal is in play. It does **not** prove kinematic
reachability, graspability, path existence, controller success, or survival
until the end of an attempt.

Dependencies are deliberately separate from feasibility. Goal $g_i$ is
currently actionable when

$$
\operatorname{Act}_i(x_t,A_t)
=F_i(x_t)\prod_{d\in D_i}\mathbf 1[d\in A_t].
$$

Thus a goal whose prerequisite is unfinished is not actionable yet, but is not
declared permanently impossible.

### 3.3 Completion predicate

For an `on_tray` goal, let tray center be $r=(r_x,r_y,r_z)$, tray half-size
be $b=(b_x,b_y,b_z)$, and vertical completion margin be $\epsilon_z$.
The implemented completion indicator is

$$
Y_i(x_t)=\mathbf 1\!\left[
\begin{aligned}
&e_t(o_i)=1,\\
&|p_{t,x}(o_i)-r_x|\le b_x,\\
&|p_{t,y}(o_i)-r_y|\le b_y,\\
&-10^{-4}\le p_{t,z}(o_i)-r_z\le\epsilon_z
\end{aligned}
\right].
$$

The small negative lower tolerance handles floating-point mixing at exact tray
height. The stored vertical tray half-size $b_z$ does not enter the current
completion predicate; the independent $\epsilon_z$ bound does. Completion is
evaluated against the original goal object; moving an unauthorized substitute
never earns goal credit.

### 3.4 Hard constraints and oracle violations

A constraint is $c_j=(\operatorname{id}_j,k_j,o_j,\delta_j)$, where
$k_j\in\{\texttt{never\_move},\texttt{maintain\_orientation}\}$.
Given initial state $x_0$ and current state $x_t$, the implemented oracle
violation indicators are

$$
V_j(x_0,x_t)=
\begin{cases}
\mathbf 1[\|p_t(o_j)-p_0(o_j)\|_2>\delta_j],
& k_j=\texttt{never\_move},\\
\mathbf 1[u_{t,z}(o_j)<\delta_j],
& k_j=\texttt{maintain\_orientation}.
\end{cases}
$$

If the relevant current object state is absent, the current oracle returns no
violation for these pose checks. Object destruction is handled as feasibility,
not as a constraint violation. This is an implementation convention, not a
general safety principle.

## 4. Persistent changes and four distinct feasibility notions

An intervention $I=(\kappa,\tau,\theta)$ changes the transition process at
onset $\tau$, where $\theta$ contains kind-specific parameters. Persistent
changes need not be reversible within an episode.

The system distinguishes four questions that should not be collapsed:

1. **Existence feasibility:** $F_i(x_t)$, defined in Section 3.2.
2. **Execution survival:** whether a goal that is feasible now remains
   achievable through the duration of its attempt.
3. **Geometric reachability:** whether a collision-free route or controller
   execution exists.
4. **Instruction admissibility:** whether the intended action and its predicted
   side effects respect the original hard constraints.

For a goal perceived feasible at its decision point, define

$$
p_{i,\kappa}
=\Pr\!\left(
Y_i=1
\mid F_i(x_t)=1,\ \kappa,\ \text{attempt }g_i
\right).
$$

`calibrate_survival_estimates()` estimates this quantity separately for each
$(i,\kappa)$ stratum from real attempted rollouts. Conditioning on
$\kappa$ avoids pooling risk-free and risky mechanisms into one ambiguous
number. In the current implementation, $\kappa$ is privileged metadata.

## 5. Attempt, skip, and abstain

### 5.1 Implemented local reward

Let successful completion give $R_+=1$, failed execution consume $L$
steps at cost $c=0.1$ per step, skipping give zero, and abstaining for
$L_A$ steps cost $cL_A$. The implemented expected rewards are

$$
Q_{\mathrm{attempt}}(p)=pR_+-(1-p)cL,
$$

$$
Q_{\mathrm{skip}}=0,
\qquad
Q_{\mathrm{abstain}}=-cL_A.
$$

Successful attempts do not pay the step penalty in this current reward shape;
failed attempts do. This asymmetry must be retained when interpreting results.

Attempting has positive expected value exactly when

$$
p>p^*=\frac{cL}{R_++cL}.
$$

With $R_+=1$, $c=0.1$, and the default $L=25$,

$$
p^*=\frac{2.5}{3.5}=\frac57\approx0.7143.
$$

The forced decision rule is therefore

$$
d_{\mathrm{forced}}(\hat p)=
\begin{cases}
\mathrm{attempt},&\hat p>p^*,\\
\mathrm{skip},&\hat p\le p^*.
\end{cases}
$$

Priority $w_i$ does not enter this implemented calibration rule. A
priority-weighted extension would replace $R_+$ with a declared value
$v_i$; it must not be presented as current behavior.

### 5.2 Wilson uncertainty interval

For $s$ observed successes in $n$ Bernoulli trials, let
$\hat p=s/n$ and $z=1.959963984540054$. The implemented 95% Wilson
interval is

$$
\tilde p=
\frac{\hat p+z^2/(2n)}{1+z^2/n},
$$

$$
r=
\frac{z}{1+z^2/n}
\sqrt{\frac{\hat p(1-\hat p)+z^2/(4n)}{n}},
$$

$$
[p^-,p^+]=[\max(0,\tilde p-r),\min(1,\tilde p+r)].
$$

This interval assumes that trials are representative Bernoulli observations
within a fixed stratum. It does not protect against deployment shift, correlated
episodes, a wrong intervention label, or perceptual shortcut learning.

### 5.3 Selective rule

The selective decision uses the entire interval:

$$
d_{\mathrm{sel}}([p^-,p^+])=
\begin{cases}
\mathrm{attempt}, & Q_{\mathrm{attempt}}(p^-)>0,\\
\mathrm{skip}, & Q_{\mathrm{attempt}}(p^+)<0,\\
\mathrm{abstain}, & \text{otherwise}.
\end{cases}
$$

Because $Q_{\mathrm{attempt}}(p)$ is strictly increasing in $p$, an
attempt decision has positive expected value for every $p$ inside the
reported interval, and a skip decision has negative attempt value for every
$p$ inside the interval. This is interval-conditional robustness, not a
guarantee that the true deployment probability lies inside the interval.

For the default threshold and $n=20$:

| Successes | Wilson interval | Decision |
|---:|---:|---|
| 10 | $[0.2993,0.7007]$ | skip |
| 18 | $[0.6990,0.9721]$ | abstain |
| 20 | $[0.8389,1.0000]$ | attempt |

Abstention currently means waiting a fixed number of steps. It is not yet an
active information-gathering action.

## 6. Intent guard and predicted side effects

### 6.1 Semantic effect set

For a candidate action targeting object $o_a$ and route $\pi$, define

$$
\bar E(a,\pi,x_t)=\{o_a\}\cup E(\pi,x_t).
$$

The target is always treated as an effect. `affected_objects` adds incidental
objects predicted from route geometry.

For a candidate affected object $o$, define the current goal-target exemption

$$
T(o,x_t)=
\mathbf 1\!\left[
\exists g_i\in\mathcal G:
o_i=o\ \land\ F_i(x_t)=1
\right].
$$

The implemented pre-execution guard rejects an action if

$$
\exists o\in\bar E(a,\pi,x_t),\ \exists c_j\in\mathcal C:
T(o,x_t)=0,
\quad
k_j=\texttt{never\_move},
\quad
o_j=o.
$$

If a currently feasible goal explicitly requires the constrained object, the
goal-target exemption wins. The state-aware form is important: a conditional
goal that is not currently active does not exempt its object.

The pre-execution guard currently reasons about `never_move` only.
`maintain_orientation` is evaluated by the oracle after execution but is not
predicted by the route guard.

### 6.2 Swept-corridor effect predictor

For point $q$ and segment endpoints $a,b\in\mathbb R^3$, define

$$
\lambda^*=\operatorname{clip}_{[0,1]}
\frac{(q-a)^\top(b-a)}{\|b-a\|_2^2},
$$

$$
d(q,[a,b])=\|q-(a+\lambda^*(b-a))\|_2.
$$

For a zero-length segment, the implementation uses $d(q,[a,a])=\|q-a\|_2$.
For waypoint route $\pi=(v_0,\ldots,v_K)$, robot clearance $r_R$, and
object radius $r_o$, the predicted effect set is

$$
E(\pi,x_t)=
\left\{
o\in\mathcal O:
e_t(o)=1,
\min_{0\le k<K}d(p_t(o),[v_k,v_{k+1}])
\le r_R+r_o
\right\}.
$$

The intended target is excluded from this incidental set because it is already
included explicitly in $\bar E$.

Fetch navigation is planar. Before screening, each existing object center is
projected to the route travel height:

$$
\tilde p_t(o)=\bigl(p_{t,x}(o),p_{t,y}(o),z_{\mathrm{travel}}\bigr).
$$

Production object radii are conservative circumscribed XY radii derived from
the real collision vertices when available. The robot remains a constant-radius
disc approximation, so the predictor is not full-body collision checking.

### 6.3 Conditional guard soundness

Suppose:

1. every object physically displaced by executing $\pi$ is contained in
   $\bar E(a,\pi,x_t)$;
2. constraint and object identities are correct;
3. the executor follows the screened route within the assumed clearance model;
4. only `never_move` constraints are under consideration.

Then an action accepted by the state-aware guard cannot displace an object that
is protected by `never_move` and lacks a currently feasible goal-target
exemption. The proof is immediate by contradiction: any such displaced object
would belong to $\bar E$, satisfy the rejection predicate, and therefore make
the action unacceptable.

This is a conditional software property. A missed effect, perception error,
controller deviation, wrong radius, or unmodeled orientation effect invalidates
the premises.

## 7. Discrete navigation and constrained replanning

Let an occupancy grid contain free cells $V$ and eight-neighbor edges
$E_G$. For neighboring grid cells $u,v$, the edge cost is Euclidean grid
distance:

$$
c(u,v)=\Delta\sqrt{(i_u-i_v)^2+(j_u-j_v)^2},
$$

where $\Delta$ is cell size. `plan_path()` applies Dijkstra's algorithm
between the nearest free start and goal cells. With nonnegative edge weights,
the returned route is shortest on this discretized graph. This says nothing
about optimality in continuous configuration space.

If the nominal route predicts protected effects $H\subseteq\mathcal O$, the
constrained occupancy grid is

$$
M'(q)=M(q)\lor
\bigvee_{o\in H}
\mathbf 1\left[
\|q_{xy}-p_{t,xy}(o)\|_2\le r_R+r_o
\right].
$$

The recovery sequence is:

1. find a nominal grid route $\pi_0$;
2. compute $E(\pi_0,x_t)$ and apply the intent guard;
3. if accepted, execute $\pi_0$;
4. if rejected, inflate the predicted protected objects into $M'$;
5. find $\pi_1$ on $M'$ and independently screen it;
6. execute $\pi_1$ only if it is accepted; otherwise stop;
7. verify measured arrival before manipulation receives goal credit.

Replanning can recover completion that a stop-only guard loses: if the nominal
route is rejected, a constrained route exists, screening is sound, and the
controller succeeds, replanning completes a goal that stop-only necessarily
abandons while satisfying the same modeled constraint. This conclusion remains
conditional on route existence and controller success.

## 8. Symbolic partial-goal planning

Let $R_t=\mathcal G\setminus A_t$ be remaining goals. The implemented
symbolic planner enumerates every permutation $\sigma$ of $R_t$. It scans
the permutation from left to right and constructs $S_\sigma$: a goal is added
when it is feasible and all dependencies are in $A_t\cup S_\sigma$.

The score is

$$
J(\sigma)=\sum_{g_i\in S_\sigma}(w_i+1).
$$

The selected order is

$$
\sigma^*\in\arg\max_\sigma J(\sigma).
$$

The $+1$ gives priority-zero goals positive value. The episode driver attempts
the first selected goal, observes the outcome, adds successful goals to
$A_t$, and solves again. This is intended to provide receding-horizon symbolic
replanning.

For $r=|R_t|$, exhaustive enumeration is $O(r!\,r)$. It is exact for the
implemented discrete score at the repository's small goal counts, but does not
scale to large goal sets. During scoring, a selected goal is treated as if it
will succeed; execution failure is handled only by the next replanning cycle.

There is currently no retry budget and `plan()` excludes achieved goals rather
than all previously attempted goals. Consequently, if an attempted goal fails
while its target remains perceived feasible, `run_replanner_episode()` can
select it repeatedly and is not guaranteed to terminate. Existing execution
tests cover successful attempts and infeasible-goal skips, not this failure
branch. A termination guarantee requires either a finite retry counter, an
attempted-goal exclusion set, or a belief update that makes the failed action
temporarily or permanently unavailable.

## 9. Composed recovery semantics

The current modules define the following coherent execution procedure. Some
entry points use only subsets of these steps; this procedure is the composition
contract, not a claim that one function already invokes every branch.

```text
Input: goal graph, current state/observations, achieved set, calibration data

while an unprocessed goal remains:
    update current feasibility or feasibility belief
    choose a dependency-valid goal order
    select the next candidate goal

    if the goal is currently infeasible:
        record a zero-cost skip
        continue

    apply the forced or interval-selective survival rule
    if skip: record skip and continue
    if abstain: wait, record abstention, and continue

    compute the nominal route or skill trajectory
    predict affected objects and apply the intent guard
    if rejected: search for and re-screen a constrained route
    if no accepted route exists: record a safety block or navigation failure
    else: execute, verify arrival/completion, update achieved goals
```

This pseudocode requires an explicit retry/termination rule on execution
failure. That rule is part of the composition contract but, as described in
Section 8, is missing from the current symbolic episode driver.

Three different negative outcomes remain distinct:

- **skip:** the strategy deliberately declines a goal;
- **safety block:** a candidate route conflicts with instruction constraints;
- **navigation failure:** no executable route or arrival was obtained.

Conflating these outcomes would make a system that is safe only because it
cannot move appear equivalent to one that finds a legitimate recovery.

## 10. Generalized constrained objective

A unified decision problem can be written over candidate semantic actions
$a$, routes $\pi$, and decision modes
$d\in\{\mathrm{attempt},\mathrm{skip},\mathrm{abstain}\}$:

$$
\max_{d,a,\pi}
\quad
\sum_{i=1}^{n}v_i\,\mathbb E[Y_i]
-c_{\mathrm{step}}\,\mathbb E[L]
-c_A L_A\mathbf 1[d=\mathrm{abstain}]
$$

subject to

$$
\operatorname{Act}_i(x_t,A_t)=1
\quad\text{for any attempted }g_i,
$$

$$
\operatorname{Guard}(a,\pi,x_t,\mathcal G,\mathcal C)=1,
$$

$$
\Pr\!\left(\bigvee_{j=1}^{m}V_j=1\right)\le\epsilon.
$$

This generalized expression is useful for analysis, but the repository does
not currently solve this chance-constrained optimization jointly. In current
code:

- the local survival rule uses $v_i=1$ and the asymmetric reward in Section 5;
- symbolic planning uses $v_i=w_i+1$ but not survival probabilities;
- `never_move` route admissibility is a deterministic guard over predicted
  effects, not a learned probability bounded by $\epsilon$;
- observation-based feasibility and privileged-state feasibility are evaluated
  in different experimental paths.

The main algorithmic extension needed for a genuinely unified solver is a
single belief state and objective shared by goal ordering, selective decisions,
and route constraints.

## 11. Evaluation metrics

For episode $s$, let $r_{s,i}$ be the recorded result for goal $i$.
The implemented aggregate outcomes include

$$
G_s=\sum_i\mathbf 1[r_{s,i}.\mathrm{achieved}],
$$

$$
L_s=\sum_i r_{s,i}.\mathrm{steps\_used},
$$

$$
W_s=\sum_i r_{s,i}.\mathrm{steps\_used}\,
\mathbf 1[\neg r_{s,i}.\mathrm{achieved}]
\mathbf 1[\neg r_{s,i}.\mathrm{skipped}].
$$

Thus `wasted_steps` counts unsuccessful non-skipped execution, including
failed attempts, but not deliberate zero-cost skips.

Navigation counts are

$$
R_s=\sum_i\mathbf 1[r_{s,i}.\mathrm{navigation\_replanned}],
$$

$$
B_s=\sum_i\mathbf 1[\text{route screened, blocked, and skipped}],
$$

$$
N_s=\sum_i\mathbf 1[r_{s,i}\text{ has a navigation failure reason}].
$$

The scalable evaluator currently extracts `constraint_violations` by counting
top-level outcome keys whose names end in `_violated`. This is exact only when
the policy adapter surfaces every relevant oracle violation in that form. It is
not automatically the same as $\sum_j V_j$, and this limitation should be
resolved before treating the metric as complete constraint coverage.

For selective decisions $d_1,\ldots,d_N$ and correct binary actions
$d_1^*,\ldots,d_N^*$, let

$$
I=\{i:d_i\ne\mathrm{abstain}\}.
$$

Coverage and selective risk are

$$
\operatorname{coverage}=\frac{|I|}{N},
\qquad
\operatorname{risk}=\frac{1}{|I|}\sum_{i\in I}\mathbf 1[d_i\ne d_i^*].
$$

When $|I|=0$, code reports risk zero, so risk must always be shown together
with coverage.

## 12. Paired statistical estimation

Every compared policy is evaluated on the same case identities. For policy
$m$, reference policy $m_0$, metric $Z$, and case $s$, define

$$
\Delta_{s,m}=Z_{s,m}-Z_{s,m_0}.
$$

The paired estimator is

$$
\bar\Delta_m=\frac1N\sum_{s=1}^{N}\Delta_{s,m}.
$$

The evaluator resamples the $N$ paired deltas with replacement, computes a
mean for every resample, and reports percentile endpoints. If
$\bar\Delta_m^{*(1)},\ldots,\bar\Delta_m^{*(B)}$ are bootstrap means, a
two-sided $(1-\alpha)$ interval is

$$
\left[
Q_{\alpha/2}(\bar\Delta_m^*),
Q_{1-\alpha/2}(\bar\Delta_m^*)
\right].
$$

The default is $B=2000$, $1-\alpha=0.95$, and deterministic bootstrap
seed zero. Overall summaries are accompanied by strata indexed by environment,
scene variant, and condition. The current estimator is not hierarchical and
does not itself correct for multiple exploratory comparisons.

Strict validation requires exactly one completed record for every
case-policy pair before aggregation. Missing, extra, duplicated, or unpaired
records cause failure instead of silent averaging.

## 13. Deterministic experiment identity and sharding

A case identity contains

$$
(\text{environment},\text{scene},\text{intervention},
\text{onset range},\text{seed},\text{condition}).
$$

Canonical sorted JSON is hashed with SHA-256 and truncated to 20 hexadecimal
characters for `case_id`. The manifest fingerprint is a separately truncated
SHA-256 digest over the complete normalized specification. Duplicate expanded
case IDs are checked explicitly.

For $J$ execution shards, a case is assigned by

$$
\operatorname{shard}(s)=\operatorname{int}(\operatorname{case\_id}_s,16)\bmod J.
$$

Every policy for one case stays in the same shard, preserving paired execution.
The frozen v1 configuration expands to

$$
700+500+500+1500=3200\text{ cases},
$$

and four policies produce

$$
3200\times4=12800\text{ policy episodes}.
$$

## 14. Worked recovery example

Consider two placement goals, `place_mug` and `place_bowl`, plus a
`never_move(glass)` constraint.

1. The bowl is destroyed at $\tau$, so
   $F_{\mathrm{bowl}}(x_t)=0$. Its goal is skipped with zero goal credit.
2. The mug still exists, so $F_{\mathrm{mug}}(x_t)=1$.
3. If the mug survival estimate is $18/20$, its Wilson interval crosses the
   default threshold, so the selective rule abstains. With $20/20$, it
   attempts.
4. The nominal mug route passes within the modeled swept radius of the glass,
   so `glass` enters $E(\pi_0,x_t)$.
5. The glass is not a currently feasible goal target and is protected by
   `never_move`; therefore the guard rejects $\pi_0$.
6. The constrained planner inflates the glass into the occupancy grid and
   searches for $\pi_1$.
7. If $\pi_1$ is found, independently screened, executed, and verified, the
   mug earns completion while the bowl remains honestly incomplete. If no
   route is found, the system reports a block or navigation failure rather than
   moving the glass or crediting a substitute.

This example captures the intended distinction between an impossible goal, a
difficult route, and an inadmissible recovery.

## 15. Exact properties and non-guarantees

### Properties supported by code

1. Goal completion is credited only to the original target object.
2. State-aware guarding checks predicted incidental objects as well as the
   named action target for `never_move` constraints.
3. Selective attempt/skip decisions have one sign of expected value throughout
   the reported Wilson interval.
4. Dijkstra returns a shortest route on the constructed finite grid graph.
5. Constrained replanning does not mutate the cached nominal occupancy grid.
6. Case expansion, identity, and shard assignment are deterministic for a fixed
   manifest.
7. Aggregation rejects incomplete and unpaired result sets.

### Properties not established

1. Global optimality of the continuous robot trajectory.
2. Complete collision safety for the robot body, arm, grasped object, or
   deformable contacts.
3. Pre-execution enforcement of orientation constraints.
4. Correctness of learned or crop-based visual feasibility outside evaluated
   conditions.
5. Calibration validity under a changed timing, scene, object, embodiment, or
   intervention distribution.
6. Statistical independence of all simulator episodes.
7. Generalization from simulation to physical hardware.
8. A joint optimum over goal ordering, uncertainty, semantic constraints, and
   motion.
9. Termination of the symbolic episode driver after a feasible execution
   failure.

These boundaries are part of the mathematical specification: a conditional
property should never be reported as an unconditional system guarantee.

## 16. Code correspondence

| Mathematical component | Implementation |
|---|---|
| Goal graph and dependencies | `src/atr/language/goal_graph.py` |
| Feasibility, completion, oracle violations | `src/atr/feasibility/oracle.py` |
| Survival calibration and selective rule | `src/atr/feasibility/calibrated_feasibility.py` |
| Intent guard | `src/atr/constraints/intent_guard.py` |
| Swept-corridor effects | `src/atr/constraints/effect_predictor.py` |
| Grid planning, screening, constrained detour | `src/atr/envs/navigation.py` |
| Static and feasibility-aware policies | `src/atr/policies/baselines.py` |
| Symbolic goal ordering | `src/atr/policies/symbolic_replanner.py` |
| Bootstrap evaluation | `src/atr/evaluation/harness.py` |
| Scaled execution and paired aggregation | `src/atr/evaluation/benchmark_suite.py` |
| Frozen experiment matrix | `configs/benchmark_v1.json` |
