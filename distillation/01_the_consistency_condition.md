# The Consistency Condition

## The Question

A consistency model replaces the whole sampling trajectory with one
function: `f(x, t) =` the PF-ODE solution map from `(x, t)` down to
`t_{\min}`. Training never sees that map — it only enforces LOCAL
agreement between adjacent points of trajectories. This file proves
the two theorems that make the local-to-global leap sound: the
consistency condition plus the boundary condition CHARACTERIZE the
solution map (uniqueness, via characteristics), and approximate
consistency implies approximate correctness with error accumulating
LINEARLY along the trajectory (the telescope/Grönwall bound) — the
family's budget equation, which `02`–`03` spend.

## The Object And The Characterization

PF-ODE `\dot x = v(x, t)` (foundations/03; in `(u, \rho)` form,
`samplers_and_convergence/02`), trajectories on `[t_{\min}, T]`,
well-posed (Lipschitz `v` — the audit carries it). Define the exact
solution map `f^*(x, t) :=` the value at `t_{\min}` of the trajectory
through `(x, t)`. Two defining properties:

```text
(SC)  self-consistency:  f(x(t), t) is constant along trajectories;
(B)   boundary:          f(x, t_min) = x.
```

**Theorem (characterization).** A function satisfies (SC) and (B) iff
it is `f^*`. In differential form: (SC) for `C^1` functions is the
transport equation

```math
\partial_t f + (v\cdot\nabla)\,f = 0,
```

and `f^*` is its unique solution with boundary data (B).

*Proof.* (`f^*` satisfies both): constancy along trajectories is the
definition; (B) is the trivial trajectory. (Uniqueness): the
characteristics of the transport equation are exactly the ODE
trajectories — along any trajectory,
`\frac{d}{dt}f(x(t), t) = \partial_t f + v\cdot\nabla f = 0` — so any
solution is constant along each trajectory and hence equals its value
at the trajectory's endpoint `(x(t_{\min}), t_{\min})`, which (B)
pins to `x(t_{\min}) = f^*(x, t)`. Well-posedness supplies a unique
trajectory through every point. ∎

The theorem is what licenses the method: consistency is not a proxy
objective that happens to work — its exact solutions ARE the object,
with no slack. Everything interesting is therefore in the
approximation, which is the second theorem.

## The Accumulation Theorem

Training enforces (SC) on a grid `t_{\min} = t_0 < t_1 < \dots < t_N
= T`, imperfectly.

**Theorem (linear accumulation).** Suppose `f` satisfies (B) exactly
(free, below) and, along every trajectory, adjacent-grid consistency
holds to tolerance `\varepsilon_k`:
`\|f(x(t_k), t_k) - f(x(t_{k+1}), t_{k+1})\| \le \varepsilon_k`. Then
for every trajectory point,

```math
\big\|f(x(t_N), t_N) - f^*(x(t_N), t_N)\big\| \;\le\; \sum_{k=0}^{N-1}\varepsilon_k .
```

*Proof.* Telescope down the same trajectory:

```math
f(x(t_N), t_N) - \underbrace{x(t_{\min})}_{=f^*}
= \sum_{k=0}^{N-1}\big[f(x(t_{k+1}), t_{k+1}) - f(x(t_k), t_k)\big]
+ \big[\underbrace{f(x(t_0), t_0) - x(t_{\min})}_{=0\ \text{by (B)}}\big],
```

triangle inequality. ∎ (Continuum version: with defect
`\delta := \partial_t f + v\cdot\nabla f`, the same telescoping is an
integral and gives error `\le \int\|\delta\|` along the trajectory —
Grönwall enters only if one wants the bound under a perturbed FIELD
as well, where Lipschitzness of `f` in `x` converts trajectory error
to output error.)

Read it as the family's budget: **`N` segments means `N` chances to
leak, and the leaks ADD.** No contraction cleans consistency error —
the map is deterministic transport, and this is
`samplers_and_convergence/05`'s ODE-transports-error theorem
reappearing at the level of the learned map itself. Consequences the
literature knows empirically, here as corollaries: fine grids do not
automatically help (smaller per-step `\varepsilon_k`, more of them —
the tradeoff `02` prices for each training scheme); and the
quality of a consistency model is bottlenecked by its WORST noise
band, since every sample's error passes through all bands.

## The Boundary For Free

**Proposition.** Parametrize

```math
f_\theta(x, t) \;=\; c_{\mathrm{skip}}(t)\,x + c_{\mathrm{out}}(t)\,F_\theta(x, t),
\qquad
c_{\mathrm{skip}}(t_{\min}) = 1,\ \ c_{\mathrm{out}}(t_{\min}) = 0 :
```

then (B) holds IDENTICALLY for every `\theta`. *Proof.* Evaluate. ∎
Trivial and load-bearing: the accumulation theorem charged the
boundary term at zero because this parametrization (the EDM-style
skip scaling, used by every consistency model) makes it structurally
zero — one of the two defining properties is compiled into the
architecture, and training's entire budget goes to (SC).

The right mental model, connecting to phase A: the teacher's few-step
samplers approximate the quadrature
`\int \hat x_0(\rho)\rho^{-2}d\rho` rule by rule
(`samplers_and_convergence/02`); the consistency model LEARNS THE
INTEGRAL as a function of its upper limit. Solvers pay per
evaluation at sampling time; consistency models pay once, at
training time, in the accumulation theorem's currency.

## Load-Bearing Audit

```text
well-posedness of the ODE     characterization needs unique
                              trajectories — inherited from
                              flow_matching/01's audit, with the same
                              atomic-data caveat at t_min;
(B) exact                     the telescope's zero endpoint; the
                              proposition is why it is safe to assume;
same-trajectory evaluation    the theorem bounds error along TEACHER
                              trajectories; training evaluates (SC)
                              at sampled points — the sampling
                              distribution mismatch is part of 02's
                              estimator analysis;
determinism                   no contraction, hence linear
                              accumulation; a stochastic student
                              (noise-injecting few-step CM) re-enters
                              A/05's tradeoff (statement).
```

## Position In The Coordinate System

A new point in the solver coordinate: `S` = a single learned
evaluation of the solution map, with exactness characterized and
error budgeted. The estimand has quietly changed too — from the score
(local) to the flow map (global) — and the accumulation theorem is
the price of the globalization: local errors in a local estimand
average out under SDE sampling; local errors in a global estimand
add.

## What Remains Open

Whether (SC) enforced at SAMPLED point pairs controls (SC) along
trajectories uniformly (a distribution-shift question between the
training distribution and the trajectory ensemble — assumed away in
every current analysis including this one); multistep consistency
(2–4 step models: the accumulation theorem suggests optimal segment
PLACEMENT, nobody has derived it); and the right regularity class for
`f^*` on rough data — near `t_{\min}` the map's Lipschitz constant
inherits the score blowup (`statistical_theory/04`), and the
approximation-theoretic cost of representing `f^*` there is the
honest reason consistency models are hard to train at high
resolution.
