# Rectified Flow

## The Question

`01`'s marginal flow forgets the coupling: `p_t` sees only conditional
MEANS of velocities, so the flow's own endpoint pairing
`(z_0, z_1)` generally differs from the coupling `\Pi` it was built
from. Rectification (Liu–Gong–Liu) makes that a feature: use the
flow's pairing as a NEW coupling, re-interpolate, repeat. This file
proves the three theorems that make the loop meaningful — marginal
preservation, non-increase of every convex transport cost, and
straight couplings as fixed points — and then does something the
papers gloss: exhibits exactly where the first theorem's hypotheses
FAIL (atomic crossing data) and proves, by a symmetry argument, what
the smoothed flow does there instead: it un-crosses the coupling.

Throughout: linear interpolant `x_t = (1-t)x_0 + t x_1`, conditional
velocity `x_1 - x_0`, marginal field `v^*` as in `01`; the RECTIFIED
coupling `\Pi' = \mathrm{Law}(z_0, z_1)` where `z` solves
`\dot z = v^*(z, t)`, `z_0 = x_0 \sim p_0`.

## Theorem 1: Marginals Are Preserved

**Theorem.** If the flow is well-posed (unique solutions through
`p_t`-a.e. point), then `z_t \sim p_t` for all `t`; in particular
`z_1 \sim p_1`: rectification changes the coupling, never the
endpoint laws. *Proof.* `01`'s Theorem 1: `v^*` solves the continuity
equation for `p_t`; under well-posedness the flow's pushforward is
the unique solution of that equation with initial value `p_0`. ∎

## Theorem 2: Every Convex Cost Weakly Decreases

**Theorem.** For every convex `c: R^d \to R`,

```math
\mathbb{E}\,c\big(z_1 - z_0\big) \;\le\; \mathbb{E}\,c\big(x_1 - x_0\big):
```

the rectified coupling's transport cost is no larger than the
original's — simultaneously for ALL convex costs.

*Proof.* Along the flow,
`z_1 - z_0 = \int_0^1 v^*(z_t, t)\,dt = \int_0^1
E[x_1 - x_0\,|\,x_t = z_t]\,dt`. Apply Jensen twice:

```math
c\big(z_1 - z_0\big)
\;\le\; \int_0^1 c\big(\mathbb{E}[x_1 - x_0\,|\,x_t = z_t]\big)\,\mathrm{d}t
\;\le\; \int_0^1 \mathbb{E}\big[c(x_1 - x_0)\,\big|\,x_t = z_t\big]\,\mathrm{d}t
```

(first over the time average, second over the conditional law). Take
expectations over the flow; by Theorem 1, `z_t \sim p_t`, so the
right side integrates the tower property to
`\int_0^1 E[c(x_1 - x_0)]\,dt = E\,c(x_1 - x_0)`. ∎

**Theorem 3 (fixed points).** If the coupling is STRAIGHT — the
displacement `x_1 - x_0` is a.s. determined by `x_t` for every `t`
(paths do not merge) — then `v^*(x_t) = x_1 - x_0` along each path,
the flow retraces the interpolants exactly, and `\Pi' = \Pi`; both
Jensen steps hold with equality. Conversely, equality for a strictly
convex `c` forces the conditional displacement to be a.s. constant in
both averagings — straightness. Straight couplings are exactly the
fixed points, and on them ONE Euler step of the flow is exact
(`z_1 = z_0 + v^*`): rectification is a distillation method
(phase D's neighbor) as much as a transport method. ∎

## Where Theorem 1 Fails, Exactly

The well-posedness hypothesis is not decoration. Take the atomic
crossing coupling in one dimension:
`x_0 \in \{-1, +1\}` uniform, `x_1 = -x_0`. The two interpolation
lines cross at `(t, x) = (\tfrac12, 0)`. The marginal field on the
atoms is `+2` on the rising line, `-2` on the falling line, and `0`
at the crossing instant (the two atoms coincide and their velocities
average). A flow trajectory arriving at `x = 0` at `t = \tfrac12`
carries mass `1` (both atoms), and ANY selection of a continuation
sends all of it up or all of it down — the output law becomes a
single atom, not `p_1 = \tfrac12\delta_{-1} + \tfrac12\delta_{+1}`:
**marginal preservation genuinely fails**, because at the crossing
the ODE has no uniqueness and the continuity equation has multiple
solutions. This is `01`'s audit item made concrete, and it is the
same singularity as the empirical-score sharpening of
`score_foundations/06`: atomic data makes the transport field
ill-posed exactly where paths collide.

**The smoothed resolution, proved.** Regularize with any symmetric
smoothing (a `\gamma > 0` latent, `02`, or smoothed atoms). The setup
is symmetric under `x \mapsto -x` (both `p_0` and the coupling), so
the marginal field is ODD: `v^*(-x, t) = -v^*(x, t)`, hence
`v^*(0, t) = 0` and — by uniqueness, now available — **no trajectory
crosses the origin**. Sign is conserved along the flow. Therefore the
rectified coupling transports the negative half of `p_0` to the
negative half of `p_1` and likewise for the positive halves: the
crossing coupling has been replaced by the NON-crossing (monotone)
one, with squared cost dropping from `E|x_1 - x_0|^2 = 4` to
`O(\text{smoothing})` — Theorem 2's inequality, achieved with a
vengeance, and Theorem 3's fixed point reached in a single
rectification. ∎ (The general `k`-rectification convergence toward
straight couplings is Liu et al.'s; statement. The example shows the
mechanism: rectification un-crosses paths, and un-crossed is
straight is one-step-samplable.)

## Load-Bearing Audit

```text
well-posedness            Theorem 1 and everything downstream; the
                          crossing example is its exact failure mode,
                          and smoothing is the repair — the field's
                          practice (train on data + tiny noise)
                          matches the theorem's need;
z_t ~ p_t inside Thm 2    the subtle dependence: the cost proof
                          consumes Theorem 1 — no marginal
                          preservation, no cost theorem;
convexity                 both Jensen steps; strictness for the
                          fixed-point converse;
linear interpolant        the conditional velocity x_1 - x_0 is
                          time-constant, which is what makes
                          "straight" the fixed-point notion; curved
                          interpolants have analogous statements with
                          "geodesic of the interpolant" replacing
                          "straight" (statement).
```

## Position In The Coordinate System

The coupling — silent in `01`–`02` — becomes the object: the flow is
a map `\Pi \mapsto \Pi'` on couplings that fixes marginals, weakly
contracts every convex cost, and has the straight couplings as its
fixed set. Rectification is thus a solver-improvement loop living
entirely in the `P` coordinate, and its payoff is a `S`-coordinate
one: straightness = one-step exactness. `05` prices the tempting
over-read (straight `\ne` optimal).

## What Remains Open

Convergence RATES for iterated rectification (how many reflows to
`\epsilon`-straightness, as a function of the data — only qualitative
results exist); the learned-field version of all three theorems
(practice rectifies with an imperfect `v_\theta`: Theorem 2's
inequality then holds only up to estimation error, and the loop can
DEGRADE — the accumulation is measured empirically, unpriced
theoretically); and the crossing example's high-dimensional
generalization: how much smoothing suffices for well-posedness on
`n`-atom data — the same `\gamma`-profile question `02` closed on,
now with a transport-topology reason to care.
