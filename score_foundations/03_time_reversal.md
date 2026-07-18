# Time Reversal

## The Question

`01` destroys data; sampling requires running the destruction
backward. Anderson's 1982 theorem says the time-reversed process is
again a diffusion, with a drift correction equal to `g^2` times the
score — so the ONE function `02` made learnable is the ONLY unknown
in the generative direction. This file proves the reversal at the
Fokker–Planck level, proves the probability-flow ODE shares every
marginal, and proves the one-parameter family of samplers between
them — the design space every solver in phase A lives in.

## The Fokker–Planck Computation

Forward SDE `dx = f(x,t)\,dt + g(t)\,dW` with marginals `p_t`
(assumed positive and `C^2`; the audit prices this). Its
Fokker–Planck equation:

```math
\partial_t p \;=\; -\nabla\!\cdot\!\big(f\,p\big) + \tfrac{g^2}{2}\,\Delta p .
```

**Theorem (Anderson, FP-level proof).** Define the reverse-time
process on `\tau = T - t` by

```math
\mathrm{d}x \;=\; \big[-f + g^2\,\nabla\log p_{T-\tau}\big]\mathrm{d}\tau + g\,\mathrm{d}\bar W_\tau .
```

Its marginals are `q_\tau = p_{T-\tau}`: the reverse SDE retraces the
forward marginals exactly, ending at `q_T = p_0`.

*Proof.* It suffices to check that `q_\tau := p_{T-\tau}` solves the
Fokker–Planck equation of the displayed SDE. Left side:
`\partial_\tau q = -\partial_t p|_{T-\tau} = \nabla\cdot(fp) -
\tfrac{g^2}{2}\Delta p`. Right side, with drift
`\tilde f = -f + g^2\nabla\log p`:

```math
-\nabla\cdot(\tilde f q) + \tfrac{g^2}{2}\Delta q
= \nabla\cdot(f p) - g^2\,\nabla\cdot\big(p\,\nabla\log p\big) + \tfrac{g^2}{2}\Delta p ,
```

and `p\,\nabla\log p = \nabla p`, so the middle term is
`-g^2\Delta p`; the right side totals
`\nabla\cdot(fp) - g^2\Delta p + \tfrac{g^2}{2}\Delta p
= \nabla\cdot(fp) - \tfrac{g^2}{2}\Delta p` — equal to the left
side. ∎

One identity — `p\nabla\log p = \nabla p` — carries the whole
theorem: the score is exactly the object that converts "diffuse
forward" into "concentrate backward." (Anderson's original statement
is at the SDE/pathwise level with its own regularity bookkeeping;
the FP-level proof above is the marginal statement, which is what
sampling needs, and the audit says so honestly.)

## The Probability-Flow ODE, Proved

**Theorem.** The deterministic dynamics

```math
\frac{\mathrm{d}x}{\mathrm{d}t} \;=\; f(x,t) - \tfrac{g^2}{2}\,\nabla\log p_t(x)
```

has the SAME marginals `p_t` as the forward SDE (run in either
direction).

*Proof.* A deterministic flow with velocity `v` transports its
density by the continuity equation `\partial_t p = -\nabla\cdot(vp)`.
With `v = f - \tfrac{g^2}{2}\nabla\log p`:

```math
-\nabla\cdot(vp)
= -\nabla\cdot(fp) + \tfrac{g^2}{2}\nabla\cdot(\nabla p)
= -\nabla\cdot(fp) + \tfrac{g^2}{2}\Delta p
```

— the forward Fokker–Planck equation, term for term. ∎

So the SAME learned score yields two exact samplers: a stochastic one
(the reverse SDE) and a deterministic one (the PF-ODE) — same
marginals, radically different error behavior once the score is
wrong and time is discretized (`samplers_and_convergence/05` proves
the ODE transports error while the SDE contracts it; `04` here in
foundations shows DDIM is the ODE's natural integrator). The ODE
additionally makes the model a normalizing flow: likelihoods by the
instantaneous change-of-variables formula, and a well-defined
encoding map — structure the SDE does not have.

## The Interpolating Family, Proved

**Proposition.** For every `\lambda \ge 0`, the process

```math
\mathrm{d}x = \Big[-f + \tfrac{(1+\lambda^2)}{2}g^2\,\nabla\log p_{T-\tau}\Big]\mathrm{d}\tau + \lambda\,g\,\mathrm{d}\bar W_\tau
```

has marginals `p_{T-\tau}`: `\lambda = 0` is the PF-ODE, `\lambda = 1`
is Anderson's reverse SDE, and every intermediate noise level is an
equally exact sampler.

*Proof.* The same two computations combined: the FP right side is
`\nabla\cdot(fp) - \tfrac{(1+\lambda^2)}{2}g^2\Delta p +
\tfrac{\lambda^2 g^2}{2}\Delta p = \nabla\cdot(fp) -
\tfrac{g^2}{2}\Delta p`, independent of `\lambda`. ∎

The design reading: `\lambda` buys stochastic self-correction at the
price of integration noise, and "churn"/ancestral/deterministic
sampler variants are points on this dial. Exactness is free at every
`\lambda`; ROBUSTNESS to score error is not, and pricing that is
phase A's job.

## Load-Bearing Audit

```text
p_t positive and smooth   the score must exist everywhere the sampler
                          goes; manifold-supported data violates it as
                          t -> 0 (the sigma^{-2} blowup —
                          statistical_theory/04), which is why
                          samplers stop at t_min > 0;
FP-level identification   the proofs identify MARGINALS, not path
                          laws; enough for sampling, not for
                          trajectory-level claims (Anderson's SDE
                          statement covers those, cited);
exact score               all three theorems; every real system runs
                          on an estimate — the entire error theory
                          (A, F) begins where these proofs end;
divergence theorem usage  boundary terms discarded: finite second
                          moments / decay at infinity assumed.
```

## Position In The Coordinate System

This file opens the solver coordinate `S`: given `(P, s)`, there is
not one reverse dynamics but a `\lambda`-family of exact ones, with
the ODE and the SDE as endpoints. Every sampler in this repository is
a discretization of a point on this dial, and every guarantee in
phase A is a statement about what discretization plus score error do
to these three exact theorems.

## What Remains Open

Nothing about the theorems; the openness is all inherited: which
`\lambda` is optimal under a FIXED score-error budget (partially
understood — A/05 — with no complete theory), and the minimal
regularity under which the FP-level argument is honest for real data
(manifold-supported measures need the smoothed formulation; the
`t_min` cutoff is a patch with a rate, not a resolution).
