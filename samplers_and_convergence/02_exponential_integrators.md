# Exponential Integrators

## The Question

`01` showed EM pays `O(h)` even on the exactly linear part of the
dynamics. The linear part is KNOWN — it is the schedule, not the data
— and integrating it exactly should cost nothing. Exponential
integrators do precisely that, and in `score_foundations/04`'s
`(u, \rho)` coordinates the whole subject becomes one identity: the
sampler's only real job is a QUADRATURE of the denoiser along the
noise scale. This file proves the identity, recovers DDIM as its
zeroth order, derives the second-order schemes and their error
constants, and names what DPM-Solver adds.

## The Quadrature Identity, Proved

From `score_foundations/04`: along the probability-flow ODE, with
`u = x/\alpha`, `\rho = \sigma/\alpha`,

```math
\frac{\mathrm{d}u}{\mathrm{d}\rho} = \frac{u - \hat x_0(x(\rho), \rho)}{\rho}.
```

**Theorem (variation of constants).** For any `0 < \rho_{k-1} <
\rho_k`, the EXACT solution satisfies

```math
u(\rho_{k-1})
\;=\;
\frac{\rho_{k-1}}{\rho_k}\,u(\rho_k)
\;+\;
\rho_{k-1}\int_{\rho_{k-1}}^{\rho_k}\frac{\hat x_0\big(\rho\big)}{\rho^2}\;\mathrm{d}\rho .
```

*Proof.* `\big(u/\rho\big)' = u'/\rho - u/\rho^2 =
(u - \hat x_0)/\rho^2 - u/\rho^2 = -\hat x_0/\rho^2`; integrate from
`\rho_{k-1}` to `\rho_k` and multiply by `\rho_{k-1}`. ∎

Every deterministic sampler in the diffusion literature is a
quadrature rule for this one integral. The linear dynamics — the
schedule, the stiffness, the `\rho_{k-1}/\rho_k` decay — is handled
EXACTLY, at any step size, by the prefactor; the sole approximation
anywhere is how `\hat x_0(\rho)` is interpolated between network
evaluations. The solver-design problem is thereby reduced to: given
`K` evaluations of a function along a path, integrate it against the
kernel `\rho^{-2}` well.

## The Orders, Derived

**Order 0 = DDIM.** Approximate `\hat x_0(\rho) \equiv \hat
x_0(\rho_k)` on the step:

```math
\rho_{k-1}\int_{\rho_{k-1}}^{\rho_k}\frac{\mathrm{d}\rho}{\rho^2}\;\hat x_0
= \Big(1 - \frac{\rho_{k-1}}{\rho_k}\Big)\hat x_0
\;\Longrightarrow\;
u_{k-1} = \frac{\rho_{k-1}}{\rho_k}u_k + \Big(1-\frac{\rho_{k-1}}{\rho_k}\Big)\hat x_0,
```

which is `score_foundations/04`'s DDIM update verbatim — DDIM is the
left-Riemann rule (in the reversed direction of integration) for the
quadrature. Local error: first-order Taylor of `\hat x_0` in the
integration variable gives per-step error
`O\big(h_\lambda\,\|\mathrm{d}\hat x_0/\mathrm{d}\lambda\|\big)`
where `\lambda = \log\rho` and `h_\lambda` the step in it (the
`\rho^{-2}\,d\rho` measure is `d(1/\rho)`; against `e^{-\lambda}`
weights, log-SNR is the variable in which the kernel is tame — this
is WHY the DPM-Solver family parametrizes everything in half-log-SNR,
now derived rather than chosen).

**Order 1 (two evaluations).** Interpolate `\hat x_0` linearly in
`\lambda` between the two most recent evaluations (multistep), or use
a midpoint predictor (single-step): carrying the linear term through
the same integral gives the trapezoid-type correction with local
error

```math
O\Big(h_\lambda^2\,\big\|\tfrac{\mathrm{d}^2\hat x_0}{\mathrm{d}\lambda^2}\big\|\Big)
```

— second order in the step, with the error constant now the SECOND
derivative of the denoiser along the path. These are the DPM-Solver-2
/ multistep DEIS schemes (their papers' `\lambda`-space
exponential-integrator derivations are this section in different
notation; higher orders extrapolate quadratically and inherit
third-derivative constants; statements for the bookkeeping, the
order conditions above derived).

**The theorem-shaped summary of "why 20 steps work":** total
deterministic-sampler error is a quadrature error, i.e. controlled by
derivatives of `\hat x_0` along the trajectory — a DATA property
(how fast the conditional mean turns as noise anneals), not a
schedule property. Smooth data ⇒ few steps; the steps' correct
placement is uniform-in-`\lambda` up to the derivative profile. The
frozen-denoiser hypothesis of `score_foundations/04` has become a
quantitative budget.

## The Stochastic Side, Fenced

The identity is for the ODE endpoint. The `\lambda > 0` family gains
a noise integral; exponential integrators still handle the linear
part exactly (the OU transition is Gaussian with known moments —
`01`'s map, used exactly rather than Eulerized), giving SDE solvers
with the same structure: exact linearity + denoiser quadrature +
EXACT noise moments. What stochasticity changes is not the order
bookkeeping but the error DYNAMICS — contraction versus transport —
which is `05`'s theorem, not this file's.

## Load-Bearing Audit

```text
linear forward process     the (u, rho) coordinates exist because 01
                           (foundations) is linear — the audit item
                           inherited by every solver in the family;
smoothness of x0-hat       the ONLY error source; near t_min on
                           atomic/manifold data the derivatives blow
                           up (score_foundations/06's sharpening) and
                           order collapses — high-order solvers
                           degrade FIRST where the data is roughest,
                           a real and observed failure;
learned score inside       x0-hat is the network's output; quadrature
                           error and score error compound — 03 adds
                           them in KL, and nothing here assumes the
                           score is right;
log-SNR parametrization    derived as the tame variable, not assumed.
```

## Position In The Coordinate System

The solver coordinate `S`, done right for the deterministic endpoint:
`S` reduces to a quadrature rule, and solver design is numerical
analysis of one integral. The file also fixes the family's division
of labor: deterministic error = derivatives of the denoiser (here);
stochastic error dynamics = `05`; score error = `03`–`04`.

## What Remains Open

Adaptive quadrature with guarantees (estimate the local derivative of
`\hat x_0` from evaluations, place steps accordingly — used
heuristically, no regret-style bound); the correct treatment at the
rough end (`t \to t_{\min}`), where the optimal order provably drops
but the crossover point is uncharacterized; and distillation's
relationship to this file — a one-step student (`distillation/`) is
an attempt to learn the whole quadrature as a function, and what
error IT inherits from the teacher's quadrature is priced only
empirically.
