# Score Error Propagation

## The Question

Phase A bounded the sampler's KL error by the score's `L^2` error
under the data path (`samplers_and_convergence/03`). This file
completes the stability picture with the two companion statements
that phase F's estimation theory plugs into: the Wasserstein version
(proved by synchronous coupling and Grönwall — with an honest
exponential constant the KL route avoids), and the Gaussian case
solved exactly (the error PROPAGATOR in closed form, generalizing
`samplers_and_convergence/05`'s constant-bias computation to
arbitrary time-dependent error). Together the three bounds are the
complete interface between "how well is the score estimated" (this
phase) and "what does the sampler output" (phase A).

## The Wasserstein Bound, Proved

Two reverse-time SDEs, same `g`, drifts `b` (exact) and
`\hat b = b + g^2(\hat s - s)` (learned), run from a common
initialization coupling.

**Theorem.** Suppose `b(\cdot, t)` is `L_t`-Lipschitz in space, and
let `\delta_t := \big(\mathbb{E}_{\hat q_t}\|g^2(\hat s - s)(\hat
x_t)\|^2\big)^{1/2}` be the drift error along the SAMPLER's path.
Then

```math
W_2\big(p_{t_{\min}},\ \hat q_{t_{\min}}\big)
\;\le\;
\int_{t_{\min}}^{T} e^{\int_{t_{\min}}^{t} L_r\,\mathrm{d}r}\ \delta_t\ \mathrm{d}t
\;+\; e^{\int L}\,W_2\big(p_T, \gamma\big).
```

*Proof.* Synchronous coupling: same Brownian motion, so the
difference `\Delta_\tau = x_\tau - \hat x_\tau` obeys the
noise-free equation
`\dot\Delta = [b(x) - b(\hat x)] + [b(\hat x) - \hat b(\hat x)]`,
giving `\frac{d}{d\tau}\|\Delta\| \le L\|\Delta\| +
\|g^2(s-\hat s)(\hat x)\|` pathwise; Grönwall integrates it;
take `L^2(\mathbb{P})` norms (Minkowski's integral inequality) and
bound `W_2` by the coupling's cost. The initialization term is
`score_foundations/01`'s contraction applied to the mismatch at
`T`. ∎

Two honest annotations, which are the content. First, the error is
measured under the SAMPLER's path `\hat q_t` — the quantity training
does NOT directly control (Vincent's loss lives on `p_t`) — while the
KL route (`A/03`) measures under the data path but bounds a weaker
divergence. The two bounds fail in complementary places, and no
current theorem gets both the strong metric and the trainable norm at
once (`05` lists it). Second, the constant `e^{\int L}` is the
Grönwall price: `L_t` inherits the score's Lipschitz constant, which
`04` proves blows up like `\sigma_t^{-2}` near supported data — the
Wasserstein bound is honest only down to a cutoff, which is the
`t_{\min}` story again, now visible in a THIRD proof.

## The Gaussian Propagator, Exact

Per-coordinate Gaussian data (`p_t = N(0,1)`, `s = -x`), arbitrary
time-dependent score error: `\hat s(x, t) = -x + \beta_t^{err}`.

**Theorem.** Along the `\lambda`-family sampler, the output mean
error is EXACTLY

```math
m(t_{\min})
\;=\;
\int_{t_{\min}}^{T}
\underbrace{\exp\Big(-\frac{\lambda^2}{2}\int_{t_{\min}}^{t}\beta_r\,\mathrm{d}r\Big)}_{\text{the propagator}}\ \cdot\
\frac{1+\lambda^2}{2}\,\beta_t\ \beta^{err}_t\ \mathrm{d}t .
```

*Proof.* `samplers_and_convergence/05`'s mean ODE
`\dot m = -\tfrac{\lambda^2}{2}\beta m + \tfrac{1+\lambda^2}{2}\beta\,
\beta^{err}` is linear; variation of constants. ∎

Read the propagator as the file's summary object: it weights each
noise level's score error by how much of it SURVIVES to the output.
For the ODE (`\lambda = 0`) the propagator is `1` — every error at
every level arrives intact (the transport theorem, re-derived); for
the SDE it decays exponentially in the remaining schedule — errors at
high noise are forgiven, errors near `t_{\min}` are not. The design
consequence, now quantitative: **estimation accuracy matters most at
low noise for every sampler, and at low noise ONLY for stochastic
ones** — which is where `02`–`04` will show estimation is also
hardest (small `\sigma` = small bandwidth = high variance), the
genuinely adversarial coincidence at the heart of the subject.

## The Interface, Stated

What this phase must deliver to phase A, by route:

```text
KL route (A/03)      the weighted L2 error int g^2 E_{p_t}||s-s_hat||^2:
                     Vincent's loss integrand — DIRECTLY the training
                     objective; delivered by 02-03's estimation rates;
W2 route (here)      error under the sampler path + Lipschitz budget:
                     needs 04's geometry to know where the budget
                     explodes;
exact route (here)   the propagator: which noise levels' errors
                     matter, per lambda — the weighting that 02's
                     bandwidth story and foundations/02's estimand
                     weightings should be (and are not) chosen against.
```

## Load-Bearing Audit

```text
synchronous coupling      the W2 proof's engine — third use in the
                          repository (foundations/01, A/05, here);
Lipschitz drift           the Gronwall constant; 04 prices its
                          failure near supports;
error under q-hat         the W2 route's honest weakness, stated;
linearity (Gaussian)      the exact propagator; nonlinear data makes
                          the propagator the linearized flow's
                          fundamental matrix (statement).
```

## Position In The Coordinate System

The interface file: `(P, s, S)` fixed, and the map from
estimand-error to output-error characterized in three metrics — with
one summary object (the propagator) saying WHEN errors matter. The
remaining files supply the other half: how large the estimation error
must be, given `n` samples and the data's geometry.

## What Remains Open

The mixed bound (strong metric, trainable norm — the annotation
above); propagators beyond the linear case (the fundamental-matrix
statement made quantitative for multimodal data would explain
mode-weight sensitivity, `A/06`'s open item, from the estimation
side); and using the propagator as a DESIGN principle — now `07`'s
minimax theorem: the propagator profile IS the optimal training
weight, and the same measure is the optimal step allocation. What
remains of the corollary is its non-Gaussian and empirical versions.
