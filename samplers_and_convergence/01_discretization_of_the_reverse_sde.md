# Discretizing The Reverse SDE

## The Question

`score_foundations/03` produced exact continuous-time samplers; a
computer runs steps. Euler–Maruyama on the reverse SDE is the
baseline every guarantee in this family is written against, and its
error structure — what a step of size `h` costs, where the dimension
enters, what is bias versus noise — can be computed EXACTLY in the
Gaussian case. This file does so, defines the error vocabulary the
family uses, and prices the baseline that `02`'s integrators beat.

## The Scheme

Grid `T = t_0 > t_1 > \dots > t_K = t_{\min}`, steps
`h_k = t_{k-1} - t_k`. EM freezes the reverse drift at the left grid
point:

```math
x_{k+1} \;=\; x_k + h\,\Big[\tfrac{\beta}{2}x_k + \beta\,\hat s(x_k, t_k)\Big] + \sqrt{\beta h}\;\xi_k,
\qquad \xi_k \sim N(0, I)
```

(VP, `\lambda = 1` endpoint; the `\lambda`-family discretizes the
same way). Two error notions, used throughout the family:

```text
strong error   E||x_K - x(t_K)||: pathwise closeness (couples the
               noise); relevant for trajectory claims, NOT needed for
               sampling;
weak error     |E f(x_K) - E f(x(t_K))| or a distribution distance
               (KL, TV, W2): what sampling actually requires — the
               family's guarantees (03, 04) are all weak.
```

## The Gaussian Case, Exact

Data `N(0, 1)` per coordinate, so `p_t = N(0,1)` for all `t` and
`s_t(x) = -x` (`score_foundations/06`): the reverse SDE is itself the
OU process `dx = -\tfrac{\beta}{2}x\,d\tau + \sqrt{\beta}\,d\bar W`,
whose law EM should preserve.

**Theorem (exact discretization bias).** With exact score and constant
`\beta h < 2`, one EM step maps variances by

```math
v \;\mapsto\; \big(1 - \tfrac{\beta h}{2}\big)^2 v + \beta h,
```

whose fixed point is

```math
v^*_h \;=\; \frac{1}{1 - \beta h/4}
\;=\; 1 + \frac{\beta h}{4} + O\big((\beta h)^2\big):
```

EM run to stationarity oversamples the variance by EXACTLY
`(1-\beta h/4)^{-1}` — a pure `O(h)` bias with explicit constant,
independent of the noise realizations, per coordinate.

*Proof.* The step is linear with multiplier `(1-\beta h/2)` and
additive noise variance `\beta h`; iterate the affine variance map;
solve `v = (1-\beta h/2)^2 v + \beta h`:
`v[\beta h - \beta^2h^2/4] = \beta h`. ∎

Three readings, each load-bearing for the family:

**1. Bias, not noise.** The `O(h)` error is systematic — the mean of
the sampled law is right, its spread is wrong. Halving `h` halves it;
averaging more samples does not touch it. This is the cleanest
instance of the weak-error structure the general theorems (03, 04)
bound: discretization error enters the KL as a drift-mismatch
integral, and it is deterministic in character.

**2. Where `d` enters.** Per coordinate the bias is `\beta h/4`; in
`d` dimensions the KL to the target adds across independent
coordinates: `KL \approx d\,(\beta h/4)^2/2`-scale (Gaussian KL is
quadratic in small variance mismatch — expand
`\tfrac12(v - 1 - \log v)`). Matching a KL budget `\varepsilon^2`
therefore forces `h \lesssim \varepsilon/\sqrt{d}`-type steps: the
`\mathrm{poly}(d)` in every iteration-complexity theorem is visible
already in the exactly solvable case, and `04` shows how nearly-
linear-in-`d` is recovered.

**3. Schedule sensitivity.** With time-varying `\beta_t`, the bias
per unit time scales with `\beta_t^2 h`: uniform-in-`t` grids
overspend where `\beta` is small and underspend where it is large —
the first-principles case for the non-uniform grids every practical
sampler uses, and for doing analysis in `score_foundations/04`'s
`\rho` (noise-to-signal) clock, where `02` shows the drift's stiff
linear part disappears entirely.

## The General One-Step Expansion, Stated Honestly

For non-Gaussian data the drift `b(x, t) = \tfrac{\beta}{2}x + \beta
s_t(x)` is nonlinear, and the standard Itô–Taylor bookkeeping gives:
one EM step incurs local weak error `O(h^2)` per step —
`O(h)` globally — with constants involving
`\partial_t s` and `\nabla s` along the path (the score's smoothness
in space AND time; the `t`-derivative is why samplers slow down near
`t_{\min}`, where `score_foundations/06`'s empirical score sharpens at
rate `\sigma_t^{-2}`). Proof machinery: standard numerical-SDE
theory, cited, with the diffusion-specific accounting — WHICH
smoothness constants, integrated how — done properly by the Girsanov
route in `03`, where the drift mismatch appears inside a KL integral
rather than a Taylor remainder. This file's exact Gaussian constants
are the sanity anchor for both.

## Load-Bearing Audit

```text
exact score              this file isolates DISCRETIZATION; the score
                         error term joins in 03 and the two add (in
                         KL, literally);
linearity (Gaussian)     the exact fixed point; nonlinear data
                         replaces it with Taylor remainders under
                         smoothness hypotheses — hypotheses that FAIL
                         near t_min on rough data (the audit item
                         every convergence theorem carries as an
                         early-stopping parameter);
beta h < 2               stability of the linear multiplier; violated
                         => the variance map diverges: step-size
                         limits are stability, not just accuracy;
lambda = 1               EM on the ODE endpoint has its own (smaller,
                         noiseless) bias — 05 prices the comparison
                         properly.
```

## Position In The Coordinate System

The solver coordinate `S` at its crudest point: freeze the drift,
step, add noise. Everything the family proves is relative to this
baseline: `02` removes the linear part's error exactly, `03`–`04`
bound what score error adds, `05` chooses `\lambda`, `06` appends
correctors.

## What Remains Open

Optimal grids: the schedule-sensitivity reading gives the right
qualitative rule, but the provably optimal step allocation for a
given data class and budget is open beyond Gaussian and kernel toys
(same open problem as `score_foundations/04` flagged, now with the
error functional visible); and sharp constants for the weak error of
EM near `t_{\min}` on manifold-supported data — where the smoothness
hypotheses degrade at a known rate (`statistical_theory/04`) but the
integrated cost has no tight two-sided bound.
