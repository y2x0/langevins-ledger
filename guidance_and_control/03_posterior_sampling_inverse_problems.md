# Posterior Sampling And Inverse Problems

## The Question

Inverse problems — inpainting, deblurring, superresolution, CT —
are conditioning on a measurement `y = A x_0 + \sigma_y\,\varepsilon`,
and by `01` the exact conditional sampler needs one object:
`\nabla\log p_t(y|x)`. This file solves the Gaussian-prior case in
closed form — which identifies exactly which term the popular DPS
approximation drops — then quantifies the damage with numbers, and
places the correction literature. The through-line is the family's
one error mechanism, third appearance: an expectation pushed through
a nonlinearity.

## The Exact Guidance Term

By `01`, `h_t(x) = p_t(y|x) = E[\,p(y|x_0)\,|\,x_t = x\,]` — the
measurement likelihood averaged over the denoising posterior. For the
linear-Gaussian measurement this is a Gaussian integral:

**Theorem (exact, Gaussian prior).** If `p_0 = N(\mu, \Sigma)` (so
the denoising posterior `x_0|x_t` is Gaussian with mean
`\hat x_0(x,t)` and covariance `C_t` — the Wiener filter of
`score_foundations/06`, with `C_t` independent of `x`), then

```math
p_t(y\,|\,x) \;=\; N\Big(y;\ A\,\hat x_0(x, t),\ \ \underbrace{A\,C_t\,A^\top + \sigma_y^2 I}_{=:\ S_t}\Big),
```

```math
\nabla_x \log p_t(y|x)
\;=\;
\Big(\frac{\partial \hat x_0}{\partial x}\Big)^{\!\top} A^\top\, S_t^{-1}\,\big(y - A\hat x_0(x,t)\big).
```

*Proof.* `y = A x_0 + \sigma_y\varepsilon` with
`x_0|x_t \sim N(\hat x_0, C_t)`: a linear map of a Gaussian plus
independent Gaussian noise is Gaussian with the displayed moments;
differentiate the log-density (the `x`-dependence enters through
`\hat x_0` only, `C_t` being constant for Gaussian priors). ∎

## What DPS Drops

DPS (Chung et al.) uses
`\hat h(x) = N(y; A\hat x_0(x), \sigma_y^2 I)` — the theorem's
formula with `S_t` replaced by `\sigma_y^2 I`: **the denoising
posterior's covariance `A C_t A^\top` is deleted.** In the Gaussian
case the deletion is the entire approximation, so its cost is exact:

**Proposition (mis-scaling, with numbers).** Scalar case
(`A = a`, `\Sigma = v_0`): the exact guidance weight is
`a/(a^2 c_t + \sigma_y^2)`, DPS's is `a/\sigma_y^2`; the
overweighting ratio is

```math
\frac{a^2 c_t + \sigma_y^2}{\sigma_y^2},
\qquad
c_t = \mathrm{Var}(x_0|x_t) = \frac{v_0\,\sigma_t^2}{\alpha_t^2 v_0 + \sigma_t^2}.
```

Numbers: unit-variance data (`v_0 = 1`), a 1% measurement
(`\sigma_y = 0.1`), `a = 1`. Late (`\sigma_t^2 = 0.01`):
`c_t \approx 0.0099`, ratio `\approx 2` — mild. Early
(`\sigma_t^2 = 1`): `c_t = 0.5`, ratio `= 51`: **DPS pulls toward
the measurement fifty times harder than the exact posterior warrants,
precisely at high noise** — where `samplers_and_convergence/05` says
ODE-style samplers will faithfully transport the resulting
displacement to the output. The empirical DPS phenomenology
(over-fitting the measurement, reduced diversity, step-size knobs
tuned per task) is this ratio, experienced; the "step size" that
practitioners tune is an ad-hoc scalar surrogate for the missing
`S_t^{-1}`. ∎

For NON-Gaussian priors, `C_t` depends on `x` and the posterior is
not Gaussian; then DPS's `p(y|\hat x_0)` versus the exact
`E[p(y|x_0)|x_t]` is `01`'s expectation-versus-plug-in proposition
verbatim — Jensen's gap through the Gaussian likelihood, largest for
diffuse posteriors. Corrections in the literature are exactly
attempts to restore `S_t`: `\Pi`GDM (Gaussian-posterior surrogate
with a tractable `C_t` proxy), covariance-aware variants, sequential
Monte Carlo guidance (asymptotically exact, at particle cost) —
statements, each locatable as "which surrogate for `C_t`".

## The `\sigma_y \to 0` Fence

Hard constraints (exact inpainting: `y = A x_0`) push
`S_t \to A C_t A^\top`, still well-posed — but DPS's
`\sigma_y^{-2} \to \infty`: the plug-in becomes a hard projection
onto `\{A\hat x_0 = y\}`, and replacement-style inpainting
(overwrite observed pixels each step) is its crude limit. The exact
theory says the constraint should be enforced with weight
`(A C_t A^\top)^{-1}` — softly early (posterior uncertain), hard only
as `t \to 0`. The common artifact of replacement inpainting —
boundary seams between observed and generated regions — is the
signature of enforcing at `t` what should only bind at `0`
(mechanism-level account; statement).

## Load-Bearing Audit

```text
linearity of A            the Gaussian integral; nonlinear forward
                          operators put the nonlinearity inside the
                          expectation and only the Jensen reading
                          survives;
Gaussian prior            makes C_t x-independent and the theorem
                          closed-form; it is the calibration case, in
                          this family's standard sense — the
                          UNREPRESENTATIVE solvable point
                          (score_foundations/06);
sigma_y > 0               DPS's formula; the fence above;
d(x0-hat)/dx via autodiff both exact and DPS guidance need the
                          denoiser Jacobian — the practical cost that
                          motivates Jacobian-free surrogates
                          (statements).
```

## Position In The Coordinate System

`01`'s theorem instantiated where it is checkable: the h-transform's
`h` has a closed form for linear-Gaussian measurements, the leading
approximation's error is an identified missing covariance, and the
noise-level profile of that error (large early, small late) is the
same profile as `02`'s gap bound — one geometry across conditioning,
guidance, and inverse problems.

## What Remains Open

Tractable `C_t` estimation for real priors (the entire correction
literature is surrogates; no result bounds a surrogate's end-to-end
posterior error); consistency guarantees for measurement-guided
samplers on manifold data (the exact theory needs the prior score
where the guidance pushes — `statistical_theory/04`'s territory); and
the SOLVER interaction: the proposition's early-time overweighting is
provably worse under transport-faithful (ODE) sampling than under
contractive (SDE) sampling by `samplers_and_convergence/05`'s
mechanism, but the combined statement — guidance error times solver
dynamics — has no theorem, here or anywhere in the family. It is the
family's most repeated open item, which is what makes it the real
one.
