# DDPM, DDIM, And The ELBO

## The Question

The discrete-time face of the machinery: DDPM's ancestral sampler,
its variational training objective, and DDIM's deterministic
shortcut. Three proofs settle what each IS: the ancestral step is an
exact Gaussian posterior with the denoiser plugged in; the ELBO is a
weighted regression on `02`'s estimand; and DDIM is not a heuristic
but the EXACT solution of `03`'s probability-flow ODE under a frozen
denoiser — the theorem that makes fast samplers principled.

## The Ancestral Step Is A Posterior, Proved

**Theorem.** In the DDPM chain (`01`), the conditional of `x_{k-1}`
given `x_k` AND `x_0` is Gaussian:

```math
q(x_{k-1}\,|\,x_k, x_0) = N\big(\tilde\mu_k(x_k, x_0),\ \tilde\sigma_k^2 I\big),
```

```math
\tilde\mu_k = \frac{\sqrt{\bar\alpha_{k-1}}\,\beta^{(k)}}{1-\bar\alpha_k}\,x_0
+ \frac{\sqrt{1-\beta^{(k)}}\,\big(1-\bar\alpha_{k-1}\big)}{1-\bar\alpha_k}\,x_k,
\qquad
\tilde\sigma_k^2 = \frac{\big(1-\bar\alpha_{k-1}\big)\beta^{(k)}}{1-\bar\alpha_k}.
```

*Proof.* Bayes with all three conditionals Gaussian (`01`):
`q(x_{k-1}|x_k,x_0) \propto q(x_k|x_{k-1})\,q(x_{k-1}|x_0)`. The
exponent is a quadratic in `x_{k-1}`; collecting coefficients: the
precision is `\frac{1-\beta^{(k)}}{\beta^{(k)}} +
\frac{1}{1-\bar\alpha_{k-1}} = \frac{1-\bar\alpha_k}{\beta^{(k)}(1-\bar\alpha_{k-1})}`
(one line of algebra with `\bar\alpha_k = (1-\beta^{(k)})
\bar\alpha_{k-1}`), and the mean is the precision-weighted combination
displayed. ∎

DDPM sampling is this posterior with `x_0` replaced by the learned
`\hat x_0(x_k, k)` — the estimand of `02` — plus the posterior's own
noise. Nothing else. The sampler's honest description: "repeatedly
re-estimate the clean image, take the exact Gaussian posterior step
toward it."

## The ELBO Is Weighted Regression, Proved

**Theorem.** The negative ELBO of the DDPM generative chain against
the forward chain decomposes as

```math
\sum_{k>1}\ \mathbb{E}\ \mathrm{KL}\big(q(x_{k-1}|x_k,x_0)\ \big\|\ p_\theta(x_{k-1}|x_k)\big)
\;+\;\text{(edge terms)},
```

and with `p_\theta` Gaussian with the posterior's variance, each KL
is an exact scaled square:

```math
\mathrm{KL}_k
= \frac{\|\tilde\mu_k(x_k, x_0) - \tilde\mu_k(x_k, \hat x_0)\|^2}{2\tilde\sigma_k^2}
= w_k\,\mathbb{E}\big\|\varepsilon - \hat\varepsilon\big\|^2,
\qquad
w_k = \frac{\beta^{(k)\,2}}{2\tilde\sigma_k^2\,(1-\beta^{(k)})(1-\bar\alpha_k)},
```

*Proof.* The chain-rule decomposition of the ELBO into per-step KLs is
the standard telescoping (each `q(x_{k-1}|x_k,x_0)` term appears by
inserting `x_0` and using Markov structure); the KL between Gaussians
with equal covariance is `\|\Delta\mu\|^2/2\tilde\sigma^2`; and
`\tilde\mu` depends on its `x_0`-slot linearly with coefficient
`\sqrt{\bar\alpha_{k-1}}\beta^{(k)}/(1-\bar\alpha_k)`, which converted
to the `\varepsilon`-parametrization (via `02`'s dictionary at
`x_k` fixed) gives the displayed weight. ∎

Corollary, and the honest sentence the field repeats loosely:
"training the noise predictor with UNIT weights" — the empirical DDPM
loss — "is the ELBO with its weights `w_k` deleted." Likelihood wants
`w_k`; samples prefer the deletion; `02`'s weighting audit is the
theorem-shaped version of that tension, and it is a modeling CHOICE,
not a derivation.

## DDIM Is The Exact PF-ODE Step, Proved

**Theorem.** Change variables to `u = x/\alpha` (denoised scale) and
`\rho = \sigma/\alpha` (noise-to-signal ratio). Along `03`'s
probability-flow ODE, with the score written through `02`'s
dictionary as `\hat x_0`,

```math
\frac{\mathrm{d}u}{\mathrm{d}\rho} \;=\; \frac{u - \hat x_0(x, t)}{\rho} .
```

If `\hat x_0` is held constant over a step (the frozen-denoiser /
exponential-integrator approximation), the ODE is linear and solves
EXACTLY to

```math
u_{k-1} = \hat x_0 + \frac{\rho_{k-1}}{\rho_k}\,(u_k - \hat x_0)
\quad\Longleftrightarrow\quad
x_{k-1} = \alpha_{k-1}\,\hat x_0 + \sigma_{k-1}\,\hat\varepsilon
```

— which is precisely the DDIM update.

*Proof.* Derive the ODE: with `f = -\tfrac{\beta}{2}x`, `g^2 = \beta`,
and `s = (\alpha\hat x_0 - x)/\sigma^2` (Tweedie),
`\dot x = -\tfrac{\beta}{2}x - \tfrac{\beta}{2}\,
\frac{\alpha\hat x_0 - x}{\sigma^2}`. Then
`\dot u = \dot x/\alpha + \tfrac{\beta}{2}u
= -\tfrac{\beta}{2}\,\frac{\hat x_0 - u}{\sigma^2}\cdot\frac{\alpha^2}{\alpha^2}`
… carried through: `\dot u = -\tfrac{\beta}{2\sigma^2}(\hat x_0 - u)`;
and `\dot\rho = \tfrac{\beta}{2\sigma\alpha}(\sigma^2+\alpha^2)
= \tfrac{\beta}{2\sigma\alpha}`, so
`du/d\rho = \dot u/\dot\rho = -\tfrac{\alpha}{\sigma}(\hat x_0 - u)
= (u-\hat x_0)/\rho`. Solve the linear equation with `\hat x_0`
frozen: `u(\rho) - \hat x_0 \propto \rho`. Translate back with
`u = x/\alpha`, `\rho = \sigma/\alpha`:
`x_{k-1} = \alpha_{k-1}\hat x_0 +
\sigma_{k-1}\,(x_k - \alpha_k\hat x_0)/\sigma_k
= \alpha_{k-1}\hat x_0 + \sigma_{k-1}\hat\varepsilon`. ∎

Read what the theorem buys: DDIM's error per step is ENTIRELY the
variation of `\hat x_0` along the step — the linear part of the
dynamics is integrated exactly, at any step size. That is why DDIM
tolerates 20 steps where Euler–Maruyama needs hundreds, why the
`(u, \rho)` coordinates are the natural home of every fast solver
(DPM-Solver's higher orders = polynomial extrapolation of `\hat x_0`
in these very coordinates — `samplers_and_convergence/02`), and why
`\eta`-noised DDIM variants are `03`'s `\lambda`-dial discretized.

## Load-Bearing Audit

```text
Gaussian conjugacy         the posterior theorem — exact only because
                           every conditional in sight is Gaussian;
equal-variance KLs         the clean square in the ELBO; learned
                           variances add a trace term (statement);
frozen denoiser            DDIM's ONLY approximation; the theorem
                           converts "few-step sampling" from folklore
                           into an assumption about the smoothness of
                           x0-hat along the path — measurable, and
                           phase D's distillation targets exactly it;
the (u, rho) coordinates   the linearity; they exist because the
                           forward process is linear (01's audit).
```

## Position In The Coordinate System

The solver coordinate `S`, discretized: the ancestral sampler
discretizes the `\lambda = 1` endpoint through an exact posterior;
DDIM discretizes the `\lambda = 0` endpoint through an exact
integrator; the ELBO prices the first and is silent about the second.
Phase A takes over from here with error rates; phase D collapses the
step count to one.

## What Remains Open

The optimal step-placement problem (given a budget of `K`
evaluations, where to put them in `\rho` — solved for toy cases,
heuristic in practice); learned per-step variances vs the two exact
endpoints; and the correct statement of "few-step samplers work
because `\hat x_0` is smooth along the path" as a theorem about data
rather than an observation about models.
