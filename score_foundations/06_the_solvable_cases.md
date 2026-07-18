# The Solvable Cases

## The Question

Two data distributions admit complete closed-form treatment: the
Gaussian, and the empirical measure of a finite dataset. The first is
this repository's LQR — every object exact, every sampler verifiable.
The second is more consequential: the optimal score for the TRAINING
objective on `n` samples is computable in closed form, it is a
softmax attention over the dataset, and the exact sampler built on it
provably outputs training points. Generalization in diffusion models
is therefore a property of score-estimation ERROR — a conclusion
proved here and developed in `statistical_theory/`.

## Case 1: Gaussian Data, Everything Exact

Let `p_0 = N(\mu, \Sigma)`. Then (from `01`'s kernel)
`p_t = N(\alpha_t\mu,\ \alpha_t^2\Sigma + \sigma_t^2 I)`, and:

**Proposition.** The score, denoiser, and PF-ODE are all affine:

```math
s_t(x) = -\big(\alpha_t^2\Sigma+\sigma_t^2 I\big)^{-1}(x - \alpha_t\mu),
\qquad
\hat x_0(x) = \mu + \alpha_t\,\Sigma\big(\alpha_t^2\Sigma + \sigma_t^2 I\big)^{-1}(x-\alpha_t\mu),
```

the denoiser being exactly the Wiener filter (compute
`E[x_0|x_t]` for jointly Gaussian variables, or substitute the score
into Tweedie — the two derivations agree, which checks `02`). The
PF-ODE is linear, hence solvable, and the exact sampler started from
the prior reproduces `N(\mu, \Sigma)` exactly at `t \to 0`; DDIM with
exact `\hat x_0` is exact at ANY step count for `\Sigma \propto I`
(the denoiser is then constant along each trajectory's `u`-line —
`04`'s frozen-denoiser hypothesis holds with zero error). ∎

The Gaussian case is the calibration standard: every solver in phase
A is first run here, where its error has a formula. It is also
(`05`) the one case where CFG is exact — the solvable case is
systematically the UNREPRESENTATIVE case, and this file says so once
so the rest of the repo can cite it.

## Case 2: The Empirical Measure — The Score Is Attention

Let `p_0 = \frac1n\sum_{i=1}^n \delta_{x^{(i)}}` (the training set).
Then `p_t` is a mixture of `n` Gaussians, and:

**Theorem.** The exact score and denoiser of the empirical measure are

```math
\hat x_0(x, t) \;=\; \sum_{i=1}^n w_i(x, t)\ x^{(i)},
\qquad
w_i(x,t) \;=\; \mathrm{softmax}_i\!\Big(\frac{-\,\|x - \alpha_t x^{(i)}\|^2}{2\sigma_t^2}\Big),
```

with `s_t` given by Tweedie. **The optimal denoiser is a softmax
attention query over the training set** — query `x`, keys
`\alpha_t x^{(i)}`, values `x^{(i)}`, temperature `\sigma_t^2`.

*Proof.* The posterior over which mixture component generated `x_t=x`
is proportional to the component likelihoods
`k_t(x|x^{(i)})`; the conditional mean is the
posterior-weighted average of the components' `x_0` values; the
Gaussian likelihoods' exponents give exactly the displayed softmax. ∎

The cross-repository content is not decorative:
attention-ledger/foundations/01 proved attention is Nadaraya–Watson
kernel regression; this theorem says the Bayes-optimal denoiser IS
that estimator with the Gaussian kernel at bandwidth `\sigma_t` —
so a trained denoiser is being asked to implement, and compress, an
attention over its dataset. The temperature dial is even the same
object: `\sigma_t \to \infty` is uniform attention (the denoiser
returns the data mean), `\sigma_t \to 0` is hard retrieval. That
limit is the next theorem.

**Theorem (memorization).** Run the exact reverse dynamics (any
`\lambda` of `03`'s family) for the empirical measure. As
`t \to 0`, the sampler's output distribution converges to the
empirical measure itself: exact-score diffusion on `n` training
points GENERATES the `n` training points (up to ties on
measure-zero boundaries).

*Proof.* Exactness of the reversal (`03`): at every `t > 0` the
sampler's law is exactly `p_t`, the Gaussian-mixture smoothing of the
empirical measure. As `t \to 0`, `\alpha_t \to 1, \sigma_t \to 0`,
and `p_t \Rightarrow p_0` weakly (the mixture collapses onto its
atoms; explicitly, `W_2(p_t, p_0)^2 \le (1-\alpha_t)^2 M_2 +
\sigma_t^2 d \to 0` by coupling each component to its atom). The
trajectory-level picture matches: for `x` in the basin of atom `i`,
`w_i \to 1` exponentially in `1/\sigma_t^2` (softmax with diverging
temperature gap — the same tail bound as attention-ledger's
retrieval files), so `\hat x_0 \to x^{(i)}` and `04`'s ODE
`du/d\rho = (u - \hat x_0)/\rho` drives `u` linearly into the atom. ∎

**The corollary that frames two later families.** The training
objective (`02`, Vincent) is minimized over unrestricted functions by
THIS score. Therefore: a model that fits its objective perfectly on
the empirical data reproduces the training set, and everything we
call generalization in diffusion models — novel samples, semantic
interpolation — is attributable to the ways the learned score FAILS
to be the empirical optimum: smoothing from finite capacity,
early-stopped optimization, architectural inductive bias, and the
`t_{\min}` cutoff. Error is not the obstacle to generalization; in
this precise sense it is the mechanism. `statistical_theory/03`
develops this into the inductive-bias ledger; phase F's rates price
how much smoothing the data can support.

## Load-Bearing Audit

```text
Gaussian mixtures            both cases live on 01's closed-form
                             kernel; the theorems are exact, no
                             asymptotics hidden;
unrestricted function class  the memorization corollary needs "the
                             optimum of the objective" to be the
                             empirical score — capacity limits are
                             exactly what break it, by design of the
                             argument;
exact reversal               the memorization theorem inherits 03's
                             hypotheses; real samplers add
                             discretization error, which SMOOTHS —
                             even the sampler's defects push toward
                             generalization;
weak convergence at t -> 0   the statement is distributional;
                             per-trajectory basin claims use the
                             softmax tail bound and exclude ties.
```

## Position In The Coordinate System

The `(P, s, S)` system evaluated at its two solvable points: Gaussian
data (the estimand affine, the solver exact, the calibration case)
and empirical data (the estimand is attention, the solver is a
retrieval mechanism, and exactness equals memorization). Between
these two poles — infinitely smooth and atomically rough — sits every
real dataset, and the repository's remaining phases are about that
interval.

## What Remains Open

Partially closed by the retrofits from `statistical_theory/`: the
interpolation question now has its solved endpoints — early stopping
is EXACTLY kernel density estimation at bandwidth `\sigma_t/\alpha_t`
(`statistical_theory/02`'s reduction, with the classical rate proved),
and for linear score classes the learned score is exactly the
`L^2(\hat p_t)`-projection of this file's empirical-attention score
onto the feature span, generalization being the projection residual
(`statistical_theory/03`). The open middle is the deep case: which
smoother a given architecture implements, and the memorization
transition `n_{mem}` — named in `statistical_theory/05` as the
field's most consequential open problem. Also still open: the
attention correspondence made architectural — whether trained
denoisers implement dataset-attention in any mechanistically
identifiable way (the interpretability question this file's theorem
makes well-posed, and which nobody has answered).
