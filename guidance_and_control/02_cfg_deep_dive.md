# CFG: The Deep Dive

## The Question

`score_foundations/05` proved the deflationary theorem: CFG's
per-noise-level tilt is not the diffusion path of any fixed target,
with the first-order gap identified as a Jensen gap. This file does
the constructive half: in the jointly Gaussian case — where
foundations/05 proved CFG IS exact — the `\omega`-family can be
computed in closed form, and the computation turns the folklore
("guidance sharpens modes and overshoots means") into two exact
statements. A quantitative bound on the Jensen gap then explains WHEN
guidance should be applied (the interval-guidance schedule), and the
`\omega \to \infty` endpoint is identified. Retrofit to
foundations/05 lands with this file.

## The Gaussian `\omega`-Family, Exactly

Jointly Gaussian case: conditional `p_t(x|y) = N(\mu_c, \Sigma_c)`,
unconditional `p_t(x) = N(\mu_u, \Sigma_u)` (both known in closed
form from `score_foundations/06` given a Gaussian joint). The CFG
target at strength `\omega` is the geometric tilt
`\propto p_t(x|y)^{\omega}\,p_t(x)^{1-\omega}`.

**Theorem.** The tilt is Gaussian with precision and mean

```math
\Lambda_\omega = \omega\,\Sigma_c^{-1} + (1-\omega)\,\Sigma_u^{-1},
\qquad
m_\omega = \Lambda_\omega^{-1}\big(\omega\,\Sigma_c^{-1}\mu_c + (1-\omega)\,\Sigma_u^{-1}\mu_u\big),
```

valid while `\Lambda_\omega \succ 0`. Consequences, each one line of
algebra from the display:

```text
variance shrink   conditioning reduces variance (Sigma_c <= Sigma_u
                  for jointly Gaussian), so Lambda_c >= Lambda_u and
                  Lambda_omega = Lambda_c + (omega-1)(Lambda_c -
                  Lambda_u) >= Lambda_c for omega >= 1: CFG output is
                  MORE concentrated than the true conditional,
                  monotonically in omega — "sharpening", exact;
mean overshoot    for Sigma_c = Sigma_u: m_omega = mu_c + (omega-1)
                  (mu_c - mu_u): linear extrapolation PAST the
                  conditional mean along the condition direction —
                  "saturation/over-typicality", exact;
breakdown         if Sigma_c is larger than Sigma_u in some direction
                  (possible for non-jointly-Gaussian mixtures'
                  moment-matched surrogates), Lambda_omega loses
                  positivity at finite omega: the tilt stops being
                  normalizable — the algebraic warning shot for
                  large-omega pathologies.
```

*Proof.* Products of Gaussian densities: exponents add; collect the
quadratic and linear terms. ∎

Since foundations/05 proved tilt-and-noise COMMUTE for jointly
Gaussian data, these are exact descriptions of what the Gaussian
`\omega`-CFG SAMPLER outputs — the one setting where "what does CFG
sample" has a complete answer, and the answer is: a distribution
strictly narrower than the conditional, centered past it. Every
empirical CFG artifact has its caricature here.

## When To Guide: The Gap Bound

For non-Gaussian data the per-level tilt and the sampler diverge
(foundations/05), with first-order error the Jensen gap
`\log E[r|x_t] - E[\log r|x_t]`, `r = p_0(x_0|y)/p_0(x_0)`. That gap
obeys a clean bound:

**Proposition.** If `\log r(x_0) \in [a, b]` on the support of the
denoising posterior at `(x, t)`, then

```math
0 \;\le\; \log \mathbb{E}[r\,|\,x_t = x] - \mathbb{E}[\log r\,|\,x_t = x] \;\le\; \frac{(b-a)^2}{8}.
```

*Proof.* Hoeffding's lemma applied to the bounded variable
`L = \log r` under the posterior:
`\log E[e^{L}] \le E[L] + (b-a)^2/8`. ∎

Read as a schedule: the oscillation of `\log r` over the posterior is
what guidance pays. Late (`t \to 0`) the posterior concentrates,
`b - a \to 0`, CFG is locally consistent — and also unnecessary
(conditioning is already decided). Early (`t \to T`) the posterior is
the whole prior, `b - a` is maximal — the gap is largest exactly
where `01`'s plug-in error is largest, one mechanism. Mid-trajectory
is where conditioning information actually transfers per unit of
inconsistency. That is interval guidance (apply `\omega > 1` only on
a middle band — Kynkäänniemi et al.), arriving as a corollary of the
gap's profile rather than as a trick; their empirical band and the
gap-profile reasoning agree (statement). Rescaled/normalized CFG
variants (renormalizing the guided noise prediction's magnitude) are
projections of the same repair — bound the modification where it
would be largest (statements).

## The `\omega \to \infty` Endpoint

Divide the guided drift by `\omega`: the surviving field is
`g^2\nabla\log h_t` — gradient ascent on the noisy classifier
likelihood. The limit sampler is not sampling at all: it is
(noise-annealed) MAXIMIZATION of `p_t(y|x)`, i.e. it targets the
classifier's argmax set. In the Gaussian theorem this is visible as
`\Lambda_\omega \to \infty` along `\Lambda_c - \Lambda_u` (variance
to zero: a point mass) — and in the mixture picture the point mass
lands on the dominant mode: **diversity loss under strong guidance is
not an artifact but the limit's definition.** The practical `\omega`
is an interpolation between sampling the conditional (`\omega = 1`)
and optimizing the classifier (`\omega = \infty`) — which is exactly
the reward-tilting dial of `04` with `p(y|x_0)` as the reward and
`1/(\omega-1)`-type temperature; the two files describe one object,
conditioning-flavored and control-flavored.

## Load-Bearing Audit

```text
joint Gaussianity          the exact family: both the commutation
                           (foundations/05) and Sigma_c <= Sigma_u;
                           the mixture reality inherits these
                           per-branch as approximations;
bounded log r              the Hoeffding bound; unbounded likelihood
                           ratios (deterministic labels) make the gap
                           bound vacuous exactly where guidance is
                           most aggressive — honest;
first-order regime         as in foundations/05: the gap governs
                           moderate omega; omega = 7.5 is described
                           only by the Gaussian caricature and the
                           infinite endpoint, with the middle open;
per-t independence         the tilt is per-level; all statements
                           about the SAMPLER compose levels through
                           the solver — the compounding is priced
                           qualitatively (05), not exactly.
```

## Position In The Coordinate System

The estimand-modification dial, now with its endpoints and its
schedule characterized: `\omega = 1` exact conditioning (`01`),
finite `\omega` a sharpened-overshooting family (exact in the
Gaussian case), `\omega = \infty` classifier ascent, and the
noise-level profile of the inconsistency bounded by an oscillation.
Retrofit landed at `score_foundations/05`: its open list pointed
here; what remains open there now genuinely remains open here.

## What Remains Open

The nonperturbative middle (`\omega` large but finite, non-Gaussian
data): no description of the sampled law exists beyond bounds; the
correct notion of "effective omega" after solver compounding (the
same guidance strength acts differently through ODE vs SDE —
`samplers_and_convergence/05`'s dichotomy, unquantified for guided
drifts); and principled per-(t, x) guidance schedules: the gap bound
suggests gating on posterior oscillation, which is estimable from the
model itself — an obvious, untried, and unproved algorithm.
