# Minimax Rates: Diffusion As Density Estimation

## The Question

Strip away the networks and the solvers and ask the statistician's
question: given `n` samples from `p_0`, how well can ANY procedure
learn to sample it — and do diffusion models achieve that limit? The
modern answer (Oko–Akiyama–Suzuki and successors) is yes, minimax
rates over smoothness classes, up to logs. This file proves the
reduction that makes the claim legible — **the empirical-score
diffusion sampler stopped at `t` is EXACTLY a kernel density
estimator with bandwidth `\sigma_t`** — and then proves the classical
KDE rate in one dimension in full, so the minimax statement rests on
worked mathematics rather than citation. The learned-network upgrade
is then stated with its real content located (approximation theory),
honestly.

## The Reduction, Proved

**Theorem.** Run the EXACT reverse dynamics (any `\lambda`,
`score_foundations/03`) for the empirical measure
`\hat p_0 = \frac1n\sum_i \delta_{x^{(i)}}`, and STOP at time `t`,
outputting `x_t`-rescaled... precisely: output the sample at noise
level `t`, rescaled by `1/\alpha_t`. Its law is exactly

```math
\frac{1}{n}\sum_{i=1}^n N\Big(x^{(i)},\ \frac{\sigma_t^2}{\alpha_t^2}\,I\Big)
```

— the Gaussian kernel density estimator with bandwidth
`h = \sigma_t/\alpha_t`.

*Proof.* Exactness of the reversal: the sampler's law at level `t` is
`\hat p_t` = the empirical measure pushed through the forward kernel
(`score_foundations/06`), which is the displayed mixture after the
`1/\alpha_t` rescaling (`score_foundations/01`'s kernel). ∎

Early stopping IS bandwidth selection — `foundations/06`'s
memorization theorem (stop at `0`: output the data points) and
generalization-by-smoothing (`03`) are the endpoints of one dial, and
the dial is the oldest one in nonparametric statistics. The
`(u, \rho)` clock of `samplers_and_convergence/02` even uses the
right variable: `\rho = \sigma/\alpha` IS the bandwidth.

## The Classical Rate, Proved In One Dimension

**Theorem (KDE risk).** Let `p_0` on `\mathbb{R}` have bounded second
derivative, `K` the standard Gaussian kernel, `\hat p_h` the KDE with
bandwidth `h` from `n` i.i.d. samples. Then the mean integrated
squared error obeys

```math
\mathrm{MISE}(h)
= \underbrace{\frac{h^4}{4}\int (p_0'')^2}_{\text{bias}^2}
+ \underbrace{\frac{R(K)}{n h}}_{\text{variance}}
+ o\big(h^4 + (nh)^{-1}\big),
\qquad R(K) = \int K^2 = \tfrac{1}{2\sqrt\pi},
```

minimized at `h^* = c\,n^{-1/5}` with
`\mathrm{MISE}(h^*) = \Theta(n^{-4/5})`.

*Proof.* Bias: `\mathbb{E}\hat p_h(x) = (K_h * p_0)(x)`; Taylor
`p_0(x - hu) = p_0 - hu\,p_0' + \tfrac{h^2u^2}{2}p_0'' + o(h^2)`
under the kernel integral: the odd term vanishes (symmetry), leaving
bias `= \tfrac{h^2}{2}p_0''(x)\int u^2K + o(h^2)` (`\int u^2 K = 1`);
square and integrate. Variance: i.i.d. average of kernel evaluations:
`\mathrm{Var} = \tfrac1n\big[\tfrac1h\int K^2\cdot p_0(x) + O(1)\big]`
(substitute `u = (x-y)/h` in `\mathbb{E}K_h^2`); integrate in `x`.
Balance `h^4 \asymp (nh)^{-1}`. ∎

With the reduction, this is a statement ABOUT DIFFUSION: the
empirical-score sampler stopped at bandwidth `n^{-1/5}` estimates a
`C^2` density at rate `n^{-4/5}` in MISE — and the general
`d`-dimensional, `s`-smooth version of the same computation gives the
minimax rate `n^{-2s/(2s+d)}` (the curse of dimension in its
canonical form; classical, cited). The stopping-time schedule
`\sigma_{t_{\mathrm{stop}}} \sim n^{-1/(2s+d)}` is the translation.

## The Learned-Score Upgrade, Stated With Its Content Located

The reduction covers the EMPIRICAL score. Real models learn a network
approximation, and the theorem-grade results (Oko–Akiyama–Suzuki
2023; successors for manifold-supported and low-dimensional-structure
data) say: diffusion models with appropriately sized networks achieve
the minimax rates (up to logs) in TV/`W_1` over Besov classes —
where the proof's labor is APPROXIMATION THEORY: constructing
networks that represent the smoothed scores `\nabla\log \hat p_t` to
sufficient accuracy at every level, then invoking exactly this
repository's chain (Vincent's objective → `A/03`'s Girsanov bound →
early stopping). Statements; the chain is ours, the network
construction is theirs. Two honest riders: the rates are worst-case
over smoothness classes — they CANNOT explain why image models beat
the curse of dimension (the data is not a generic Besov ball; the
structured-data results — manifold support, hierarchical/local
structure — are the active frontier, statements); and nothing in the
minimax program constrains WHICH smoother a trained network
implements, which is `03`'s question and the honest gap between this
theory and practice.

## Load-Bearing Audit

```text
exact reversal            the reduction; solver error adds A-phase
                          terms on top (and is itself a smoother —
                          03's third exit);
i.i.d. samples            the variance computation;
C^2 / Besov smoothness    the bias term's currency; manifold data has
                          s = nothing in this scale — 04's geometry
                          replaces it;
worst-case classes        the riders above: minimax optimality and
                          practical success are claims about
                          different quantifiers.
```

## Position In The Coordinate System

The estimand's statistical price, at the solvable point: with the
empirical score, the whole `(P, s, S)` machine reduces to the oldest
estimator in the book, and its rate theory transfers verbatim. What
the reduction buys conceptually: diffusion models did not repeal
nonparametric statistics — they re-parametrized it (bandwidth =
stopping time, estimator = sampler), and every claim of beating the
classical rates is exactly a claim about data structure or inductive
bias, to be priced in `03`–`04`.

## What Remains Open

Adaptive stopping (choosing `t_{\mathrm{stop}}` from data,
Lepski-style, inside a sampler — unexplored); rates in the metrics
that matter for generation (`W_2` and perceptual gaps vs the TV/W1
theorems); and the structured-data program: minimax theory ON the
support geometries `04` describes, where the first results exist
(intrinsic-dimension rates; statements) and the definitive statement
does not.
