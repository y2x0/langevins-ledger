# Manifold Geometry: Where The Score Blows Up

## The Question

Real data concentrates near low-dimensional structure, and every
phase of this repository carried the same audit item: something
degrades as `t \to 0` on supported data. This file proves what, at
what rate, and for which estimand — all from one two-point
computation whose single formula yields the `\sigma^{-2}` score
blowup, the `O(1)` regularity of the denoiser AWAY from decision
boundaries, and the `\sigma^{-2}` spike of the denoiser ON them. The
consequences are then collected: the `t_{\min}` cutoff as a theorem,
the parametrization ladder, and the quantitative form of guidance's
manifold-leaving.

## The Two-Point Case, One Formula

Data `\tfrac12\delta_{-a} + \tfrac12\delta_{+a}` in one dimension
(the minimal "manifold": two zero-dimensional components), noise
level `\sigma` (working at `\alpha = 1` for clarity; schedules
rescale). The smoothed density is a two-Gaussian mixture, and every
object of interest is elementary:

**Theorem.** The denoiser and score are exactly

```math
\hat x_0(x) \;=\; a\,\tanh\!\Big(\frac{a\,x}{\sigma^2}\Big),
\qquad
s(x) \;=\; \frac{\hat x_0(x) - x}{\sigma^2},
```

and consequently:

```text
(i)   score blowup, sigma^{-2}: for fixed x not an atom,
      |s(x)| -> |(-x ± a)|/sigma^2 -> infinity as sigma -> 0
      (tanh saturates to the NEAREST atom's sign): the score's
      magnitude off the support diverges at exactly rate sigma^{-2};
(ii)  denoiser regularity off the ridge: dx0-hat/dx =
      (a^2/sigma^2) sech^2(ax/sigma^2); for |x| >= delta > 0 this is
      <= (a^2/sigma^2) e^{-2a delta/sigma^2} -> 0: the denoiser is
      FLAT (Lipschitz -> 0) away from the midpoint — it locks onto
      the atom;
(iii) denoiser spike on the ridge: at x = 0 (equidistant — the
      medial axis), dx0-hat/dx = a^2/sigma^2 -> infinity: ALL the
      denoiser's Lipschitz constant concentrates on the decision
      boundary, at the score's own rate.
```

*Proof.* Tweedie plus the two-point posterior
(`score_foundations/06`'s softmax with two atoms is the tanh);
differentiate the tanh once for (ii)/(iii); take the stated limits. ∎

The linear-subspace generalization is equally exact and worth its
display: data on a subspace `V` (Gaussian along `V`), noise
`\sigma`: the normal component of the score is `-x_\perp/\sigma^2`
EXACTLY (the normal factor is a pure Gaussian), while the denoiser's
normal action is the projection `x \mapsto x_V` — bounded Lipschitz,
uniformly in `\sigma`. Curved manifolds inherit both statements
locally, with the medial axis (points equidistant to distinct
patches) playing `x = 0`'s role (statement; the two-point case is its
transverse model).

## Consequence 1: The `t_{\min}` Cutoff Is A Theorem

Every phase assumed it; (i) proves it necessary: the score's
`L^\infty` and Lipschitz scales diverge as `\sigma^{-2}`, so
Girsanov's Novikov condition (`A/03`), the discretization constants
(`A/01`), the Grönwall budget (`01`), and the consistency-map
regularity (`distillation/01`) all fail AT `t = 0` on supported data
— and all hold on `[t_{\min}, T]` with constants polynomial in
`1/\sigma_{t_{\min}}`. Early stopping is not a hack: it is the
statement that supported data has no density to sample, and `02`'s
reduction says what stopping samples instead (the KDE). One cutoff,
five theorems' hypotheses, now sourced.

## Consequence 2: The Parametrization Ladder

The estimands (`score_foundations/02`) are affinely equivalent as
targets but NOT as functions to represent — the two-point formulas
grade them:

```text
score s              magnitude sigma^{-2} off-support: the WORST
                     object to regress near t_min;
noise eps-hat        = -sigma s: magnitude sigma^{-1}-scale: better
                     by one power;
denoiser x0-hat      bounded range (the convex hull!), Lipschitz O(1)
                     except on the medial axis: the tamest estimand —
                     and flow_matching/04's velocity inherits its
                     regularity (the endpoint cancellation, now with
                     rates).
```

The practical stack — `x_0`/`v`-prediction at low noise,
`\varepsilon`-prediction at high — is this ladder read as a schedule;
the residual irreducible difficulty is (iii): NO parametrization
tames the medial-axis spike, because it is real structure (the
sampler must DECIDE between branches there), not coordinate
artifact. Few-step methods' failure location (`distillation/04`'s
curvature) is exactly this set.

## Consequence 3: Guidance's Displacement, Quantified

`guidance_and_control/05`'s mechanism, now with the constant proved:
against a guidance pull `\omega c`, the stationary off-support
displacement solves `s(x) + \omega c = 0`, and by the subspace
formula `x_\perp^* = \omega c\,\sigma_t^2` — restoring force
`\sigma^{-2}`, displacement `\propto \omega\sigma_t^2`: large early,
negligible late, transported or healed per the solver
(`samplers_and_convergence/05`). The failure-modes file's toy is
hereby promoted to a computation.

The constructive readings close the file: the blowup is also a
SIGNAL — the score Jacobian's spectrum separates `\sigma^{-2}`-normal
from `O(1)`-tangent directions, making local intrinsic dimension
readable from a trained model (statement: the score-based
dimension-estimation literature); and one-sided interpolants
(`flow_matching/02`) with `\gamma > 0` at the data end are exactly
blowup management — a design dial this file finally prices.

## Load-Bearing Audit

```text
two-point / subspace exactness   the theorem is the transverse
                                 caricature; curvature adds O(1)
                                 tangential terms (statement) but the
                                 normal rates are the caricature's;
alpha = 1 normalization          cosmetic; schedules reinsert
                                 alpha_t's via the (u, rho) clock;
medial axis measure zero         (iii) is a thin set — which is why
                                 AVERAGE-case bounds (A/03's L2)
                                 survive while UNIFORM bounds
                                 (Lipschitz, Novikov) die: the
                                 phase's two bound-styles part ways
                                 exactly here;
Gaussian normal factor           the subspace exactness.
```

## Position In The Coordinate System

The geometry that prices all three coordinates at once: the path's
endpoint (`t_{\min}`, consequence 1), the estimand's choice
(consequence 2), and the solver-plus-steering budget (consequence 3)
are all set by one exponent — the `\sigma^{-2}` of the normal score
— computed here from a tanh.

## What Remains Open

Curved-manifold constants (reach, curvature, and medial-axis geometry
entering the rates — the transverse model ignores them; the
intrinsic-dimension minimax results of `02`'s riders are the current
best); the medial-axis spike's interaction with LEARNED scores
(networks smooth (iii) — is that a fourth generalization channel or
an error? nobody has separated them); and schedule design against
the ladder — a principled `\gamma_t`/estimand schedule minimizing
represented-function hardness for given support geometry, the
design problem this file makes well-posed and leaves unsolved.
