# Score Entropy: The Discrete Estimation Theory

## The Question

`01` identified the discrete estimand — marginal ratios
`r_t(x,y) = p_t(y)/p_t(x)` along the forward chain's edges. Two
things are owed: the discrete Tweedie (what the ratios reduce to
under masking — the answer wires this phase to `02`), and the
discrete Vincent (how ratios are learnable from samples). The second
answer is the file's centerpiece and one of the series' better
unifications: the projection trick that carried Vincent, CFM, and
the CT identity is a special case of a ONE-LINE lemma about Bregman
divergences — proved here, then instantiated to score entropy.

## Discrete Tweedie: The Ratio Is A Posterior, Proved

Masked forward chain (`01`), sequence state `x` with position `\ell`
masked; the reverse rates (`01`) need the ratio between `x` and its
unmasking neighbors `y_v = (x^{-\ell}, v)`, `v \in V`.

**Theorem.**

```math
\frac{p_t\big(y_v\big)}{p_t\big(x\big)}
\;=\;
\frac{\alpha_t}{1-\alpha_t}\ \ \mathbb{P}\big(x_0^\ell = v\ \big|\ \text{the evidence } x\big),
```

where the conditioning is on the observed partial sequence (unmasked
tokens equal their originals, masked positions masked).

*Proof.* Both numerator and denominator are mixtures over `x_0` of
factorized per-token forward probabilities. Every factor for
positions `\ne \ell` is common. At `\ell`: the numerator requires the
token to have SURVIVED with value `v` — probability `\alpha_t`, and
only `x_0` with `x_0^\ell = v` contribute; the denominator requires
it masked — probability `1-\alpha_t`, contributed by every `x_0`.
The quotient of the two mixture sums is
`\tfrac{\alpha_t}{1-\alpha_t}` times the ratio of (posterior mass
with `x_0^\ell = v`) to (total posterior mass). ∎

So the discrete score is a SCHEDULE FACTOR times exactly the
masked-LM prediction — `02`'s estimand, rederived from the reversal
side: the two training philosophies (ELBO cross-entropy, score/ratio
matching) target the same conditional posterior, as they must. This
is Tweedie's discrete face: there, score `\leftrightarrow` posterior
mean; here, ratio `\leftrightarrow` posterior probability — in both
cases the reverse dynamics runs on a conditional expectation the
forward kernel makes learnable.

## The Bregman Projection Lemma, Proved Once For The Series

**Lemma.** Let `\varphi` be strictly convex differentiable,
`D_\varphi(a, b) = \varphi(a) - \varphi(b) - \varphi'(b)(a-b)` its
Bregman divergence. For any random target `A` and prediction
variable `z`:

```math
\arg\min_z\ \mathbb{E}\,\big[D_\varphi(A, z)\big] \;=\; \mathbb{E}[A],
```

and more generally, predicting from side information `X`,
`\arg\min_{z(\cdot)} E[D_\varphi(A, z(X))] = E[A|X]` — the
conditional mean, for EVERY Bregman divergence.

*Proof.* `E[D_\varphi(A,z)] - E[D_\varphi(A, E A)] =
\varphi(EA) - \varphi(z) - \varphi'(z)(EA - z) = D_\varphi(EA, z)
\ge 0`, with equality iff `z = EA` (strict convexity) — the
`A`-linear terms cancel because `D_\varphi` is affine in its first
argument up to `\varphi(a)`. Condition on `X` pointwise for the
general case. ∎

One line, and the series' recurring instrument becomes a single
theorem with instances:

```text
phi(a) = ||a||^2      L2: Vincent (foundations/02), CFM
                      (flow_matching/01), CT's identity
                      (distillation/02);
phi(a) = a log a - a  the generalized-KL/I-divergence Bregman:
                      D(a,b) = a log(a/b) - a + b — SCORE ENTROPY's
                      per-edge loss, this file;
any strictly convex   the license: ANY Bregman regression against
                      per-sample surrogates learns the conditional
                      expectation of the surrogate.
```

## Score Entropy, Instantiated

Score entropy (Lou–Meng–Ermon) trains ratio estimates
`r_\theta(x, y) > 0` with the per-edge loss (weights `w_{xy}` from
the forward rates)

```math
\ell(x, y) \;=\; r_\theta(x,y) \;-\; r^*(x,y)\,\log r_\theta(x,y) \;+\; c\big(r^*\big),
```

which is exactly `D_\varphi(r^*, r_\theta)` for
`\varphi(a) = a\log a - a` (expand the definition: the `a\log a`
terms sit in `c(r^*)`). Consequences, each now immediate from the
lemma:

```text
well-posedness   strictly convex in r_theta, unique minimum at the
                 true ratio, and positivity of r_theta is enforced by
                 the domain of phi — the reason this loss and not L2
                 (ratios must be positive; L2 lets estimates cross
                 zero and the reverse rates explode);
denoising form   r*(x,y) is intractable, but it is a conditional
                 expectation of per-sample ratios (Tweedie above:
                 posterior probabilities are conditional expectations
                 of indicators of x_0): replace the target by the
                 per-sample surrogate and the Bregman lemma projects
                 it back — the discrete Vincent, for free;
ELBO link        with the masking chain, minimizing score entropy and
                 minimizing 02's weighted cross-entropy estimate the
                 same posterior (this file's theorem); SEDD's
                 likelihood bounds (statement) make the connection
                 quantitative on their side.
```

## Load-Bearing Audit

```text
factorized forward kernel   discrete Tweedie's common-factor
                            cancellation; correlated corruption
                            breaks the per-position reduction;
masking chain               the specific posterior form; uniform-
                            chain ratios involve likelihood ratios of
                            v-vs-current-token — same lemma, messier
                            surrogates (statement);
strict convexity of phi     uniqueness in the lemma; boundary cases
                            (phi affine on a region) lose
                            identifiability exactly there;
positivity of ratios        why the KL-Bregman and not L2 — an
                            audit item that is a DESIGN THEOREM:
                            the loss geometry is chosen by the
                            estimand's constraint set.
```

## Position In The Coordinate System

The estimand coordinate on discrete spaces, completed: the reversal
needs ratios, the ratios are posteriors (Tweedie), the posteriors are
conditional expectations, and conditional expectations are learnable
by ANY Bregman regression on per-sample surrogates (the lemma). The
estimation theory of this entire repository — continuous and
discrete — is one lemma instantiated at three convex functions.

## What Remains Open

The finite-sample side (phase F poses it for continuous scores;
discrete ratio estimation over `|V|^L` states with a transformer has
no rates at all); loss geometry beyond Bregman (are there
non-Bregman losses with the projection property adapted to sequence
structure? the lemma's converse — projections characterize Bregman —
is classical, so the answer is no WITHIN point-prediction, but
set-valued and quantile analogues are unexplored here); and the
uniform-chain estimation problem, which practice abandoned for
masking partly on estimation-difficulty grounds that have never been
made precise.
