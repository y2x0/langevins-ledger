# The Masked-Diffusion ELBO

## The Question

The masked forward chain (`01`) has a two-point marginal per token;
its variational training objective should be correspondingly simple,
and it is: this file proves the MDLM-style simplification — the
entire ELBO collapses to a SCHEDULE-WEIGHTED MASKED-LANGUAGE-MODELING
LOSS — and its striking corollary, schedule invariance: after a
change of variables the objective does not depend on the masking
schedule at all. Masked language modeling (BERT's objective) is
thereby a slice of a principled likelihood bound, and the phase's
bridge to attention-ledger is load-bearing from here on: the
denoiser is a bidirectional transformer predicting tokens at masked
positions.

## The Per-Token Posterior, Solved

Discrete-time grid, per-token survival probabilities
`\alpha_0 = 1 > \alpha_1 > \dots > \alpha_K` (`01`). The DDPM-style
ELBO needs `q(x_{k-1}|x_k, x_0)` — masking's absorbing structure
makes it a two-case computation:

**Lemma.** Per token `\ell`, given `(x_k^\ell, x_0^\ell)`:

```text
x_k^ell = x_0^ell (unmasked):  then x_{k-1}^ell = x_0^ell surely
                               (absorption: unmasked later implies
                               unmasked earlier);
x_k^ell = M (masked):          x_{k-1}^ell = M        w.p. (1-alpha_{k-1})/(1-alpha_k),
                               x_{k-1}^ell = x_0^ell  w.p. (alpha_{k-1}-alpha_k)/(1-alpha_k):
                               the posterior either KEEPS the mask or
                               REVEALS THE TRUE TOKEN — nothing else.
```

*Proof.* Bayes on the two-point forward marginals: the token is
masked by `k-1` and still masked at `k` with probability
`1-\alpha_{k-1}` (masked stays masked), and unmasked at `k-1` but
masked at `k` with probability `\alpha_{k-1} - \alpha_k`; normalize
by `1 - \alpha_k`. ∎

## The ELBO Collapses To Weighted Cross-Entropy, Proved

The generative model mirrors the posterior's structure with the same
schedule constants, replacing the unknown `x_0^\ell` by the network's
token distribution `p_\theta(\cdot\,|\,x_k)`:

**Theorem.** The per-step KL terms of the ELBO reduce, per masked
token, to a weighted cross-entropy:

```math
\mathrm{KL}_k
\;=\;
\sum_{\ell:\,x_k^\ell = M}\ \frac{\alpha_{k-1}-\alpha_k}{1-\alpha_k}\ \big(-\log p_\theta\big(x_0^\ell\,\big|\,x_k\big)\big),
```

so the full objective is

```math
-\,\mathrm{ELBO}
\;=\;
\sum_k\ \mathbb{E}_{x_k}\ \sum_{\ell:\,x_k^\ell = M}
\frac{\alpha_{k-1}-\alpha_k}{1-\alpha_k}\,\big(-\log p_\theta(x_0^\ell|x_k)\big)
\;\xrightarrow[\text{cont. limit}]{}\;
\int_0^1 \frac{-\dot\alpha_t}{1-\alpha_t}\ \mathbb{E}_{x_t}\!\!\sum_{\ell:\,x_t^\ell=M}\!\!\big(-\log p_\theta(x_0^\ell|x_t)\big)\,\mathrm{d}t .
```

*Proof.* Unmasked tokens contribute zero (both posterior and model
are the same deterministic copy). For a masked token, both
distributions put the SAME schedule-determined mass `s =
(1-\alpha_{k-1})/(1-\alpha_k)` on keep-mask (that term's KL
contribution is zero), and the reveal branch compares
`\delta_{x_0^\ell}` against `p_\theta(\cdot|x_k)` with weight `1-s`:
`(1-s)\,\mathrm{KL}(\delta_{x_0^\ell}\|p_\theta) =
(1-s)(-\log p_\theta(x_0^\ell|x_k))`. Sum; take the grid limit. ∎

**Masked language modeling, located exactly.** The inner expectation
at fixed `t` is BERT's loss at masking ratio `u = 1-\alpha_t`
(predict the originals at masked positions, given the rest). So: MLM
= one integrand slice of a likelihood bound; masked diffusion = MLM
integrated over masking ratios with weight `-\dot\alpha/(1-\alpha)`;
and a single bidirectional transformer trained this way IS a
generative model with an ELBO — the attention-ledger bridge in its
sharpest form: **the masked LM was a one-step discrete denoiser all
along**, and generation is iterating it (`04`–`05` for what iterating
costs and buys).

## Schedule Invariance, Proved

**Corollary.** Substitute the masking ratio `u = 1-\alpha_t` as the
integration variable (`du = -\dot\alpha\,dt`):

```math
-\,\mathrm{ELBO}
\;=\;
\int_0^1 \frac{1}{u}\ \mathbb{E}_{\text{mask each token iid w.p. } u}\ \sum_{\ell\ \text{masked}}\big(-\log p_\theta(x_0^\ell\,|\,x_u)\big)\ \mathrm{d}u :
```

every schedule with the same endpoints yields the SAME objective —
the schedule is a time reparametrization the loss cannot see (MDLM's
invariance result, now a one-line substitution). ∎

The reading: for masked diffusion, the path coordinate `P`
degenerates — only the masking-ratio MEASURE ever mattered, and it is
fixed to `du/u` by the ELBO itself. Design effort moves to the
sampler (how many tokens to reveal per step, `05`) and the estimand's
conditioning (`03`). The `1/u` weight is worth a sentence: low
masking ratios get LARGE weight per masked token but few masked
tokens (the expected count is `Lu`, cancelling to an `O(L)` density)
— the objective spreads evenly over ratios, which is why training at
a uniform random ratio (the practical recipe) is exactly right rather
than a convenience.

## Load-Bearing Audit

```text
absorbing structure       the two-case lemma: "unmasked implies always
                          was" kills half the case analysis, and
                          "reveal = truth" makes the KL a cross-entropy
                          — uniform-chain ELBOs keep both complications
                          and stay messier (statement);
shared schedule constants the keep-mask branches cancel ONLY because
                          the model uses the forward's constants — a
                          modeling choice, standard and load-bearing;
factorized reveal         p_theta reveals tokens independently given
                          x_k — invisible at the ELBO level (each KL
                          term is per-token), decisive at SAMPLING
                          (05's total-correlation tax);
grid limit                monotone convergence of the Riemann sums;
                          nothing subtle.
```

## Position In The Coordinate System

The estimand made concrete: for the masking path, `s` = the
conditional token posteriors at masked positions — a bidirectional
transformer's native output — and the objective is their
cross-entropy under a canonical ratio measure. One family, all three
repositories: the estimand is trained by the same
conditional-expectation logic as everything else (`03` proves the
discrete projection), computed by attention (attention-ledger's
machinery), toward a likelihood bound (this file).

## What Remains Open

The gap between the ELBO and sample quality, in its discrete
costume (the weighting that optimizes likelihood is fixed by the
corollary; the weighting that optimizes generation quality is not,
and practice deviates — the third repository where this gap appears
and stays open); and conditioning richer than
masks (`p_\theta(x_0^\ell|x_t)` sees a PARTIAL sequence: the
ragged-context estimation burden is now posed and partly answered in
`statistical_theory/08` — the table estimator's rate is
`(|V|-1)m/2n` in the well-sampled regime `n\gtrsim m\log m` and
minimax-bad below it, which is exactly this burden quantified; the
structured-denoiser rate that beats the table is the open remainder).
