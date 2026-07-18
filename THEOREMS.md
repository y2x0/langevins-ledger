# Theorem Index

One line per result proved in full, with the file that carries the
proof. Grows as phases land (see PLAN.md). Companion indexes:
[bellmans-ledger](https://github.com/y2x0/bellmans-ledger/blob/main/THEOREMS.md),
[attention-ledger](https://github.com/y2x0/attention-ledger/blob/main/THEOREMS.md).

## score_foundations/

| Result | File |
|---|---|
| The VP transition kernel `x_t = α_t x_0 + σ_t ε`, via integrating factor | 01 |
| `W₂(p_t, γ) ≤ α_t W₂(p_0, γ)` — exponential forgetting, by synchronous coupling | 01 |
| The DDPM chain reproduces the SDE's marginals exactly (no discretization error forward) | 01 |
| Tweedie's formula `E[x₀|x_t] = (x_t + σ²∇log p_t)/α_t` | 02 |
| The estimand dictionary: score ↔ noise ↔ denoiser ↔ velocity, affine bijections | 02 |
| Vincent's theorem: denoising regression = score matching (conditional-expectation projection) | 02 |
| Anderson's time reversal, proved at the Fokker–Planck level | 03 |
| The probability-flow ODE shares every marginal (continuity equation) | 03 |
| The λ-family: every noise level between ODE and SDE is an exact sampler | 03 |
| The DDPM ancestral step is an exact Gaussian posterior (μ̃, σ̃² derived) | 04 |
| The ELBO = per-step KLs = SNR-weighted noise regression (weights derived) | 04 |
| DDIM = the exact PF-ODE solution under a frozen denoiser (`du/dρ = (u−x̂₀)/ρ`) | 04 |
| Classifier guidance is Bayes: `∇log p_t(x|y) = ∇log p_t(x) + ∇log p_t(y|x)` | 05 |
| CFG's score is the gradient of the per-t geometric tilt `p_t(x|y)^ω p_t(x)^{1−ω}` | 05 |
| The Jensen-gap theorem: the CFG family is not the noised path of any tilted data law | 05 |
| CFG is exact for jointly Gaussian data (tilting commutes with the semigroup) | 05 |
| Gaussian data: score, Wiener-filter denoiser, and sampler all exact in closed form | 06 |
| The empirical measure's optimal denoiser is softmax attention over the training set | 06 |
| The memorization theorem: exact-score sampling on n points outputs the n points | 06 |

## The Recurring Instruments (so far)

```text
insert-the-kernel-and-divide   Tweedie (02), Vincent's cross term (02),
                               the mixture ratio E[r|x_t] (05), the
                               empirical posterior (06) — one
                               manipulation, four theorems
p ∇log p = ∇p                  Anderson's reversal, the PF-ODE, the
                               λ-family (03) — the score's whole job
the exponential tilt           classifier-free guidance (05): the same
                               Legendre/tilting transform as
                               bellmans-ledger rlhf/02 and
                               attention-ledger's softmax — now
                               indexed by noise level, where it stops
                               being consistent (the Jensen gap)
softmax retrieval              the empirical score (06) IS
                               attention-ledger foundations/01's
                               kernel smoother; temperature = σ_t²
```
