# Theorem Index

One line per result proved in full, with the file that carries the
proof. All six phases of PLAN.md are complete. Companion indexes:
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

## samplers_and_convergence/

| Result | File |
|---|---|
| Exact EM discretization bias in the Gaussian case: `v* = 1/(1 − βh/4)` | 01 |
| The quadrature identity: exact PF-ODE solution = `ρ⁻²`-weighted integral of `x̂₀` | 02 |
| DDIM is the zeroth-order quadrature rule; second-order error constants derived | 02 |
| The Girsanov KL decomposition: prior + score error + discretization, additive | 03 |
| The λ-family error dynamics: ODE accumulates bias `b·B/2`, SDE saturates at `2b` | 05 |
| Robustness is bought quadratically in λ (restoring coefficient `−λ²β/2`) | 05 |
| Langevin stationarity of `p`; de Bruijn's identity `d/dt KL = −I` | 06 |
| LSI ⇒ exponential KL decay of Langevin (Grönwall chain) | 06 |

## flow_matching/

| Result | File |
|---|---|
| The marginal velocity generates the path (continuity equation, weak form) | 01 |
| The CFM identity: per-pair regression = marginal-velocity regression | 01 |
| The interpolant score formula `s = −E[z|x_t]/γ_t` (latent buys a score) | 02 |
| Every interpolant carries a λ-dial of exact SDE samplers | 02 |
| Rectification preserves marginals (under well-posedness) | 03 |
| Every convex transport cost weakly decreases under rectification (double Jensen) | 03 |
| Straight couplings are exactly the fixed points; one Euler step exact on them | 03 |
| Marginal preservation FAILS for the atomic crossing coupling (worked) | 03 |
| The smoothed crossing rectifies to the monotone coupling (odd-symmetry proof) | 03 |
| The dictionary: `v* = (β̇/β)x + γ(β̇γ/β − γ̇)s` on Gaussian paths | 04 |
| Velocity stays regular where the score blows up (endpoint cancellation) | 04 |
| In d = 1 the FM flow map is monotone, hence optimal transport | 05 |
| The FM path ≠ the OT geodesic (the two-atom bulge, exact) | 05 |

## guidance_and_control/

| Result | File |
|---|---|
| `h_t(x_t)` is a reverse-time martingale (tower property) | 01 |
| The h-transform adds `g²∇log h` to the drift (generator computation) | 01 |
| Exact conditioning = the Doob h-transform of the reverse process | 01 |
| The plug-in guidance error is exactly `g²∇log(h/ĥ)` — expectation vs plug-in | 01 |
| The Gaussian ω-family exactly: `Λ_ω = ωΛ_c + (1−ω)Λ_u`, mean extrapolation | 02 |
| CFG variance shrinks below the conditional, monotonically in ω | 02 |
| The Jensen gap ≤ `(b−a)²/8` (Hoeffding's lemma under the posterior) | 02 |
| ω → ∞ is classifier ascent (sampling degenerates into optimization) | 02 |
| Exact inverse-problem guidance for Gaussian priors: `S_t = A C_t Aᵀ + σ_y²I` | 03 |
| DPS deletes the posterior covariance — overweighting ratio computed (×51) | 03 |
| KL-regularized reward fine-tuning: closed form `dQ*/dP ∝ e^{r/β}` (path space) | 04 |
| The optimum is the h-transform of the pretrained sampler by `E[e^{r/β}|x_t]` | 04 |
| `V = β log h` solves the HJB; Hopf–Cole linearizes it (linearly solvable control) | 04 |
| Path KL = expected quadratic control cost (Girsanov dictionary) | 04 |
| The composite theorem: KL ≤ prior + C_λ∫g²[score + plug-in + (ω−1)²·Fisher] | 06 |
| `C_λ = 3(1+λ²)²/8λ²` uniquely minimized at λ = 1 (the SDE is the robust point) | 06 |

## distillation/

| Result | File |
|---|---|
| Consistency + boundary characterize the solution map (transport equation, characteristics) | 01 |
| Linear accumulation: N consistency leaks ADD (the telescope) | 01 |
| The skip parametrization enforces the boundary identically | 01 |
| CD's fixed point is the teacher's map; solver order inherited as bias `O(h^p)` | 02 |
| The CT input's conditional mean = the exact-score DDIM step | 02 |
| CT's bias is a Jensen gap: `½∇²f[Cov]` with `Cov = O(h²)` | 02 |
| The progressive-distillation target: the effective denoiser, inverted in closed form | 03 |
| Halving is lossless for isotropic Gaussian data (frozen denoiser) | 03 |
| Guided flow maps exist and distill soundly; only the law's pedigree stays open | 04 |

## discrete_diffusion/

| Result | File |
|---|---|
| The discrete Anderson theorem: reverse rates `Q̄(y,x) = Q(x,y)p(x)/p(y)` | 01 |
| The masking chain solved: two-point marginals, survival `α_t` | 01 |
| The masked posterior: keep the mask or reveal the truth (two cases) | 02 |
| The ELBO collapses to schedule-weighted masked cross-entropy (MDLM) | 02 |
| Schedule invariance: the `du/u` substitution | 02 |
| Discrete Tweedie: the ratio = `α/(1−α)` × the masked-token posterior | 03 |
| The Bregman projection lemma: every Bregman regression learns `E[A|X]` | 03 |
| Score entropy = the KL-Bregman instance; positivity by loss geometry | 03 |
| Masked diffusion ELBO = expected any-order AR likelihood (Beta integral) | 04 |
| Left-to-right AR = the deterministic suffix-mask schedule | 04 |
| The parallel-decoding tax = conditional total correlation (exact KL identity) | 05 |
| The two-token worked case: one-step reveal puts half its mass on impossible strings | 05 |

## statistical_theory/

| Result | File |
|---|---|
| The W₂ stability bound: Grönwall over the synchronous coupling | 01 |
| The Gaussian error propagator, exact (variation of constants over λ) | 01 |
| Early stopping = kernel density estimation at bandwidth `σ_t/α_t` (exact reduction) | 02 |
| The classical KDE rate in 1-D: `MISE(h*) = Θ(n^{−4/5})`, proved | 02 |
| The generalization trichotomy: smoothing, estimation, solver — no other channel | 03 |
| Linear score classes: the learned score = the projection of the empirical score | 03 |
| The two-point formula `x̂₀ = a·tanh(ax/σ²)` and the σ⁻² score blowup | 04 |
| Denoiser regularity: flat off the medial axis, σ⁻² spike on it (one derivative) | 04 |
| The subspace case: normal score exactly `−x⊥/σ²`; denoiser = bounded projection | 04 |
| Guidance displacement `x⊥* = ωcσ_t²` (the failure-mode toy, promoted) | 04 |
| The mode posterior is a λ=1 reverse-time martingale; the `λ²−1` defect computed | 06 |
| The mode-weight bound: `|ŵ−w| ≤` score error × the mode-Fisher budget | 06 |
| The budget is O(1) and spent at the merge scale (one bit, one window) | 06 |
| The optimal grid: density ∝ `βe^{−λ²B/2}`; error `(1−(α_T/α_min)^{λ²})²/K`, schedule-invariant | 07 |
| The minimax training weight = the SAME measure (the propagator profile; saddle point) | 07 |
| The CFM floor = the angular energy `∫ρ²θ̇²` (the Wronskian identity, polar) | 07 |
| The trigonometric path uniquely minimizes the VP floor (value `π²/4`) | 07 |
| The endpoint dichotomy: velocity floor and score target never both integrable; the bridge rate is critical | 07 |
| The discrete excess-risk identity: masked loss − data entropy = order-averaged conditional KL | 08 |
| The multinomial KL floor `(|V|−1)/2n_R` (Pearson's exact mean) | 08 |
| The conditional count: any-order/AR ratio = `L(1+|V|⁻¹)^{L−1}` (≈L large-vocab, exp. small) | 08 |
| The reveal-schedule identity `T = C − Σ I(x_ℓ; predecessors)`; serial is the zero-tax optimum | 08 |

## The Recurring Instruments

```text
insert-the-kernel-and-divide   Tweedie (02), Vincent's cross term (02),
                               the mixture ratio E[r|x_t] (05), the
                               empirical posterior (06), the
                               interpolant score (flow_matching/02) —
                               one manipulation, five theorems
the Bregman projection lemma   the series' projection trick, now ONE
                               theorem (discrete_diffusion/03): L2
                               gives Vincent (foundations/02), CFM
                               (flow_matching/01), and the CT identity
                               (distillation/02); the KL-Bregman gives
                               score entropy — every estimand in the
                               repository is trainable by one lemma
                               at some convex phi
p ∇log p = ∇p                  Anderson's reversal, the PF-ODE, the
                               λ-family (foundations/03), Langevin
                               stationarity (samplers/06) — the
                               score's whole job, four theorems
the exponential tilt           CFG (foundations/05), reward fine-tuning
                               on path space (guidance/04 — the SAME
                               one-line proof as bellmans-ledger
                               rlhf/02), and Todorov's desirability
                               z = e^{V/β} (guidance/04 = bellmans
                               lqr/04): one transform, three ledgers,
                               and the noise index is where it stops
                               being consistent (the Jensen gap)
h = E[terminal | x_t]          the h-transform triple: conditioning
                               (guidance/01), measurements
                               (guidance/03), rewards (guidance/04) —
                               one martingale, three steering problems
softmax retrieval              the empirical score (06) IS
                               attention-ledger foundations/01's
                               kernel smoother; temperature = σ_t²
```
