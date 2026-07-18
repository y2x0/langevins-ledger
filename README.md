# Langevin's Ledger

*The mathematics of diffusion models — every path priced, every
hypothesis audited.*

Third of three. Companion repositories:
[Bellman's Ledger](https://github.com/y2x0/bellmans-ledger) (the
mathematics of decisions) and
[attention-ledger](https://github.com/y2x0/attention-ledger) (the
mathematics of sequence models). Same rules: no survey files, nothing
transcribable from a blog post; every file carries at least one theorem
proved in full or a counterexample worked with explicit numbers, a
load-bearing hypothesis audit, and an honest "what remains open."
The index of proved results is [THEOREMS.md](THEOREMS.md); the
per-file contracts are in [PLAN.md](PLAN.md).

This notebook family asks:

```text
What distribution does the sampler actually draw from, how does error
in the learned score propagate through the solver, and which design
choices are theorems rather than conventions?
```

Every diffusion-family method transports a reference measure to the
data measure along a path of distributions. Methods differ in exactly
three coordinates:

```math
\big(\ \mathcal{P},\ s,\ \mathcal{S}\ \big)
```

```text
P:
    the path — which family {p_t} connects data to noise (VP/VE
    Ornstein–Uhlenbeck, flow-matching interpolants, masking chains on
    discrete spaces) and which forward process realizes it

s:
    the estimand — which one function is regressed from data (score
    grad log p_t, noise eps, denoiser E[x_0|x_t], velocity v), all
    affinely equivalent along Gaussian paths, none equivalent in loss
    weighting

S:
    the solver — how the path is traversed at sampling time (reverse
    SDE, probability-flow ODE, ancestral steps, exponential
    integrators, distilled one-step maps) and what each does to score
    error
```

Each notebook should answer:

```text
1. Which object on which coordinate is being constructed or analyzed?
2. What is proved about it — exactly, and in which metric?
3. Which hypothesis is load-bearing, and where in the proof?
4. What breaks when it is violated — worked, not gestured?
5. What does the sampler provably output, versus what it is said to?
6. What remains open?
```

## Source Texts

| Reading | Where it lives here |
|---|---|
| Anderson 1982 (time reversal) | `score_foundations/03` |
| Vincent 2011 (denoising score matching) | `score_foundations/02` |
| Ho–Jain–Abbeel 2020 (DDPM); Song et al. 2021 (SDE) | `score_foundations/01, 03–04` |
| Song–Meng–Ermon 2021 (DDIM) | `score_foundations/04` |
| Ho–Salimans 2022 (CFG); Dhariwal–Nichol 2021 | `score_foundations/05` |
| Chen–Chewi et al. 2023 (convergence) | `samplers_and_convergence/` |
| Lipman et al. 2023 (flow matching); Albergo–Vanden-Eijnden | `flow_matching/` |
| Song et al. 2023 (consistency models) | `distillation/` |
| Lou–Meng–Ermon 2024; Sahoo et al. 2024 (discrete/masked) | `discrete_diffusion/` |

## Folder Map

```text
score_foundations/             the objects, exactly
    01  the forward process: closed-form marginals proved, W2
        contraction to the prior, DDPM as exact discretization
    02  Tweedie and the estimands: the formula proved, the affine
        dictionary, Vincent's theorem (denoising = score matching)
    03  time reversal: Anderson's SDE proved at the Fokker–Planck
        level; the probability-flow ODE; the lambda-family
    04  DDPM and DDIM: the Gaussian posterior derived, the ELBO as
        weighted regression, DDIM = exact PF-ODE step (proved)
    05  guidance as tilting: classifier guidance proved; CFG's target
        is not a diffusion path — the Jensen-gap theorem
    06  the solvable cases: Gaussian data (exact), empirical data
        (the score IS attention; the memorization theorem)

samplers_and_convergence/      what the sampler provably outputs
    01  EM discretization: the exact Gaussian bias 1/(1-beta h/4);
        where the dimension enters; schedule sensitivity
    02  exponential integrators: the quadrature identity proved,
        DDIM as zeroth order, DPM-Solver's orders derived
    03  the Girsanov decomposition proved: prior + score error +
        discretization, with the audit that keeps it honest
    04  polynomial convergence assembled: Chen-Chewi and Benton et
        al. stated, every term traced, five things it does NOT say
    05  ODE vs SDE error dynamics: transport vs contraction proved
        exactly; the B/2 vs 2 amplification gap; the lambda price
    06  Langevin correctors: de Bruijn and LSI decay proved; the
        multimodal boundary and the third metastability

flow_matching/                 the path coordinate, unified and fenced
    01  continuity + CFM: the marginal-velocity theorem and the CFM
        identity proved (Vincent's trick, second appearance)
    02  stochastic interpolants: the latent buys a score (proved);
        every interpolant has a lambda-dial (proved); the safe region
    03  rectified flow: marginal preservation, the double-Jensen cost
        theorem, straight fixed points; the crossing failure and its
        odd-symmetry resolution, proved
    04  the dictionary: velocity = a x + b score with derived
        coefficients; FM = score matching reweighted; endpoint
        regularity explained
    05  what FM is not: 1-D IS optimal transport (proved); the path
        is not the geodesic (worked bulge example); straight != OT

guidance_and_control/          steering, exactly and approximately
    01  Doob h-transforms: conditioning = h-transform, proved; the
        plug-in error identified exactly
    02  CFG deep dive: the Gaussian omega-family solved (shrink +
        overshoot), the (b-a)^2/8 gap bound, the omega -> inf endpoint
    03  inverse problems: exact Gaussian guidance derived; DPS's
        deleted covariance, priced (x51 at high noise)
    04  reward fine-tuning = KL control: the path-space closed form,
        the h-transform sampler, Hopf-Cole/HJB — the trilogy's bridge
    05  failure modes: the table of causes, each traced to a theorem
    06  the composite theorem: guidance x solver x estimation in one
        KL bound; C_lambda minimized at lambda = 1 (added post-F,
        closing statistical_theory/05's problem 2)

distillation/                  collapsing the solver into one step
    01  the consistency condition characterizes the solution map
        (proved); linear accumulation — N leaks add; the free boundary
    02  CD vs CT: the fixed-point theorem and inherited solver order;
        the CT conditional-mean identity (= exact DDIM step) and its
        Jensen-gap bias
    03  progressive distillation: the effective-denoiser target
        derived; lossless for Gaussians (proved); round accumulation
    04  scoreboard: the guarantee chain assembled; the guided-teacher
        question resolved; one curvature rules all few-step methods

discrete_diffusion/            tokens, masks, and the attention bridge
    01  the discrete Anderson theorem proved (master-equation level);
        the masking chain solved
    02  the ELBO collapses to weighted masked cross-entropy (MDLM);
        schedule invariance; MLM = one-step denoiser
    03  discrete Tweedie (ratio = posterior) and the Bregman
        projection lemma — the series' trick, proved once
    04  masked diffusion = any-order autoregression (the Beta-integral
        theorem); AR as the degenerate schedule
    05  the parallel-decoding tax = conditional total correlation,
        proved with a worked half-mass-impossible example

statistical_theory/            score error is the whole game
    01  error propagation: the Gronwall W2 bound and the exact
        Gaussian propagator (which noise levels' errors survive)
    02  minimax rates: early stopping = KDE, exactly (proved); the
        classical 1-D rate proved; the learned-score upgrade located
    03  memorization vs generalization: the trichotomy; the linear
        projection theorem; the retrofit to foundations/06
    04  manifold geometry: the sigma^{-2} blowup, the parametrization
        ladder, and the medial-axis spike — one tanh formula
    05  what remains open: the six cross-cutting problems; the
        trilogy, closed
```

All six families of PLAN.md are complete (31 files). The index of
every result proved in full is THEOREMS.md; the per-file contracts
are in PLAN.md.

## Notation

```math
\mathrm{d}x = -\tfrac{\beta_t}{2}\,x\,\mathrm{d}t + \sqrt{\beta_t}\,\mathrm{d}W
\qquad\text{(VP forward)},
```

```math
\alpha_t = e^{-\frac12\int_0^t\beta},\qquad
\sigma_t^2 = 1-\alpha_t^2,\qquad
p_t = \mathrm{Law}(x_t),\qquad
s_t(x) = \nabla\log p_t(x).
```

Data measure `p_0` on `R^d`; prior `gamma = N(0, I)`; hats denote
learned or estimated quantities; `k_t(x|x_0) = N(x; alpha_t x_0,
sigma_t^2 I)` is the forward kernel. Discrete time uses `bar-alpha_k`
products in the DDPM convention, and `04` proves the two conventions
agree.
