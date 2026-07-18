# The Girsanov Decomposition

## The Question

The sampler differs from the exact reverse process in three ways: it
starts from the prior instead of `p_T`, it uses a learned score, and
it discretizes time. The Girsanov argument — the engine of the modern
convergence theory (Chen–Chewi et al., "sampling is as easy as
learning the score") — turns all three into ADDITIVE terms of one KL
bound. This file proves the decomposition for the continuous-time
sampler in full, shows where the discretization term attaches, and
audits the hypotheses; `04` assembles the headline iteration
complexities on top of it.

## The Theorem

Setup: true reverse process (from `score_foundations/03`) with drift
`b_t(x) = -f + g^2 s_t(x)`, initialized at `p_T`; the algorithm runs
the same equation with `\hat s` in place of `s`, initialized at the
prior `\gamma`. Let `q_0` be the algorithm's output law at time
`t_{\min} \approx 0`.

**Theorem (continuous-time decomposition).** Under Novikov's
condition on the drift difference (declared, not free — see the
audit),

```math
\mathrm{KL}\big(p_{t_{\min}}\ \big\|\ q_0\big)
\;\le\;
\underbrace{\mathrm{KL}\big(p_T\,\|\,\gamma\big)}_{\text{prior mismatch}}
\;+\;
\underbrace{\frac{1}{2}\int_{t_{\min}}^{T} g_t^2\;
\mathbb{E}_{x\sim p_t}\big\|s_t(x) - \hat s_t(x)\big\|^2\,\mathrm{d}t}_{\text{score error}} .
```

*Proof.* Three steps, each one idea.

*(i) Lift to path space.* Let `P` and `Q` be the laws of the two
processes on path space `C([0, T-t_{\min}]; R^d)` — same diffusion
coefficient `g`, drifts differing by `\Delta_t(x) = g_t^2(s_t -
\hat s_t)(x)`, initial laws `p_T` and `\gamma`.

*(ii) Chain rule + Girsanov.* The KL between the path laws splits as
initial-condition KL plus the conditional path KL given the start;
Girsanov's theorem evaluates the latter: for SDEs with common `g`,

```math
\mathrm{KL}(P\,\|\,Q)
= \mathrm{KL}(p_T\,\|\,\gamma)
+ \mathbb{E}_P\Big[\frac12\int \frac{\|\Delta_t(x_t)\|^2}{g_t^2}\,\mathrm{d}t\Big]
= \mathrm{KL}(p_T\|\gamma) + \frac12\int g_t^2\,\mathbb{E}_{p_t}\|s_t-\hat s_t\|^2\,\mathrm{d}t,
```

using `\|\Delta\|^2/g^2 = g^2\|s - \hat s\|^2` and that under `P` the
time-`t` marginal is exactly `p_t` (the true reverse process retraces
the forward marginals — `score_foundations/03`; this is why the score
error is measured under the DATA path, the detail that makes the
bound usable: the training loss of `score_foundations/02` estimates
precisely this integrand).

*(iii) Data processing.* The output is the path's endpoint — a
measurable function of the path — and KL contracts under measurable
maps: `KL(p_{t_{\min}}\|q_0) \le KL(P\|Q)`. ∎

**Where discretization attaches.** The implemented algorithm's drift
is `\hat b(x_{t_k}, t_k)` — frozen at grid points (`01`). Run the
same proof with `Q` the law of the INTERPOLATED discrete process
(the standard continuous-time embedding of EM): the drift difference
gains a second piece, and `\|a + c\|^2 \le 2\|a\|^2 + 2\|c\|^2`
splits the integral:

```math
\mathrm{KL}\big(p_{t_{\min}}\|q_0\big)
\;\le\;
\mathrm{KL}(p_T\|\gamma)
\;+\; \int g^2\,\mathbb{E}_{p_t}\|s - \hat s\|^2
\;+\; \underbrace{\int g^2\,\mathbb{E}\big\|\hat s_t(x_t) - \hat s_{t_k}(x_{t_k})\big\|^2}_{\text{discretization}},
```

the last term bounded, under Lipschitz-in-space and regular-in-time
score, by `O(\sum_k h_k^2\,(\cdot))`-type sums — the term whose
careful estimation (without Lipschitzness, via the Gaussian kernel's
own smoothing) is the actual content of the modern papers, cited in
`04`. The decomposition itself, proved above, is where the subject's
CONCEPTUAL work ends and its technical work begins.

## Readings

**1. The three knobs are independent.** Prior mismatch decays as
`e^{-\int\beta/2}` in the horizon (`score_foundations/01`'s
contraction, in KL form for Gaussians); score error is the training
loss on the true path; discretization is `01`–`02`'s subject. The
theorem LICENSES the division of labor this family is organized by.

**2. "Sampling is as easy as learning the score," literally.** The
bound's score term is the `L^2(p_t)` error — the quantity Vincent's
theorem says training minimizes. No log-concavity, no isoperimetry,
no mixing assumptions on the DATA appear anywhere in the proof: the
hard part of sampling (which for MCMC is the data's own geometry) has
been shifted into the forward process's contraction, which is
unconditional. The price appears in the audit below.

**3. TV and W2 versions.** Pinsker converts the bound to total
variation; Wasserstein statements need different couplings
(`statistical_theory/01`); the KL is the native metric of the
Girsanov calculus.

## Load-Bearing Audit

```text
same diffusion coefficient   Girsanov compares drifts only; solvers
                             that alter g (lambda-family members)
                             need the matching version per lambda;
Novikov's condition          integrability of the exponential
                             martingale — implied by the score-error
                             integral being finite in practice, but a
                             genuine hypothesis: scores that blow up
                             at t_min faster than L2 void the proof
                             (manifold data again — the t_min > 0
                             cutoff is load-bearing HERE, not just in
                             foundations);
marginals under P are p_t    step (ii)'s evaluation of the error
                             under the data path — inherited from the
                             exactness of the reversal; it is also
                             why the theorem says nothing about error
                             under the SAMPLER's path (the
                             distribution-shift gap: the algorithm
                             visits its own states; see 05's
                             contraction for why this is survivable);
data processing              the output-marginal step; also silently
                             the reason path-level guarantees are
                             stronger than needed.
```

## Position In The Coordinate System

The master error ledger for `S`: one inequality that accepts any
path `P` (through the prior term), any estimand quality (the score
term), and any solver (the discretization term), and adds them. The
rest of the family instantiates: `04` the rates, `05` the
solver-choice term's dynamics, `06` the corrector's improvement.

## What Remains Open

Tightness: the decomposition is an upper bound whose triangle-type
splits are lossy, and matching lower bounds (which sampler errors are
NECESSARY given a score-error budget) exist only in restricted
settings; the shift question flagged in the audit — errors measured
under `p_t` versus incurred under `q_t` — is handled inside current
proofs by Girsanov's symmetry but remains the obstruction to
extending the argument to samplers that leave the data path far
behind (heavily guided sampling, `guidance_and_control/05`); and the
ODE endpoint: Girsanov is a stochastic-calculus tool, the PF-ODE has
no noise to reweight, and ODE-sampler analyses need different
machinery — which is exactly `05`'s subject.
