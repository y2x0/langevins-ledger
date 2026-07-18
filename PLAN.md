# PLAN: The Proof Program

This document governs all future notebooks in this repository. It is a
contract, not a wishlist: a family enters only with a per-file
specification of the results to be proved, and a file ships only if it
meets the depth contract.

## The Depth Contract

```text
1. at least one theorem proved in full, OR one counterexample worked
   with explicit numbers (distributions, schedules, samplers) — never
   gestured;
2. an explicit statement of where each hypothesis is used in the proof
   (the "load-bearing" audit);
3. its position in the (path P, estimand s, solver S) coordinate
   system of the README;
4. a "what remains open" section naming the honest frontier.
```

Banned: survey files, sampler zoos, files whose math could be
transcribed from a blog post, files that state results without proof
or counterexample. Where a paper's proof is too heavy to reproduce,
the file names exactly which lemma IS proved and which theorem is
assembled around it.

## Dependency Graph

```text
score_foundations (first) ──┬── A. samplers_and_convergence
                            │      (what error the sampler pays)
                            ├── B. flow_matching
                            │      (the path coordinate generalized)
                            ├── C. guidance_and_control   <- A
                            │      (steering; the Bellman bridge)
                            ├── D. distillation           <- A, B
                            │      (collapsing the solver)
                            ├── E. discrete_diffusion
                            │      (the attention-ledger bridge)
                            └── F. statistical_theory     <- A
                                   (learning the score from samples)
```

Recommended order: **A → B → C → D → E → F.**

---

# Phase A — `samplers_and_convergence/` — 6 files

Central question: *the sampler runs a discretized reverse process with
a wrong score; what, provably, does it output?*

```text
01_discretization_of_the_reverse_sde.md
   Euler–Maruyama on the reverse SDE: PROVE the one-step weak/strong
   error bookkeeping for the linear (Gaussian-data) case exactly;
   schedule sensitivity; where the sqrt(d) enters.
02_exponential_integrators.md
   PROVE: DPM-Solver-style updates = exact linear part + polynomial
   extrapolation of the denoiser in log-SNR (extends foundations/04's
   u-rho coordinates); order conditions derived; why they beat EM.
03_the_girsanov_decomposition.md
   PROVE the KL error decomposition (Chen et al. skeleton):
   KL(data || samples) <= (score L2 error) + (discretization) +
   (prior mismatch e^{-T}) via Girsanov + data processing; every term's
   hypothesis audited.
04_polynomial_convergence.md
   Assemble the poly(d, 1/eps) convergence theorem for VP samplers
   under L2-accurate scores, no log-concavity — statements + the
   assembled proof-skeleton with 03's engine; what the theorem does
   NOT say about learned scores (F's subject).
05_ode_vs_sde_error_dynamics.md
   PROVE in the Gaussian case: ODE transports score error without
   decay, SDE contracts it (the stochastic sampler's self-correction);
   the bias-variance reading of solver choice; worked 1-D example
   with numbers.
06_langevin_and_correctors.md
   Underdamped/overdamped Langevin as the inner loop: PROVE the LSI ⇒
   exponential KL decay one-liner chain for the corrector under
   log-concavity; predictor-corrector assembled; the honest gap:
   multimodality (where metastability — attention_flows/05's cousin —
   returns).
```

Sources: Chen–Chewi–Li–Li–Salim–Zhang 2023; Lee–Lu–Tan 2023;
Benton et al. 2024; Lu et al. 2022 (DPM-Solver); Karras et al. 2022.

# Phase B — `flow_matching/` — 5 files

Central question: *the path coordinate freed from OU: which
(path, velocity) pairs are learnable by regression, and what is
actually equivalent to what?*

```text
01_continuity_and_cfm.md
   PROVE the continuity equation for interpolant paths and the
   conditional flow matching identity (the marginal velocity is the
   conditional expectation of conditional velocities — Vincent's
   conditioning trick, second appearance); the regression target is
   exact, not approximate.
02_stochastic_interpolants.md
   The general (alpha_t, sigma_t, coupling) family: PROVE which
   choices give well-defined velocities and scores simultaneously;
   one framework containing DDPM, FM, and bridges.
03_rectified_flow.md
   PROVE: one rectification step preserves marginals and cannot
   increase transport cost; straight-line flows are fixed points;
   the coupling-improvement lemma worked on a 2-point example.
04_the_dictionary.md
   PROVE the affine dictionary on Gaussian paths: velocity = a_t x +
   b_t score (exact coefficients derived); FM training = score
   matching with a different weighting — the estimand coordinate is
   a reweighting choice, proved.
05_what_fm_is_not.md
   The OT fence: PROVE FM with independent coupling does NOT yield
   the OT map (worked counterexample); minibatch-OT couplings and
   what they provably buy; honest scoreboard.
```

Sources: Lipman et al. 2023; Albergo–Boffi–Vanden-Eijnden 2023;
Liu–Gong–Liu 2023; Pooladian et al. 2023.

# Phase C — `guidance_and_control/` — 5 files

Central question: *steering a diffusion is conditioning or control —
which approximations sample what, exactly?* The Bellman bridge.

```text
01_doob_h_transforms.md
   PROVE: exact conditional sampling = the h-transform of the reverse
   process, h = the conditional likelihood under the forward noise;
   classifier guidance as its plug-in approximation, the
   approximation's exact error term identified.
02_cfg_deep_dive.md
   Continue foundations/05: the omega-family's endpoints, interval
   guidance, rescaled CFG — each as a modified target, derived; when
   sharpening is provably mode-collapse.
03_posterior_sampling_inverse_problems.md
   DPS and friends: PROVE the Gaussian-likelihood case exactly
   (linear inverse problems: the posterior score has closed form —
   the Wiener filter of foundations/06 with measurements); the
   Jensen-type gap of the general DPS approximation, exhibited.
04_reward_finetuning_is_kl_control.md
   The bridge file: PROVE that KL-regularized reward fine-tuning of a
   diffusion is the stochastic control problem whose optimal
   controlled process is the exponentially tilted path measure — the
   value function IS log h; Bellman's rlhf_mathematics/02, run
   through path space; RLHF-for-diffusion (DPO variants) statements.
05_failure_modes.md
   Guidance pathologies with mechanisms: over-saturation (05's Jensen
   gap compounding), leaving the data manifold (F/04's blowup),
   reward hacking (Bellman's Goodhart file, third repo appearance).
```

Sources: Doob; Chung et al. 2023 (DPS); Uehara et al. 2024;
Domingo-Enrich et al. 2024 (adjoint matching); Wallace et al. 2023.

# Phase D — `distillation/` — 4 files

Central question: *collapsing the solver into one step: what survives,
with what guarantee?*

```text
01_the_consistency_condition.md
   PROVE: self-consistency along the PF-ODE + boundary condition
   characterizes the exact solution map (uniqueness via Gronwall);
   the parametrization that enforces the boundary for free.
02_consistency_training_vs_distillation.md
   PROVE the CD estimator is exact at the teacher's solution; the CT
   estimator's bias term derived (the one-step ODE error inside the
   target); why CT needs schedules.
03_progressive_distillation.md
   PROVE the halving step exact for linear (Gaussian) dynamics; the
   error recursion for L halvings; DDIM's role (foundations/04).
04_scoreboard.md
   One-step samplers: what is proved (consistency: exact-teacher
   guarantees), what is empirical (adversarial post-training), and
   the open theory of distilling GUIDED processes.
```

Sources: Song–Dhariwal–Chen–Sutskever 2023; Salimans–Ho 2022;
Kim et al. 2024.

# Phase E — `discrete_diffusion/` — 5 files

Central question: *diffusion on token spaces — and the exact bridge to
the attention ledger.*

```text
01_ctmc_forward_and_reversal.md
   Continuous-time Markov chains: PROVE the reverse-rate formula
   (the discrete Anderson theorem) from detailed
   time-marginal bookkeeping; uniform vs absorbing forward chains.
02_masked_diffusion_elbo.md
   PROVE: the masked/absorbing ELBO reduces to a weighted sum of
   masked-token cross-entropies (the MDLM simplification) — masked
   language modeling IS one-step discrete denoising, as a theorem.
03_score_entropy.md
   Ratio matching / score entropy: PROVE the discrete analogue of
   Vincent's theorem; what replaces Tweedie on finite spaces.
04_any_order_autoregression.md
   PROVE: absorbing diffusion's ELBO = expected any-order
   autoregressive likelihood; AR as a degenerate (deterministic-order)
   schedule; exact factorization, both directions.
05_scoreboard_vs_ar.md
   Parallel decoding vs the serial ceiling: attention-ledger's
   E/03–07 applied to denoisers; what discrete diffusion provably
   buys (bidirectional conditioning) and pays (token independence
   within a step — the worked failure).
```

Sources: Austin et al. 2021; Campbell et al. 2022; Lou–Meng–Ermon
2024; Sahoo et al. 2024; Shi et al. 2024.

# Phase F — `statistical_theory/` — 5 files

Central question: *the score is learned from n samples; what does the
sampled distribution provably inherit?*

```text
01_score_error_propagation.md
   PROVE Gronwall-type stability: W2/TV between exact and
   approximate-score samplers bounded by the L2 score error along the
   path (the Gaussian case exact; A/03's engine reused).
02_minimax_rates.md
   Diffusion-based distribution estimation attains minimax rates over
   smoothness classes (Oko et al.-line): statements + the provable
   kernel-smoothing reduction in one dimension.
03_memorization_vs_generalization.md
   Continue foundations/06: PROVE that the empirical-optimal score is
   attention over the training set and the exact sampler memorizes;
   smoothing/early-stopping/architecture error as the ONLY sources of
   generalization — the inductive-bias ledger.
04_manifold_geometry.md
   PROVE the sigma^{-2} score blowup normal to a supported manifold
   (the two-point case exactly); schedule/parametrization choices as
   blowup management; guidance's manifold-leaving read (C/05).
05_what_remains_open.md
   The honest frontier file: why learned-score generalization has no
   theory matching practice; the repo's cross-cutting open list.
```

Sources: Oko–Akiyama–Suzuki 2023; De Bortoli 2022; Pidstrigach 2022;
Kadkhodaie et al. 2024.

---

# Scope Summary

```text
family                      files   the one-line payoff
score_foundations             6     the objects, exactly; CFG's Jensen
                                    gap; the score IS attention
A samplers_and_convergence    6     what the sampler provably outputs
B flow_matching               5     the path coordinate, unified
C guidance_and_control        5     steering = control; the Bellman
                                    bridge (tilting, fourth repo file)
D distillation                4     one-step maps with real guarantees
E discrete_diffusion          5     the attention bridge; MLM = denoiser
F statistical_theory          5     score error is the whole game
total                        36
```

## Retrofit List

```text
score_foundations/05   CFG deep-dive results          <- C/02
score_foundations/06   memorization completed          <- F/03
foundations cross-refs attention-ledger (score = attention over data;
                       parallel-decoding ceiling) and bellmans-ledger
                       (tilting; Goodhart) as phases land
```
