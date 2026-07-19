# PLAN_G: The Second Proof Program (draft)

**STATUS: draft, unscheduled.** PLAN.md's six phases are complete;
this document is the contract for what a second pass would add. It
covers (i) whole subjects the first program never touched
(Schrödinger bridges, stochastic localization, latent diffusion,
parallel sampling, learning dynamics, evaluation theory) and (ii)
second passes where the first program's own "what remains open"
sections now have a literature to audit. Nothing here is started;
a phase enters the repository only when its per-file contracts below
survive the depth contract.

## The Depth Contract (inherited, verbatim)

```text
1. at least one theorem proved in full, OR one counterexample worked
   with explicit numbers — never gestured;
2. an explicit statement of where each hypothesis is used in the
   proof (the "load-bearing" audit);
3. its position in the (path P, estimand s, solver S) coordinate
   system of the README;
4. a "what remains open" section naming the honest frontier.
```

Banned, as before: survey files, sampler zoos, anything
transcribable from a blog post. Two of the phases below (K and L)
sit closer to the empirical frontier than anything in PLAN.md; the
contract binds them hardest — each file must isolate the provable
kernel of a literature that is mostly measurement.

## Dependency Graph

```text
PLAN.md (complete) ──┬── G. schrodinger_bridges
                     │      (the path coordinate, coupled at both ends)
                     ├── H. stochastic_localization   <- A
                     │      (the samplers phases, re-derived)
                     ├── I. latent_diffusion           <- F
                     │      (the composition nobody prices)
                     ├── J. parallel_sampling          <- A
                     │      (the solver coordinate, in parallel time)
                     ├── K. learning_dynamics          <- F
                     │      (problem 1 of statistical_theory/05)
                     └── L. evaluation_theory          <- D
                            (problem 4 of statistical_theory/05)

second passes: M. distillation (DMD theory), N. discrete rates —
each retrofits an existing family rather than opening a folder.
```

Recommended order: **G → H → J → I → M → N → K → L.** The first
four are theorem-rich (safest under the contract); K and L are the
consequential ones and go last, when the retrofit machinery exists.

---

# Phase G — `schrodinger_bridges/` — 4 files

Central question: *flow matching fixed one endpoint's coupling by
fiat (independent); the Schrödinger bridge is the path P whose
coupling is OPTIMAL under an entropy budget — what is provable
about computing it?* Closes the fence `flow_matching/05` left:
straight ≠ OT, but entropic OT is attainable.

```text
01_the_schrodinger_system.md
   The static problem: PROVE existence/uniqueness of the entropic
   OT plan via the Schrödinger system (Fortet/Sinkhorn potentials);
   the Gaussian case in closed form (the entropic Bures formula);
   epsilon -> 0 recovers OT, epsilon -> inf recovers independence —
   both limits derived, placing flow_matching/05's fence as the
   epsilon = inf endpoint of one family.
02_the_dynamic_problem.md
   PROVE: the dynamic SB = the h-transform PAIR of the reference
   process (forward and backward potentials — guidance/01's
   machinery, two-sided); KL(path || reference) decomposes as
   static KL + bridge mismatch; SB with OU reference at horizon T
   -> the diffusion path as T -> inf (the ledger's VP path as a
   degenerate bridge, derived).
03_ipf_and_imf.md
   The two projection algorithms: PROVE IPF = alternating
   h-transforms (marginal fixing), IMF = alternating
   Markovian/reciprocal projections; the Gaussian case: exponential
   convergence with explicit contraction factor (the 2024-25
   results' solvable core, reproduced in full); where log-concavity
   is load-bearing in the general rates (statements, audited).
04_bridge_matching_and_the_fence.md
   DSBM as CFM with a learned coupling: PROVE the bridge-matching
   regression target is exact given the current coupling (the
   Bregman lemma, again); the error a WRONG coupling induces on the
   marginals, worked on the two-point example; the honest
   scoreboard: what trained bridges provably inherit from 03's
   idealized convergence (currently: nothing — the gap named).
```

Sources: Léonard 2014 (survey of the Schrödinger problem);
De Bortoli et al. 2021 (DSB/IPF); Shi et al. 2023 (DSBM/IMF);
Peluchetti 2023; Gushchin et al. 2024, arXiv:2410.02601 (IPMF);
Gentiloni Silveri–Conforti–Durmus 2025, arXiv:2510.20871 (IMF
exponential convergence, NeurIPS 2025); Sokolov–Korotin 2025,
arXiv:2508.02770 (finite-state IMF contraction).

# Phase H — `stochastic_localization/` — 3 files

Central question: *the same reverse process, discovered from the
other end: Eldan's measure-valued martingale IS the diffusion
sampler under a time change — what does the second viewpoint prove
that the first could not?*

```text
01_the_equivalence_theorem.md
   PROVE the El Alaoui–Montanari/Montanari time-change dictionary:
   the SL observation process U_s = s*x + W_s and the VP diffusion
   x_t have the same law under t(s) = (1/2)log(1 + 1/s), drift =
   posterior mean = Tweedie (foundations/02, rediscovered); the
   localization limit mu_s -> delta_xi IS sampling — proved in the
   Gaussian case with every constant.
02_the_covariance_decay_lemma.md
   PROVE Eldan's covariance/variance decay identity (the SL
   analogue of de Bruijn): d/ds of the posterior covariance =
   -Cov^2; the measure-decomposition reading (entropy vs
   covariance) and its information-theoretic proof (the pinning
   lemma route) — the repository's estimation-theory lens applied
   to the sampler itself.
03_what_localization_buys.md
   Assemble the nearly d-linear KL convergence bound (Benton et
   al.) with SL as the engine: which steps of samplers/03-04's
   Girsanov argument the SL viewpoint replaces, and where the
   d-dependence improves; the honest audit — what is genuinely new
   versus renamed (the file's thesis: the viewpoints are one
   theorem apart, and the theorem is 01's).
```

Sources: Eldan 2013; El Alaoui–Montanari 2022 (IEEE TIT,
information-theoretic view); Montanari 2023, arXiv:2305.10690
(sampling/diffusions/SL dictionary); Benton–De Bortoli–Doucet–
Deligiannidis 2024, arXiv:2308.03686 (nearly d-linear bounds);
arXiv:2510.04460 (Perspectives on Stochastic Localization, 2025).

# Phase I — `latent_diffusion/` — 3 files

Central question: *practice runs the diffusion in a learned latent
space; the guarantee chain (distillation/04) starts at "data" — who
pays for the encoder?* The composition nobody in the first program
priced.

```text
01_the_composition_decomposition.md
   PROVE the pushforward triangle inequality: W2(data, D#(samples))
   <= reconstruction error + Lip(D) x latent sampling error — each
   term identified, the decoder Lipschitz constant load-bearing and
   generically unbounded (the fence); the linear (PCA/linear-VAE)
   case exact: latent diffusion = diffusion on principal
   coordinates, ambient error = latent error x singular values,
   proved in full.
02_geometry_of_the_latent_path.md
   What the encoder does to the (P, s, S) coordinates: PROVE that
   for a linear isometric encoder the latent score is the ambient
   score restricted (and for non-isometric E it is NOT — the
   Jacobian correction derived, worked 2-D example); near-isometry
   as the condition under which phase-A/F guarantees transfer at
   all; the manifold reading against statistical_theory/04 (the
   encoder as blowup management — the ladder's missing rung).
03_end_to_end_rates_read.md
   The latent-SB end-to-end W2 rate (2024) and the
   information-theoretic VAE+DM generalization bounds (ICLR 2025):
   statements, hypothesis audit (what "pre-trained encoder" quietly
   assumes about the data-encoder coupling), and the honest gap —
   the encoder is trained on reconstruction + KL, the diffusion on
   score matching, and no theorem couples the two objectives.
```

Sources: Rombach et al. 2022 (LDM, for the object); arXiv:2404.13309
(latent Schrödinger bridge, end-to-end W2 rates); arXiv:2506.00849
(unified VAE/DM information-theoretic generalization, ICLR 2025);
Surendran et al. 2025, arXiv:2410.16750 (VAE optimization
convergence); Chen et al. 2023 (score approximation on manifolds,
for 02's transfer question).

# Phase J — `parallel_sampling/` — 2 files

Central question: *the solver coordinate has a second axis the
first program ignored: wall-clock depth. Sequential steps can be
traded for parallel Picard iterations — with proofs.*

```text
01_picard_iteration.md
   PROVE: the PF-ODE solution is the fixed point of the Picard map;
   contraction in sup-norm over a time block with rate set by the
   drift's Lipschitz constant (Banach, in full); the Gaussian case
   exact — iteration count as a function of blocking, versus the
   sequential step count of A/01-02; where the drift Lipschitz
   constant inherits the score blowup (statistical_theory/04) and
   kills the contraction near t = 0 — the honest boundary, worked.
02_parallel_rounds_with_guarantees.md
   Assemble the polylog parallel-rounds theorems (randomized
   midpoints; parallel simulation): TV-accuracy in O(polylog(d/eps))
   rounds under L2-accurate scores — statements, the
   Picard-plus-discretization error recursion audited (the
   a, b < 1 double-contraction bookkeeping reproduced for the
   linear case); the scoreboard against A/04: what parallelism
   provably costs in total work versus buys in depth.
```

Sources: Shih et al. 2023 (ParaDiGMS, NeurIPS 2023); Gupta et al.
2024, arXiv:2406.00924 (randomized midpoints, sequential and
parallel); arXiv:2412.07435 (parallel simulation for log-concave
and SGM, 2024); Anari et al. / parallel Langevin line (for 02's
lineage).

# Phase K — `learning_dynamics/` — 4 files

Central question: *problem 1 of `statistical_theory/05`, attacked:
optimization and architecture were invisible to every theorem in
the first program; the trained score is what GD makes it.* The
consequential phase; the contract binds hardest here.

```text
01_gd_learns_the_score.md
   The NTK/kernel regime: PROVE that GD on denoising score matching
   with a linearized network solves a sequence of kernel ridge
   regressions with noisy labels; early stopping as the
   regularization dial (the F/02 bandwidth channel, now produced by
   the ALGORITHM rather than assumed); the linear-network case in
   full — GD's trajectory converges to statistical_theory/03's
   projected empirical score, making the projection theorem
   algorithmic.
02_the_memorization_transition.md
   The target's variance IS the regularizer: PROVE in the
   over-parameterized linear/random-feature case that the gap
   between the empirical-score minimizer and the GD-at-finite-time
   iterate scales with the DSM label noise, and identify the
   n-vs-capacity crossover in the solvable model (the n_mem of
   statistical_theory/05, computed where it can be); large
   learning-rate implicit regularization: statements, audited.
03_architecture_as_smoother.md
   The inductive-bias file: PROVE the equivariant projection
   theorem — for a convolutional (translation-equivariant, local)
   score class, the learned score is the projection of the
   empirical score onto equivariant fields, and the sampler's law
   gains the corresponding symmetry (Kamb–Ganguli's mechanism, as
   a theorem in the linear case); GAHB (harmonic shrinkage) as the
   diagonalization of that projection: statements + the C-alpha
   near-optimality read; DiT locality: statements.
04_the_composed_guarantee.md
   Assemble the first end-to-end chain with the OPTIMIZER inside:
   GD score error (01) + sampling (A/03-04) -> distribution error,
   per the 2024-25 line (score estimation without ERM access);
   every hypothesis audited against a real training run's
   violations — the file that says exactly how far "provably
   trained diffusion" currently reaches, and where the exponential
   fine print lives.
```

Sources: Han et al. 2024, arXiv:2401.15604 (GD-trained score
networks: optimization + generalization); arXiv:2505.18344 (sample
complexity without ERM access, 2025); Kadkhodaie–Guth–Simoncelli–
Mallat 2024, arXiv:2310.02557 (GAHR, ICLR 2024); Kamb–Ganguli 2025
(locality/equivariance and creativity); NeurIPS 2025 DiT
inductive-bias line; arXiv:2605.06077 (rethinking generalization,
2026 — the three-family map of explanations); Bonnaire et al. 2025
(dynamical regularization); Cui et al. 2024.

# Phase L — `evaluation_theory/` — 2 files

Central question: *problem 4 of `statistical_theory/05`: the
theory's currencies and the benchmarks' currency. Most of this
literature is empirical; these two files extract everything that is
actually a theorem.*

```text
01_fid_as_mathematics.md
   PROVE the Gaussian Frechet/W2 formula (the trace identity FID
   rests on); PROVE the plug-in estimator's O(1/N) bias with a
   generator-dependent constant (the Chong–Forsyth term derived in
   the Gaussian case — why fixed-N comparisons are invalid, as
   mathematics not complaint); what FID actually is: W2 between
   Gaussian PROJECTIONS of feature pushforwards — three deletions
   (feature map, Gaussianization, finite N), each priced where
   possible, each a hypothesis the number silently asserts.
02_the_metric_gap_fenced.md
   The honest file: PROVE the data-processing fence — every
   feature-space IPM/W2 is dominated by the input-space divergence
   (so the chain's KL bounds DO bound feature-space FID/MMD, with
   the feature map's Lipschitz constant), and the converse fails by
   counterexample (two laws, KL far apart, identical features —
   worked); MMD/CMMD as an actual metric with unbiased estimators
   (the U-statistic, proved); what would count as closing problem
   4, stated as a target theorem.
```

Sources: Heusel et al. 2017 (FID); Chong–Forsyth 2020,
arXiv:1911.07023 (bias, FID-infinity); Jayasumana et al. 2024
(Rethinking FID / CMMD, CVPR 2024); Gretton et al. 2012 (MMD
U-statistics); the Inception-bias critique line (OpenReview
mLG96UpmbYz).

# Phase M — retrofit: `distillation/` +1 file

```text
05_distribution_matching.md
   The route distillation/04 called "least theorized," now
   contracted: PROVE the DMD gradient identity — the reverse-KL
   gradient between diffused generator and data laws equals the
   difference of two scores (real minus fake), integrated over
   noise levels (Tweedie + the chain rule, in full); reverse-KL
   mode-seeking exhibited on the two-mode Gaussian (worked
   numbers); the fixed-point audit: zero DMD gradient at matched
   laws, but the fake-score inner loop is load-bearing and its
   staleness is DMD2's two-timescale problem (statements); the
   f-divergence generalization as a one-line change of the same
   identity; and the one-step lower bound (03's open item) restated
   against this route — distribution matching evades the TEACHER
   floor, not the capacity floor.
```

Sources: Yin et al. 2024, arXiv:2311.18828 (DMD); arXiv:2405.14867
(DMD2); Wang et al. 2023 (VSD/ProlificDreamer, the identity's
origin); f-distill 2025 (general f-divergences); Luo et al. 2024
(Diff-Instruct line).

# Phase N — retrofit: `discrete_diffusion/` +1 file

```text
06_convergence_rates.md
   Phase E proved the objects; the 2024-25 literature now has the
   rates, and statistical_theory/08's open items have targets:
   PROVE the masking-chain analogue of the Girsanov decomposition
   (KL <= score error + discretization + mixing, the discrete
   triple, assembled at the master-equation level with E/01's
   reverse rates); the absorbing-vs-uniform comparison as theorems
   (linear-in-d for absorbing; where the singleton stationary law
   breaks naive KL and the surrogate-initialization fix); the
   tau-leaping and uniformization samplers' error terms audited;
   the first-hitting/MATU observation — each token unmasks at most
   once — as the structural reason masked chains beat uniform ones
   (the E/04 any-order theorem, now paying rent in rates).
```

Sources: Chen–Ying 2024, arXiv:2402.08095 (uniformization);
arXiv:2410.02321 (discrete-time analysis, 2024); arXiv:2506.02318
(Absorb and Converge, 2025); OpenReview ZXZoV3OCE7 (masked
complexity, MATU, 2025); arXiv:2512.00580 (masked and random-walk
dynamics, sharp non-asymptotic rates, 2025); Campbell et al. 2022
(tau-leaping, for lineage).

---

# Scope Summary

```text
phase                        files   the one-line payoff
G schrodinger_bridges          4     the coupling learned, not fixed;
                                     IPF/IMF convergence proved where
                                     provable
H stochastic_localization      3     the sampler re-derived; d-linear
                                     bounds' engine exposed
I latent_diffusion             3     the encoder finally priced
J parallel_sampling            2     depth vs work, with proofs
K learning_dynamics            4     problem 1 attacked: GD, n_mem,
                                     architecture as projection
L evaluation_theory            2     problem 4 fenced: FID as
                                     mathematics, the gap as a target
M distillation retrofit        1     DMD's gradient identity proved
N discrete retrofit            1     phase E's rates, assembled
total                         20
```

## Reading List (the papers behind the contracts, by phase)

```text
G   Léonard 2014; De Bortoli et al. 2021; Shi et al. 2023 (DSBM);
    arXiv:2410.02601 (IPMF); arXiv:2510.20871, arXiv:2508.02770
    (IMF exponential convergence, 2025)
H   Eldan 2013; El Alaoui–Montanari 2022; Montanari
    arXiv:2305.10690; Benton et al. arXiv:2308.03686;
    arXiv:2510.04460 (perspectives survey, 2025)
I   arXiv:2404.13309 (latent SB rates); arXiv:2506.00849 (VAE+DM
    generalization, ICLR 2025); arXiv:2410.16750 (VAE optimization)
J   Shih et al. 2023 (ParaDiGMS); arXiv:2406.00924 (randomized
    midpoints); arXiv:2412.07435 (parallel simulation)
K   arXiv:2401.15604 (GD score estimation); arXiv:2505.18344
    (no-ERM sample complexity); arXiv:2310.02557 (GAHR);
    Kamb–Ganguli 2025; arXiv:2605.06077 (generalization map, 2026)
L   Heusel et al. 2017; arXiv:1911.07023 (FID bias); Jayasumana
    et al. 2024 (CMMD); Gretton et al. 2012 (MMD)
M   arXiv:2311.18828 (DMD); arXiv:2405.14867 (DMD2); VSD 2023;
    f-distill 2025
N   arXiv:2402.08095; arXiv:2410.02321; arXiv:2506.02318;
    ZXZoV3OCE7 (MATU); arXiv:2512.00580
```

## What This Program Deliberately Leaves Out

Consistency with the ban list: no file on "applications of
diffusion to X"; no benchmark meta-analysis beyond L's two
theorem-bearing files; no architecture zoo — K/03 admits exactly
the symmetry classes with a projection theorem. The two problems
of statistical_theory/05 that this program still does NOT close:
the deep (nonlinear, trained-at-scale) generalization channel — K
reaches its linear and kernel shadows only — and the perceptual
side of the metric gap, where L can fence but not cross. Those
stay on the open list, which is where honest frontiers belong.
