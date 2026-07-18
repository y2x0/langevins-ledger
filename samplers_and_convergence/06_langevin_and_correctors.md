# Langevin And The Correctors

## The Question

The predictor–corrector pattern interleaves reverse-diffusion steps
(the predictor: move down the noise scale) with Langevin steps at a
FIXED noise level (the corrector: equilibrate toward `p_t` before
moving on). The corrector is the one component of the sampler that is
classical MCMC, and its guarantee is the one-line chain
de Bruijn + log-Sobolev ⇒ exponential KL decay. This file proves the
chain, computes the Gaussian constant, assembles the
predictor–corrector logic, and closes on the honest boundary —
multimodal targets, where Langevin's guarantee collapses and this
series meets its third metastability theorem.

## The Langevin Diffusion And de Bruijn's Identity, Proved

Target a fixed smooth positive density `p` (for the corrector:
`p = p_t` at the current noise level, whose score the network already
provides). The overdamped Langevin diffusion:

```math
\mathrm{d}x \;=\; \nabla\log p(x)\,\mathrm{d}t + \sqrt{2}\,\mathrm{d}W_t .
```

**Lemma (stationarity).** `p` is invariant. *Proof.* Fokker–Planck:
`\partial_t q = -\nabla\cdot(q\nabla\log p) + \Delta q`; at `q = p`:
`-\nabla\cdot(p\nabla\log p) + \Delta p = -\nabla\cdot(\nabla p) +
\Delta p = 0` — the score identity `p\nabla\log p = \nabla p` doing
its fourth job in this repository. ∎

**Theorem (de Bruijn).** Along the flow,

```math
\frac{\mathrm{d}}{\mathrm{d}t}\,\mathrm{KL}\big(q_t\,\|\,p\big)
\;=\;
-\ I\big(q_t\,\|\,p\big)
\;:=\; -\int q_t\,\Big\|\nabla\log\frac{q_t}{p}\Big\|^2 ,
```

the relative Fisher information: KL decays, and its decay RATE is the
Fisher information.

*Proof.* Rewrite Fokker–Planck in divergence form:
`\partial_t q = \nabla\cdot\big(q\,\nabla\log(q/p)\big)` (expand:
`\nabla\cdot(q\nabla\log q) = \Delta q` and the `p`-term matches the
drift). Then, using mass conservation (`\int\partial_t q = 0`),

```math
\frac{\mathrm{d}}{\mathrm{d}t}\int q_t\log\frac{q_t}{p}
= \int \partial_t q_t\,\log\frac{q_t}{p}
= -\int q_t\,\Big\|\nabla\log\frac{q_t}{p}\Big\|^2
```

after one integration by parts (boundary terms discarded under the
standing decay assumptions). ∎

**Corollary (LSI ⇒ exponential decay).** Say `p` satisfies a
log-Sobolev inequality with constant `\mu` if
`\mathrm{KL}(q\|p) \le \tfrac{1}{2\mu} I(q\|p)` for all `q`. Then

```math
\mathrm{KL}(q_t\|p) \;\le\; e^{-2\mu t}\ \mathrm{KL}(q_0\|p).
```

*Proof.* `\frac{d}{dt}\mathrm{KL} = -I \le -2\mu\,\mathrm{KL}`;
Grönwall. ∎

**The Gaussian constant.** `N(m, \sigma^2 I)` satisfies LSI with
`\mu = 1/\sigma^2` (Gross; statement — the classical computation),
and Bakry–Émery upgrades it: any `\mu`-strongly-log-concave `p` has
LSI(`\mu`) (statement). For the corrector at noise level `t` this is
immediately useful: `p_t` is the data convolved with
`N(0, \sigma_t^2)`, and at HIGH noise the convolution dominates —
`p_t` inherits near-Gaussian log-concavity and the corrector mixes at
rate `\approx 1/\sigma_t^2`-to-`1` scales. At low noise it inherits
the data's geometry, and the guarantee dies exactly where the honest
section below begins.

## The Predictor–Corrector Logic, Assembled

Why interleave at all, given `03`–`04` already guarantee the
predictor? Because the two components fix DIFFERENT errors:

```text
predictor    moves the noise level; accumulates discretization and
             score-transport error (01, 05) — its output at level t is
             near p_t but off by the accumulated budget;
corrector    holds the level; by the corollary, contracts whatever
             error is there toward p_t at rate e^{-2 mu_t s} — using
             ONLY the same learned score (Langevin's drift IS
             s_hat(x, t)).
```

The corrector is `05`'s contraction mechanism made explicit and run
deliberately: `05` showed the `\lambda = 1` sampler re-contracts its
errors in passing; the corrector spends dedicated compute on that
contraction wherever it is cheap (high noise, large `\mu_t`). The
composite guarantee is the obvious composition: predictor error in,
`e^{-2\mu_t s}`-contracted error out, plus the corrector's own
score-error floor — with `\hat s` in the drift, Langevin's stationary
point is not `p_t` but a perturbation at distance
`O(\|s - \hat s\|_{L^2}/\mu_t)`-scale (the same bias-floor structure
as `05`'s saturation at `2b`; one-line linear computation in the
Gaussian case). Nothing is free: the corrector converts iteration
budget into error reduction at exchange rate `\mu_t`.

## The Honest Boundary: Multimodality

For multimodal `p_t` (low noise, separated data modes), the LSI
constant is exponentially small: for a symmetric double well with
barrier height `\Delta` at temperature 1, the mixing time is
`e^{\Theta(\Delta)}` (Eyring–Kramers; statement) — the corrector will
NOT move mass between modes in any feasible budget. This is the
third appearance of metastability in this series
(`attention_flows/05`: token clusters, lifetime `e^{3\beta/2}`;
Bellman's ledger: exploration's exponential lower bounds;
here: Langevin's barrier crossing), and the diffusion-specific
resolution deserves its plain statement: **the sampler does not need
the corrector to cross barriers, because the PREDICTOR carries mass
between modes at high noise, where the barriers do not yet exist.**
Mode weights are decided early (when `p_t` is nearly unimodal and
mixing is fast) and refined locally late. That is the mechanism by
which diffusion sampling evades the data's isoperimetry — `04`'s
"the forward process did the mixing in advance," now visible as a
division of labor across noise levels: global structure at high `t`,
local structure at low `t`, and the corrector useful precisely
within basins.

## Load-Bearing Audit

```text
p ∇log p = ∇p            stationarity — the score identity again;
integration by parts      de Bruijn; decay at infinity assumed;
LSI                       the exponential rate; log-concavity is
                          SUFFICIENT (Bakry–Émery) and at low noise
                          FALSE — the boundary section is not a
                          technicality, it is the regime map;
score error in the drift  the corrector's floor: Langevin with s-hat
                          equilibrates to the wrong target at
                          distance ~ error/mu — the corrector cleans
                          solver error, never score error;
discretization            ULA (Euler of Langevin) adds its own O(h)
                          bias with known constants (classical,
                          cited); the corrector inherits 01's story
                          at fixed t.
```

## Position In The Coordinate System

The last component of `S`: a within-level equilibrator whose
guarantee is classical, whose useful regime the noise scale
stratifies, and whose limits close the family's arc — the solver
coordinate is now priced end to end: discretization (`01`–`02`),
score error (`03`–`04`), noise level (`05`), and equilibration
(`06`), with the multimodal boundary honestly marked as the point
where only the PATH coordinate `P` (the forward process's early
mixing) saves the enterprise.

## What Remains Open

Quantitative mode-weight accuracy — closed by retrofit:
`statistical_theory/06` proves the mode-weight bound (score error
times an `O(1)` mode-Fisher budget concentrated at the merge scale),
making the "weights decided at high noise" mechanism two-sided:
decided early, undecidable late;
optimal corrector placement and budget (the exchange-rate reading
suggests corrector steps belong at mid-noise: no proof); and LSI
constants along the ACTUAL path `\{p_t\}` for structured data —
interpolating between the Gaussian rate at high `t` and
Eyring–Kramers at low `t` — where only the two endpoints are
understood.
