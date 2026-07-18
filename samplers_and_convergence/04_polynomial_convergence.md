# Polynomial Convergence

## The Question

Assemble `03`'s decomposition into the headline theorems: with an
`L^2`-accurate score, diffusion sampling reaches `\varepsilon`
accuracy in polynomially many steps — for essentially ARBITRARY data.
This file states the two generations of results precisely, assembles
the proof skeleton from the family's proved parts, and spends most of
its length where an honest file should: on what the theorems do and
do not say.

## The Statements

**Theorem (first generation — Chen–Chewi–Li–Li–Salim–Zhang 2023;
statement).** For data with bounded second moment and Lipschitz
scores along the path, the EM-discretized reverse SDE with a score
satisfying
`\frac1T\int E_{p_t}\|s_t-\hat s_t\|^2 \le \varepsilon_{sc}^2`
outputs a law within
`O(\varepsilon_{sc} + \text{poly}(d)\,\sqrt{h}\cdot(\cdot) + e^{-T})`
in TV of the (slightly smoothed) data — polynomial iteration
complexity in `(d, 1/\varepsilon)`, with NO log-concavity or
isoperimetry assumed of the data.

**Theorem (second generation — Benton–De Bortoli–Doucet–Deligiannidis
2024; statement).** Under finite second moment ONLY (no Lipschitz
score assumption; early stopping at `t_{\min}`, i.e. against the
`\delta`-perturbed data), the iteration complexity improves to

```math
K \;=\; O\!\Big(\frac{d\,\log^2(1/\delta)}{\varepsilon^2}\Big)
```

steps for `\varepsilon^2` KL error — NEARLY LINEAR in dimension, via
a stochastic-localization view of the same process and exponentially
decaying step sizes near `t_{\min}`.

## The Assembled Skeleton

Every term traces to a proved piece of this repository:

```text
KL(p_delta || output)
   <=  KL(p_T || gamma)              score_foundations/01: W2/KL
                                     contraction of the forward OU —
                                     e^{-Theta(T)}: choose T ~ log(1/eps);
   +   int g^2 E||s - s_hat||^2      03's Girsanov term = the training
                                     loss (score_foundations/02): the
                                     ASSUMED budget eps_sc^2;
   +   discretization sum            03's third term; bounding it is
                                     the genuine labor:
                                     1st generation: Lipschitz scores +
                                     01's Ito-Taylor bookkeeping =>
                                     poly(d) h per step;
                                     2nd generation: replace assumed
                                     Lipschitzness by the PATH's OWN
                                     regularity — the forward kernel's
                                     Gaussian smoothing bounds score
                                     derivatives ALONG p_t in terms of
                                     moments alone (the technical core,
                                     theirs), with geometric steps
                                     h_k ~ t_k absorbing the t_min
                                     blowup.
```

The step-size schedule of the second generation is worth naming
because it closes a loop with `01`: steps proportional to the current
time (uniform in `\log t`, i.e. uniform in log-SNR) are exactly what
`01`'s schedule-sensitivity reading and `02`'s quadrature variable
prescribe — three files, one clock.

## What The Theorems Say

Read as this family's summary: sampling inherits NONE of the data's
hardness. Multimodality, non-convexity, arbitrary separation between
modes — properties that make direct MCMC on `p_0` exponentially slow
(`06`) — cost the diffusion sampler nothing, because the forward
process's unconditional contraction (`score_foundations/01`) did the
mixing in advance, and the reverse dynamics only ever needs to be
LOCALLY right (the score) along a path of smoothed measures. The
entire difficulty has been moved into obtaining the score.

## What They Do Not Say

The audit that keeps the headline honest:

```text
1. the score is ASSUMED. eps_sc is a hypothesis, not a conclusion —
   the theorems price sampling GIVEN estimation; estimating the score
   of arbitrary data to L2 accuracy eps_sc is precisely as hard as
   the minimax theory says (statistical_theory/02), and for worst-case
   data it is cursed. "Sampling is easy" and "generation is easy"
   differ by exactly phase F;
2. L2-under-p_t is the right norm only because Girsanov made it so;
   guided samplers leave the data path and the guarantee with it
   (03's audit; guidance_and_control/05);
3. early stopping is a modeling statement: the theorems approximate
   p_delta, the delta-smoothed data — on manifold data this is the
   difference between a density and none (statistical_theory/04);
   delta enters the rate only logarithmically, which is why practice
   never notices;
4. constants and log factors are real: "nearly d-linear" at d = 10^7
   with eps^{-2} is not a small number of steps — the theorems
   explain feasibility, not the 20-step regime; the 20-step regime is
   02's quadrature story (smooth denoisers), which is a DATA property
   the worst-case theorems cannot see;
5. KL against p_delta says nothing per-sample: a vanishing fraction
   of bad mass is invisible at eps accuracy.
```

## Load-Bearing Audit

```text
finite second moment       both generations; the only data assumption
                           left standing — heavy tails degrade the
                           prior term's contraction constants;
score error under p_t      inherited from 03 (Girsanov's evaluation
                           point) — the theorems' interface with
                           training is exactly Vincent's objective;
early stopping t_min       load-bearing for regularity (03's Novikov
                           audit) and honest as approximation target;
SDE sampler                both theorems are for the stochastic
                           endpoint; ODE analyses need 05's machinery
                           and currently pay extra assumptions —
                           the lambda-dial is not yet free in theory.
```

## Position In The Coordinate System

The family's summit for the `S` coordinate: given `(P, s)` at stated
quality, the solver provably delivers, at nearly-`d`-linear cost, no
conditions on the data's shape. The two files that remain price the
choices the theorems left open: `05` the `\lambda`-dial under score
error, `06` the corrector's role.

## What Remains Open

Matching lower bounds in `d` (is linear optimal?); the ODE-endpoint
analogue at the same generality (currently weaker — the noise is
load-bearing in the proofs, and `05` shows it is load-bearing in the
error dynamics too, so the gap may be real rather than technical);
convergence under the LEARNED (correlated-error) score rather than an
oracle `L^2` budget — the score's errors are structured, not
adversarial, and no current theorem exploits or even models that;
and the constants-versus-practice gap of reading 4, which is the
same "worst-case theory, benign practice" frontier every repository
in this series ends at.
