# The Dictionary: Flow Matching Is Score Matching, Rotated

## The Question

Two communities train the same models with different words: one
regresses noise on a VP path, the other regresses velocities on a
linear path. This file proves the exact translation on Gaussian
paths — velocity is an affine combination of position and score, with
derived coefficients — and draws the two consequences the slogan
misses: the training objectives differ ONLY by a time-dependent
weighting (the estimand coordinate is a reweighting choice, now a
theorem across frameworks), and the velocity field stays regular at
the very endpoint where the score blows up (the analytic reason the
FM parametrization won).

## The Setting

Gaussian path in interpolant form (`02`, with the source absorbed
into the latent):

```math
x_t \;=\; \beta_t\,x_1 + \gamma_t\,z,
\qquad x_1 \sim p_{\mathrm{data}},\ z\sim N(0,I),
```

covering VP diffusion (`\beta = \bar\alpha`, `\gamma = \sigma`, time
reversed) and linear-path FM from Gaussian noise
(`\beta = t, \gamma = 1-t`). Available objects: the score
`s(x,t) = -E[z|x_t=x]/\gamma_t` (`02`, Theorem 1) and the marginal
velocity `v^*(x,t) = E[\dot\beta x_1 + \dot\gamma z\,|\,x_t=x]`
(`01`).

## The Dictionary Theorem

**Theorem.** On the Gaussian path,

```math
v^*(x, t)
\;=\;
\frac{\dot\beta_t}{\beta_t}\,x
\;+\;
\gamma_t\Big(\frac{\dot\beta_t}{\beta_t}\gamma_t - \dot\gamma_t\Big)\,s(x, t):
```

velocity `=` (a known scalar) `\times` position `+` (a known scalar)
`\times` score. Equivalently, through `score_foundations/02`'s
dictionary, an affine function of any of the four estimands.

*Proof.* Two linear equations in the two conditional expectations.
Taking `E[\,\cdot\,|x_t = x]` of the interpolant identity:
`x = \beta_t\,E[x_1|x] + \gamma_t\,E[z|x]`. The score formula gives
`E[z|x] = -\gamma_t\,s`. Substitute:
`E[x_1|x] = (x + \gamma_t^2 s)/\beta_t` — which is Tweedie, re-derived
as a consistency check. Then

```math
v^* = \dot\beta_t\,\frac{x + \gamma_t^2 s}{\beta_t} + \dot\gamma_t\,(-\gamma_t s)
= \frac{\dot\beta_t}{\beta_t}\,x + \Big(\frac{\dot\beta_t}{\beta_t}\gamma_t^2 - \dot\gamma_t\gamma_t\Big)s. \qquad\blacksquare
```

**Consistency check against the foundations.** For VP (in diffusion
time), `\dot\beta/\beta = -\beta_t^{drift}/2`-type and the coefficient
of `s` reduces to `-g^2/2` — the probability-flow ODE of
`score_foundations/03`, recovered exactly (two conventions, one
field). The FM flow on a Gaussian path IS the PF-ODE; sampling-side,
nothing new was invented, and phase A's solver theory applies
verbatim.

## Consequence 1: One Objective Family, Different Weights

The CFM regression on `v` and the DSM regression on `\varepsilon`
target affinely-related functions, so (bias-variance algebra on
linear maps of targets) the population objectives coincide up to a
time-dependent scalar reweighting `w(t)` computable from the
coefficients above. Under INFINITE capacity, identical minimizers
(the dictionary); under finite capacity, the choice of framework is
EXACTLY a choice of `w(t)` — which noise levels the model's limited
accuracy is spent on — extending `score_foundations/02`'s weighting
audit across frameworks: DDPM-vs-FM is not diffusion-vs-flows, it is
one estimator family indexed by (path schedule, loss weight), and
empirical comparisons that vary both at once measure the pair, not
the philosophy.

## Consequence 2: Regularity At The Data End

As `t \to 1` (`\gamma \to 0`): the score coefficient
`\gamma(\tfrac{\dot\beta}{\beta}\gamma - \dot\gamma) \to 0` — the
velocity remains bounded while `s` itself diverges like
`\gamma^{-1}`-to-`\gamma^{-2}` on rough data
(`score_foundations/06`): the divergence is exactly cancelled by the
vanishing coefficient. The FM estimand is the PRODUCT, learned as one
bounded object; the score estimand learns the diverging factor and
multiplies later. Same function, catastrophically different
conditioning at the boundary — the analytic content of "FM trains
more stably near the data," and the mirror of `02`'s classification
(the score exists where `\gamma > 0`; the velocity exists even where
it does not).

## Load-Bearing Audit

```text
Gaussian latent            the score formula and hence the dictionary;
                           non-Gaussian sources with gamma = 0 have a
                           velocity but no such rotation — the
                           dictionary is a GAUSSIAN-path theorem;
beta_t > 0                 division in Tweedie/consistency; at the
                           noise end use the symmetric form in E[z|x];
infinite vs finite capacity Consequence 1's fork — the slogan "FM =
                           diffusion" is the infinite-capacity half;
                           the finite-capacity half is a weighting
                           decision with empirical consequences;
time convention            FM time here; the check against
                           foundations/03 crosses conventions and is
                           stated with both signs traced.
```

## Position In The Coordinate System

The estimand coordinate `s` closed out: across every Gaussian path,
score, noise, denoiser, velocity are one function in four affine
frames (foundations/02 within diffusion; this file across
frameworks), and the honest residual content of "which framework" is
(i) the path schedule, (ii) the loss weight, (iii) boundary
conditioning — three explicit dials, no philosophy. Phase B's
remaining file prices the last folklore claim attached to FM: the
optimal-transport story.

## What Remains Open

The finite-capacity weighting question, now sharpened by the
dictionary into a single well-posed problem (choose `w(t)` for a data
class and architecture — the third appearance of this open problem,
and the dictionary means solving it once solves it for every
framework); and the non-Gaussian-path dictionary: for interpolants
with non-Gaussian sources and `\gamma > 0`, the score formula
survives but the clean two-term rotation does not — what replaces it
is a conditional-moment expansion nobody has organized.
