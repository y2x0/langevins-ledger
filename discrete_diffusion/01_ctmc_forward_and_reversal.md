# CTMC Forward Processes And The Discrete Reversal

## The Question

Diffusion on token spaces replaces the SDE with a continuous-time
Markov chain: states are sequences, noise is random token corruption,
and the whole apparatus of this repository needs discrete
counterparts. This file supplies the two foundations: the reversal
theorem — the discrete Anderson theorem, proved at the
master-equation level in exact parallel to
`score_foundations/03` — which identifies the discrete "score" as a
RATIO of marginal probabilities; and the closed-form analysis of the
two standard forward chains (uniform and absorbing/masking), the
second of which the rest of the phase runs on.

## The Setup

Finite state space `\mathcal{X}`; forward CTMC with time-varying rate
matrix `Q_t(x, y) \ge 0` for `y \ne x`, diagonal
`Q_t(x,x) = -\sum_{y\ne x}Q_t(x,y)`; marginals obey the master
equation

```math
\frac{\mathrm{d}}{\mathrm{d}t}\,p_t(y) \;=\; \sum_{x} p_t(x)\,Q_t(x, y).
```

## The Discrete Anderson Theorem, Proved

**Theorem.** The time reversal of the chain (reverse clock
`\tau = T - t`) is again a CTMC, with rates

```math
\bar Q_\tau(y, x) \;=\; Q_t(x, y)\ \frac{p_t(x)}{p_t(y)}
\qquad (x \ne y,\ t = T-\tau):
```

reverse rate from `y` to `x` = forward rate from `x` to `y`, weighted
by the marginal ratio.

*Proof.* It suffices to check that `q_\tau := p_{T-\tau}` solves the
reverse master equation. Left side:
`\partial_\tau q(x) = -\partial_t p(x) = -\sum_y p(y)Q(y,x)`. Right
side, with the displayed rates and their diagonal:

```math
\sum_{y\ne x} q(y)\,\bar Q(y,x) \;-\; q(x)\sum_{y\ne x}\bar Q(x,y)
= \sum_{y\ne x} p(y)\,\frac{Q(x,y)\,p(x)}{p(y)}
\;-\; p(x)\sum_{y\ne x} \frac{Q(y,x)\,p(y)}{p(x)}
```

```math
= p(x)\sum_{y\ne x}Q(x,y) \;-\; \sum_{y\ne x}p(y)\,Q(y,x)
= -\,p(x)\,Q(x,x) - \Big[\sum_y p(y)Q(y,x) - p(x)Q(x,x)\Big]
= -\sum_y p(y)\,Q(y,x). \qquad\blacksquare
```

The structural parallel to the continuous theorem deserves its
sentence: there, the reversal's whole content was
`p\nabla\log p = \nabla p`; here it is the two telescoping sums —
and the object the reverse dynamics needs from learning is no longer
a gradient but the family of **marginal ratios**
`r_t(x, y) = p_t(y)/p_t(x)` between states connected by the forward
rates. The discrete score IS this ratio; `03` builds its estimation
theory, and the pathwise justification (reversed Markov chains are
Markov with these rates — the classical two-time-marginal argument)
is cited rather than re-proved, exactly as Anderson's SDE-level
statement was in the continuous file.

## The Two Forward Chains

Text diffusion factorizes the corruption per token: each of `L`
positions runs an independent single-token chain on the vocabulary
`V` (plus, for masking, an extra symbol `M`).

**Absorbing (masking) chain, solved.** Rates: every real token jumps
to `M` at rate `\beta_t`; `M` is absorbing (rate out `= 0`).

```math
P\big(x_t^\ell = x_0^\ell\big) = \alpha_t = e^{-\int_0^t\beta},
\qquad
P\big(x_t^\ell = M\big) = 1-\alpha_t,
```

and nothing else is reachable. *Proof.* The token survives iff the
exponential clock has not rung: survival probability of an
inhomogeneous exponential; absorption gives the two-point support. ∎
Three consequences that make masking the practical winner and the
rest of this phase's setting: the marginal is a TWO-point mixture per
token (the state either IS the answer or carries no information —
no intermediate corruption to model); the needed ratios reduce to
conditional token posteriors (`03`'s theorem); and the reverse
process only ever UNMASKS — monotone generation, `04`'s bridge to
autoregression.

**Uniform chain, stated with its derivation sketch.** Rates
`\beta_t/|V|` to every other token: the single-token marginal
interpolates `p_0^\ell \to \mathrm{Unif}(V)` with the same
exponential weight (`p_t = \alpha_t p_0 + (1-\alpha_t)\mathrm{Unif}`,
by diagonalizing the doubly stochastic generator — one line of
spectral calculus). The price relative to masking: a corrupted token
looks like a real one — the reverse model must decide WHETHER a
token is noise, not just what a mask hides; empirically and in the
ELBO's conditioning this is the harder problem (statement; the
phase's theorems are run on masking, with the uniform case's
analogues noted where they differ).

## Load-Bearing Audit

```text
p_t > 0                  the ratio in the reversal; masking violates
                         it benignly (unreachable states have rate 0
                         into them anyway) — the pairing of zeros is
                         what to check, and absorbing chains pass;
master-equation level    as in the continuous file: marginals
                         identified, path law cited — sampling needs
                         the former;
per-token independence   the factorized forward; the REVERSE does not
                         factorize (unmasking one token should depend
                         on all others) — the tension 05 prices;
time-inhomogeneity       carried by beta_t throughout; nothing used
                         stationarity.
```

## Position In The Coordinate System

The path coordinate `P` on discrete spaces: corruption chains in
place of OU, survival probabilities in place of `\alpha_t`-schedules
(literally the same symbol, by design), and the estimand `s`
identified by the reversal theorem as the marginal-ratio family.
`02`–`03` make the estimand trainable, `04` reveals the masking
path's exact relationship to autoregression, `05` prices the solver.

## What Remains Open

Forward-chain design beyond the two classics: state-dependent or
data-adapted corruption (semantic noise) breaks the closed-form
marginals that everything downstream uses — the discrete version of
the non-OU question from `score_foundations/01`, open in the same
way; and the correct discrete analogue of the `\lambda`-dial:
reverse chains can be sped or slowed against correctors
(`05`'s remasking), but a clean one-parameter family of
marginal-preserving discrete samplers with a tunable
stochasticity — the analogue of `score_foundations/03`'s theorem —
has only partial versions.
