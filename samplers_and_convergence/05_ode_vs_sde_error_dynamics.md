# ODE vs SDE: The Error Dynamics

## The Question

`score_foundations/03` proved every `\lambda` on the dial from ODE
(`\lambda = 0`) to SDE (`\lambda = 1`) is an EXACT sampler. The
theorems of `03`–`04` covered the stochastic end; practice often
prefers the deterministic end. The dial only matters once something
is wrong — and then it matters enormously: this file proves, in the
solvable Gaussian case with a controlled score error, that the ODE
TRANSPORTS score error while the SDE CONTRACTS it, with explicit and
strikingly different amplification constants. The folklore "SDE is
more robust, ODE is faster" becomes a computation.

## The Setup

Per-coordinate Gaussian data: `p_t = N(0, 1)` for all `t`,
`s_t(x) = -x` (`score_foundations/06`). Corrupt the score with a
constant bias: `\hat s(x, t) = -x + b`. Run the `\lambda`-family
reverse dynamics (`score_foundations/03`) with `\hat s`, in reverse
time `\tau`, and track the sampled law — which stays Gaussian
(everything is affine), so the entire error is two numbers: the mean
`m(\tau)` and variance `v(\tau)` of the sampler's law.

## The Theorem

**Theorem.** The sampler's mean obeys

```math
\frac{\mathrm{d}m}{\mathrm{d}\tau}
= -\frac{\lambda^2}{2}\,\beta\,m \;+\; \frac{1+\lambda^2}{2}\,\beta\,b ,
```

so along a horizon with `\int\beta\,\mathrm{d}\tau = B`:

```text
ODE  (lambda = 0):   m' = (beta/2) b          — NO restoring term:
                     m(T) = b·B/2.            error accumulates
                                              LINEARLY in the
                                              integrated schedule;
SDE  (lambda = 1):   m' = -(beta/2) m + beta b — contraction:
                     m(infty) -> 2b,           error SATURATES at 2b
                     approached exponentially.
```

*Proof.* The reverse drift is
`\tfrac{\beta}{2}x + \tfrac{(1+\lambda^2)}{2}\beta\,\hat s(x)
= \tfrac{\beta}{2}x - \tfrac{(1+\lambda^2)}{2}\beta x +
\tfrac{(1+\lambda^2)}{2}\beta b`, whose linear coefficient is
`-\tfrac{\lambda^2}{2}\beta`; take expectations. Solve the two linear
ODEs. ∎

**Numbers, per the contract.** A standard VP schedule has
`B = \int_0^T\beta \approx 10` (this is `2\log(1/\alpha_T)`, i.e.
`\alpha_T \approx e^{-5}`: the prior-mismatch requirement of `04`
fixes `B \approx 10`–`20`). Then a score bias `b` becomes a sampled
mean error of:

```math
\text{ODE}:\ \ \frac{B}{2}\,b \;=\; 5b\ \text{to}\ 10b,
\qquad
\text{SDE}:\ \ \le 2b .
```

A 2.5–5x amplification gap, from the same score, the same schedule,
the same marginals-in-exactness — entirely a property of the error
DYNAMICS. The mechanism in one sentence: the SDE's injected noise is
subsequently re-contracted using the score's restoring force, so the
dynamics keeps overwriting its own past (errors included); the ODE
is measure transport — deterministic, invertible, memory-preserving —
and faithfully transports every error it has ever absorbed to the
output. Exactness and robustness are different virtues, and the
`\lambda`-dial trades them.

**Variance version (same computation).** Perturb the score's slope
instead (`\hat s = -(1+\epsilon)x`): the variance ODE gives the same
dichotomy — ODE: multiplicative error compounding across the horizon;
SDE: geometric-mixing toward a fixed point with `O(\epsilon)` offset.
Nothing about the dichotomy is special to bias. ∎ (one-line
adaptation of the proof)

## The Other Side Of The Ledger

Two honest counterweights, so the theorem is not over-read:

```text
1. discretization: the SDE's noise must itself be integrated; at K
   steps the stochastic endpoint pays the sqrt(h)-type terms of
   01/03–04 while the ODE pays only 02's quadrature error — with a
   GOOD score and few steps, the ODE's transport is transporting
   almost nothing, and it wins (the empirical 20-step regime);
2. the contraction is data-dependent: the SDE re-contracts errors at
   the rate of the score's restoring force — for multimodal data the
   force vanishes between modes, the contraction stalls there, and
   the SDE's advantage concentrates within basins (06's subject: the
   corrector is this contraction, run deliberately).
```

Combined design statement, as proved here plus `02`–`04`: **score
error favors `\lambda > 0`; evaluation budget favors `\lambda = 0`;
the practical optimum moves toward the SDE exactly when the score is
poor or the guidance is strong** (guided sampling injects deliberate
score corruption — `score_foundations/05` — and empirically prefers
stochastic churn, which this theorem predicts).

## Load-Bearing Audit

```text
Gaussian/affine everything   the sampled law stays Gaussian, making
                             the error two ODEs — the same
                             solvable-case strategy as
                             score_foundations/06, and the reason the
                             constants are exact;
constant bias b              the cleanest corruption; state-dependent
                             errors decompose against the OU
                             eigenfunctions and each mode obeys the
                             same dichotomy with its own rate
                             (statement);
lambda in {0, 1}             the computation runs for every lambda —
                             the restoring coefficient is
                             -(lambda^2/2)beta: robustness is bought
                             QUADRATICALLY in lambda, another exact
                             design fact from the same two lines;
B ~ 10-20                    the numbers; shorter horizons shrink the
                             gap and the prior term (04) grows —
                             linked dials.
```

## Position In The Coordinate System

The `\lambda`-dial priced: `score_foundations/03` proved the dial
free in exactness; this file proves it expensive in robustness, with
the exchange rate computed. Together with `02` (deterministic error =
denoiser quadrature) and `03`–`04` (stochastic guarantees), the
solver coordinate is now priced at both endpoints and interpolated.

## What Remains Open

The optimal `\lambda(t)` schedule under a realistic (state-dependent,
learned) error model — the constant-bias calculation says "more noise
where the score is worse," which is where guidance is strongest and
`t` is smallest, but no theorem selects the schedule; the multimodal
contraction gap (the audit's item 2) quantified beyond basin
heuristics — Eyring–Kramers-type rates for the reverse-time process
would do it and do not exist; and the interaction with `04`'s open
ODE-endpoint theory: a Girsanov-free analysis matching the SDE
theorems at the same generality remains the family's missing
theorem.
