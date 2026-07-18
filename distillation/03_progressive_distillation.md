# Progressive Distillation

## The Question

Salimans–Ho's route to few steps: a student learns to do in ONE DDIM
step what the teacher does in TWO; repeat, halving the step count
each round. Three questions with exact answers: what is the student's
regression target (derived below — it is a closed-form "effective
denoiser," not the true one); when is the halving LOSSLESS (the
Gaussian case, proved); and how do errors compound across rounds
(linear in rounds, by `01`'s telescope, with the curvature the
student must absorb quantified by phase A's quadrature reading).

## The Target, Derived

Teacher: two DDIM steps (`score_foundations/04`) from `(x, t)`
through `t' = t - h` to `t'' = t - 2h`, producing `x''`. Student: ONE
DDIM step from `(x, t)` to `t''` — which, by the DDIM formula, is
determined by whatever denoiser value `\tilde x_0` it uses:

```math
x''_{\mathrm{student}} \;=\; \alpha_{t''}\,\tilde x_0 + \frac{\sigma_{t''}}{\sigma_t}\big(x - \alpha_t\,\tilde x_0\big).
```

**Proposition (the effective denoiser).** Setting
`x''_{\mathrm{student}} = x''` and solving:

```math
\tilde x_0(x, t)
\;=\;
\frac{x'' - \tfrac{\sigma_{t''}}{\sigma_t}\,x}{\alpha_{t''} - \tfrac{\sigma_{t''}}{\sigma_t}\,\alpha_t},
```

well-defined whenever `\alpha_{t''}\sigma_t \ne \sigma_{t''}\alpha_t`
— i.e. whenever the two times have different signal-to-noise ratios,
which is every non-degenerate step. *Proof.* The DDIM update is
affine in `\tilde x_0` with the displayed coefficient; invert. ∎

This is Salimans–Ho's target: the student regresses on a SYNTHETIC
denoiser — the unique one that makes a single coarse DDIM step
reproduce the teacher's two fine steps. The reading via
`samplers_and_convergence/02`'s quadrature identity is the honest
one: the true one-step map at width `2h` differs from the two-step
map by the quadrature error (curvature of `\hat x_0` in the
`\rho`-clock); the effective denoiser ABSORBS that curvature into the
estimand. Progressive distillation converts solver error into
function-approximation error, one octave at a time — and the student
network can only absorb what it can represent, which is why the
method's failure mode is capacity, not consistency.

## The Lossless Case, Proved

**Theorem (corrected — the original claim was falsified by
verification/verify.py).** The halving is lossless iff the denoiser
is frozen along trajectories — true for point-mass data
(`\hat x_0 \equiv \mu`), FALSE for isotropic Gaussians: there
`\hat x_0 = vu/(v+\rho^2)` varies along the path, two DDIM steps do
not compose to one (the `u`-coefficients `(v+\rho_i\rho_j)/(v+\rho_j^2)`
multiply to something strictly below the one-step coefficient, by
Cauchy–Schwarz), and the effective denoiser `\tilde x_0` differs from
the true one — which is exactly why the target derivation above is
needed at all. The student absorbs a REAL, computable curvature even
in the Gaussian case; the lossless case is the degenerate one. ∎

The calibration reading, as always in this repository: the solvable
case is the UNREPRESENTATIVE one — it is precisely the data whose
quadrature error is zero, i.e. the data that never needed distilling.
The theorem's value is the contrapositive: everything progressive
distillation must learn is `\hat x_0`'s variation along trajectories
— the same quantity that priced few-step solvers
(`samplers_and_convergence/02`) and consistency-map regularity
(`01`). One number rules the whole few-step enterprise: the
denoiser's curvature in the noise clock.

## Error Across Rounds

Round `\ell` trains student `\ell` against teacher = student
`\ell-1`, with training error `\epsilon_\ell` (in output norm, per
step). Two error channels compound:

```text
inheritance   student ell's target is BUILT from student (ell-1)'s
              map: any error in the teacher propagates through the
              affine target formula with the displayed coefficients —
              bounded coefficients (audit), so inherited error passes
              with O(1) amplification per round;
accumulation  at round ell the map covers 2^ell h per step and
              N/2^ell steps remain: 01's telescope over the remaining
              grid bounds the total by the sum of per-step errors.
```

**Proposition (linear-in-rounds bound).** With per-round training
error `\epsilon_\ell` and coefficient bound `C` on the target's
teacher-sensitivity, the final one-step map's error is
`\le \sum_\ell C^{L-\ell}\epsilon_\ell` — geometric only through
`C`; for schedules keeping `C \approx 1` (SNR-separated steps, the
proposition's denominator bounded away from zero), effectively the
SUM of the per-round errors. *Proof sketch, honest:* the affine
target map's Lipschitz constant in the teacher's output is
`|\alpha_{t''} - \sigma_{t''}\alpha_t/\sigma_t|^{-1}`-scaled;
composing rounds multiplies these; telescoping within a round is
`01`. The full constant-tracking is bookkeeping around these two
displayed facts. ∎

Compare `02`: consistency methods pay `N` leaks once; progressive
distillation pays `L = \log_2 N` rounds of training error plus
whatever curvature exceeds the student's capacity. Neither dominates
in theorem-space; the empirical migration to consistency-style
methods reflects optimization, not an asymptotic (statement, and the
scoreboard file's business).

## Load-Bearing Audit

```text
DDIM as the step form        the affine-in-denoiser structure is what
                             makes the target solvable in closed form;
                             distilling stochastic samplers needs a
                             distributional target (statement: e.g.
                             moment matching), not an inversion;
SNR separation               the denominator; adjacent times with
                             equal SNR make the one-step map
                             non-invertible in x0-tilde — degenerate
                             schedules break the derivation, visibly;
teacher exactness at round 0 the base teacher is a many-step DDIM
                             sampler: its own error (A/02's
                             quadrature) is inherited by ALL rounds —
                             the floor under everything;
capacity                     the absorbed curvature must be
                             representable; the theorem-level analysis
                             cannot see this, and it is where the
                             method actually bites.
```

## Position In The Coordinate System

The second route to a learned solution map: where consistency methods
(`01`–`02`) learn the map against its own invariance, progressive
distillation learns it by INDUCTION ON STEP WIDTH, with the estimand
redefined each round to carry the accumulated quadrature. Same
destination, same governing quantity (denoiser curvature), different
error economics — which is exactly what a coordinate system is for.

## What Remains Open

Optimal round schedules (halving is a choice; the accumulation bound
suggests unequal ratios keyed to the curvature profile — underived);
distilling STOCHASTIC teachers with guarantees (the affine inversion
is deterministic-only, and A/05 says stochasticity is where
robustness lives — the gap between the best-understood distillation
and the most robust teacher is real); and a lower bound: no result
says how much error ANY one-step map must incur on given data
(a capacity-independent floor in terms of the flow's curvature would
complete this family's theory and none exists).
