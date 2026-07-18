# Failure Modes Of Steered Sampling

## The Question

The family's theorems say steering is exact with the right `h` and
priced with the wrong one. Practice steers hard with wrong `h`'s.
This closing file collects the pathologies — each with its mechanism
traced to a proved result, each with its fence — in the tradition of
the failure-mode files of the other two ledgers. Four families of
failure, one table of causes.

## 1. Over-Sharpening And Diversity Collapse

Mechanism, proved upstream: `02`'s Gaussian theorem — CFG at
`\omega > 1` samples a law strictly narrower than the conditional
(`\Lambda_\omega \succeq \Lambda_c`), centered past it
(`m_\omega` extrapolates); and `02`'s endpoint — `\omega \to \infty`
is classifier ASCENT, whose output is a point. Diversity loss is not
guidance misbehaving; it is the `\omega`-dial doing exactly what the
closed form says, interpolating sampling into optimization. The
practical corollary the theorems license: diversity-quality tradeoffs
should be reported as functions of `\omega` (they are frontier
points, not model properties), and diversity lost to `\omega` cannot
be recovered by seed count — the LAW is narrow, not the sample.

## 2. Leaving The Data Manifold

The guided drift adds `\omega\,g^2\nabla\log \hat h` — a vector with
no obligation to be tangent to the data's support. The restoring
force against off-support excursions is the prior score, whose
normal component scales as `\sigma_t^{-2}` near the support
(`statistical_theory/04`; the two-point case is
`score_foundations/06`'s sharpening). The balance, in the solvable
toy: smoothed two-point data, constant guidance pull `c` — the
stationary displacement off the atom solves
`s(x) + \omega c = 0`, i.e. displacement
`\approx \omega c\,\sigma_t^2`: **negligible late, large early.**
Early displacement then meets the solver dichotomy
(`samplers_and_convergence/05`): the ODE transports it faithfully to
the output; the SDE partially re-contracts it. Assembled mechanism:

```text
strong guidance + high noise  => off-manifold displacement ~ omega sigma_t^2
+ ODE-style sampler           => displacement survives to the sample
= the classic artifacts       (saturated colors, waxy textures,
                              geometry violations) — and the standard
                              mitigations map one-to-one: interval
                              guidance (kill the omega where sigma is
                              large — 02's gap bound), stochastic
                              churn (borrow the SDE's contraction),
                              rescaled CFG (bound the displacement).
```

Each mitigation is an empirical practice whose reason-for-working is
a theorem in this repository; none has an end-to-end guarantee.

## 3. Reward Hacking, Third Ledger

`04` proved fine-tuning tilts toward `e^{r/\beta}` — for the reward
you TRAINED, which is a proxy. Bellman's `rlhf_mathematics/05`
(Goodhart as max-bias) transfers verbatim along `04`'s bridge: tilting
toward a proxy with errors selects the errors' upper tail, with the
overshoot growing in the effective selection pressure (`1/\beta`, or
`n` for best-of-`n`). Diffusion adds one specific twist worth its own
line: the proxy reward is typically a CLEAN-data functional evaluated
on generated samples that are only approximately on-support — so the
proxy is queried precisely where it was never validated (its own
off-distribution regime), coupling failure 2 to failure 3: the
sampler drifts off-manifold BECAUSE the off-manifold proxy values are
uncontrolled, and the tilt seeks exactly the uncontrolled region. The
support-is-destiny theorem (`04`) fences the idealized optimum, not
the approximate algorithms — PPO-style updates with function
approximation can and do assign mass where `P` had none, which is a
violation of the objective the theory can DETECT (measure the
per-step Girsanov KL — it diverges) but algorithms rarely monitor.

## 4. Compounding With Distillation

Statement-level, fenced for phase D: distilled one-step samplers are
trained to match a TEACHER trajectory family; guidance applied to a
distilled student (or distillation of a guided teacher —
"CFG-distillation") bakes a specific `\omega` into the weights. By
`score_foundations/05` the guided family is not a diffusion path, so
the student is matching a target outside the class its
self-consistency machinery assumes — the resulting error has no
current theory, and the empirical practice (distill each `\omega`
separately, or condition the student on `\omega`) is an
acknowledgment of the problem, not a solution. The full account
belongs to `distillation/04`.

## The Table Of Causes

```text
symptom                 proved mechanism                     file
over-saturation         mean extrapolation m_omega           02
diversity collapse      precision growth Lambda_omega;       02
                        omega -> inf = ascent
early over-guidance     Jensen gap max at diffuse posteriors 01/02
                        + DPS covariance deletion (x51)      03
artifacts under ODE     off-manifold displacement omega
                        sigma^2 transported, not healed      05/A-05
inpainting seams        hard constraints applied at t > 0    03
reward hacking          proxy-tail selection (Goodhart)      04 + B-rlhf/05
                        + proxy queried off-validation       2 + 3 coupled
```

## Load-Bearing Audit

```text
mechanism != guarantee     every row explains; none bounds — the
                           honest register of this file;
toy computations           the displacement estimate is the smoothed
                           two-point case; real manifolds have
                           anisotropic restoring the toy flattens;
solver attribution         the ODE/SDE split is proved for score
                           BIAS (A/05); its extension to guidance
                           drift is the same computation with u in
                           place of b — done, but only in the
                           Gaussian caricature.
```

## Position In The Coordinate System

The family's closing audit: steering modifies `s`, the solver `S`
decides what survives, and every observed pathology factors as
(wrong-`h` error) `\times` (solver dynamics) `+` (proxy-tail
selection). Nothing here required new mathematics — which is the
point of a ledger: the failures were already priced by the theorems;
this file is the receipts, filed.

## What Remains Open

The guidance-times-solver theorem (the family's thrice-flagged gap:
one bound combining `01/03`'s drift error profile with `A/05`'s
transport-vs-contraction dynamics — all pieces exist, nobody has
assembled them); reward-model uncertainty as a first-class object
(tilting by a POSTERIOR over rewards rather than a point estimate —
the Bayesian repair to failure 3, with the `rlhf/05` KL-budget
heuristic as its crude shadow); and diversity accounting under
guidance at scale — failure 1 predicts the law's entropy loss as a
function of `\omega` exactly in the Gaussian case, and nobody has
checked how far the caricature tracks real models.
