# Guidance As Tilting

## The Question

Conditional generation steers the sampler with modified scores:
classifier guidance adds a likelihood gradient; classifier-free
guidance (CFG) extrapolates between conditional and unconditional
scores. The first is an identity. The second is the field's most-used
and least-understood knob: this file proves what CFG's modified score
is the gradient OF, and then proves the theorem that deflates it —
the tilted family it points at is NOT the diffusion path of any data
distribution, with the gap identified exactly as a Jensen gap. This
is the third repository in this series whose central object is an
exponential tilt; the audit says what is different this time.

## Classifier Guidance Is Bayes, Proved

**Proposition.** For the joint forward process (noise `x_0 | y` as in
`01`, `y` untouched):

```math
\nabla_x\log p_t(x\,|\,y) \;=\; \nabla_x\log p_t(x) + \nabla_x\log p_t(y\,|\,x).
```

*Proof.* `p_t(x|y) = p_t(x)\,p_t(y|x)/p(y)`; take `\log`, then
`\nabla_x` (the `p(y)` term dies). ∎

Exact — IF one has the noise-level-`t` classifier `p_t(y|x)`, i.e. a
classifier trained on noised inputs. Every approximation in practice
(clean classifiers on noised data, one net for all `t`) is an
approximation of THAT object, and `guidance_and_control/01` shows the
exact version is a Doob `h`-transform.

## What CFG's Score Is The Gradient Of

CFG runs the sampler with
`s_\omega = s_t(x) + \omega\,\big(s_t(x|y) - s_t(x)\big)`.

**Proposition.** Pointwise in `t`,

```math
s_\omega(x)
= \nabla_x \log\Big[\,p_t(x|y)^{\omega}\ p_t(x)^{1-\omega}\,\Big]
\;=:\; \nabla_x\log \tilde p_t^{(\omega)}(x)
```

(up to the normalizer, which has zero gradient): the CFG score is the
exact score of the geometric tilt of conditional against
unconditional, at each noise level separately. *Proof.* Expand the
logarithm. ∎ For `\omega > 1` this is sharpening — the conditional
raised to a power, the unconditional discounted — the same
exponential-tilt object as Bellman's Ledger's KL-regularized optimum
(`rlhf_mathematics/02`) and the soft Bellman operator. Third
repository, same transform.

## The Theorem: The Tilted Family Is Not A Diffusion Path

Here is where diffusion differs from the bandit/RLHF setting: the
sampler does not draw from one tilted distribution — it runs a
DYNAMICS whose drift at each `t` is the tilted score, and exactness
would require the family `\{\tilde p_t^{(\omega)}\}_t` to be the
forward-noised path of its own endpoint. It is not:

**Theorem (the Jensen gap).** Fix the forward process and let
`r(x_0) = p_0(x_0|y)/p_0(x_0)` be the data-level likelihood ratio.
Differentiate the two candidate constructions at `\omega = 1`:

```text
tilt-then-noise:   d/dω log [ noise( p_0(·|y)^ω p_0^{1-ω} ) ]_t (x)
                     = E[ log r(x_0) | x_t = x ]
noise-then-tilt:   d/dω log [ p_t(x|y)^ω p_t(x)^{1-ω} ]
                     = log E[ r(x_0) | x_t = x ]
```

and the two differ at every `(x, t)` where the posterior
`x_0 \mid x_t = x` does not make `r` a.s. constant, by strict Jensen:
`\log E[r] > E[\log r]`. Hence for any data with a genuinely
informative condition, the CFG family `\tilde p_t^{(\omega)}`
disagrees, already at first order in `\omega - 1`, with the noised
path of ANY fixed tilted data distribution — running the diffusion
with CFG's score does not sample `p_0(x|y)^\omega p_0(x)^{1-\omega}`,
nor any other single distribution obtained by tilting the data.

*Proof.* First expression: the noised density of the
`\omega`-tilted data is
`\int k_t(x|x_0)\,p_0(x_0)\,r(x_0)^\omega\,dx_0/Z_\omega`;
its `\log`-derivative at `\omega = 1` is
`\int k\,p_0(\cdot|y)\log r\,/\,p_t(x|y) - (\text{normalizer term})
= E[\log r\,|\,x_t = x]` under the CONDITIONAL posterior, minus a
constant in `x`. Second expression: direct differentiation of the
displayed tilt, using
`p_t(x|y)/p_t(x) = E[r(x_0)|x_t = x]` (insert
`k_t p_0` and divide — the same mixture manipulation as `02`).
Strictness: Jensen for the strictly concave `\log`, strict unless
`r(x_0)` is degenerate given `x_t`. Constants in `x` do not affect
scores; the derivative SCORES differ by
`\nabla_x\{\log E[r|x_t=x] - E[\log r|x_t=x]\} \not\equiv 0`
generically. ∎

The reading, in one honest paragraph: **CFG at `\omega > 1` is a
per-noise-level modification with no single target distribution.**
Its observed effects — mode sharpening, over-saturation, reduced
diversity beyond what conditioning warrants — are the compounding of
the Jensen gap along the trajectory: at high noise, posteriors are
diffuse, `\log E > E\log` is large, and the guided drift pushes
harder toward high-likelihood-ratio regions than any static tilt
would. This is a mechanism-level account, proved at first order; the
full nonperturbative description of what CFG samples is open
(`guidance_and_control/02` collects the known partial answers:
interval guidance, rescalings, limiting `\omega` behavior). One
positive fence: for jointly Gaussian `(x_0, y)` the ratio `r` is
log-quadratic, everything stays Gaussian, tilt and noise commute, and
CFG is exact — the counterexample needs non-Gaussian data, and the
theorem says any informative non-Gaussian case suffices.

## Load-Bearing Audit

```text
p_t(y|x) at noise level t   classifier guidance's exactness lives or
                            dies on the classifier seeing NOISED
                            inputs; clean classifiers give a different
                            (biased) h;
the mixture identity        p_t(x|y)/p_t(x) = E[r|x_t] — the same
                            insert-the-kernel step as Tweedie (02),
                            fourth use in this family;
strict concavity of log     the strictness of the gap; degenerate r
                            (deterministic labels given x_t) closes it
                            — late in the trajectory CFG becomes
                            locally consistent, early it does not;
first order in omega        the theorem is perturbative; it PROVES
                            inconsistency, it does not characterize
                            the omega = 7.5 sampler (open).
```

## Position In The Coordinate System

A modification of `s` that changes what `S` converges to: guidance
lives entirely in the estimand slot, and this file's theorem is the
statement that the slot is not closed under the operations practice
performs on it. The Bellman bridge is booked: tilting a DISTRIBUTION
is `rlhf_mathematics/02`; tilting a PATH of distributions is a
stochastic control problem, and doing it exactly is
`guidance_and_control/04`'s h-transform/value-function file.

## What Remains Open

The nonperturbative question (what does `\omega`-CFG sample, as a
function of data and schedule — no answer beyond Gaussian cases and
the first-order gap above); principled `\omega` schedules (interval
guidance works empirically; the theorem suggests WHY — gate the
guidance where posteriors are diffuse — but no optimality result
exists); and the interaction with distillation (phase D): one-step
students of guided teachers inherit a target that is not a
distribution, and what THEY sample is doubly open.
