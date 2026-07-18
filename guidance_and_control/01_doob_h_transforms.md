# Doob h-Transforms: Exact Conditioning

## The Question

`score_foundations/05` proved classifier guidance is Bayes — given the
noise-level likelihood `p_t(y|x)`. This file identifies that object
structurally: **exact conditional sampling is the Doob h-transform of
the reverse process**, with `h` the conditional likelihood propagated
through the forward noise. The theorem earns its keep three times
over: it explains WHY guidance is a drift correction and not an
approximation, it isolates the exact error of every plug-in guidance
scheme in one formula (consumed by `03`), and it is the mechanism
`04` runs reward fine-tuning through.

## The Object

Condition on `y` (label, measurement, event). Define

```math
h_t(x) \;:=\; p_t(y\,|\,x_t = x) \;=\; \mathbb{E}\big[\,p(y|x_0)\ \big|\ x_t = x\,\big]
```

— the data-level likelihood, averaged over the denoising posterior
(insert-the-kernel-and-divide, as always). Two structural facts,
proved:

**Lemma (h is a reverse-time martingale).** Under the UNCONDITIONAL
joint law, for `t' < t`:
`\mathbb{E}[h_{t'}(x_{t'})\,|\,x_t] = h_t(x_t)`.
*Proof.* Both sides equal `E[p(y|x_0)\,|\,x_t]` by the tower property
and the Markov structure of the (reverse-time) chain
`x_t \to x_{t'} \to x_0`. ∎ (This is "space–time harmonic for the
reverse process," the defining property of an h-transform function.)

**Lemma (the h-transform adds `g^2\nabla\log h` to the drift).** Let
`L` be the generator of a diffusion `dx = b\,dt + g\,dW`. The
h-transformed process — transition kernels reweighted by
`h(\text{new})/h(\text{old})` — has generator

```math
L^h\varphi \;=\; \frac{1}{h}\big[L(h\varphi) - \varphi\,Lh\big]
\;=\; L\varphi + g^2\,\nabla\log h\cdot\nabla\varphi ,
```

i.e. the same diffusion with drift `b + g^2\nabla\log h`.
*Proof.* Expand `L(h\varphi)` by the product rule:
`hL\varphi + \varphi Lh + g^2\nabla h\cdot\nabla\varphi`; divide by
`h`. (The martingale property of `h` is what makes the reweighted
kernels integrate to one.) ∎

## The Theorem

**Theorem (conditioning = h-transform).** The reverse process of the
CONDITIONAL data distribution `p_0(\cdot|y)` — i.e. the exact sampler
for conditional generation — is the h-transform of the unconditional
reverse process by `h_t`: same diffusion coefficient, drift

```math
\tilde b_t(x) \;=\; \underbrace{-f + g^2 s_t(x)}_{\text{unconditional reverse}} \;+\; g^2\,\nabla\log h_t(x),
```

initialized from `p_T(\cdot|y) \propto p_T(\cdot)\,h_T(\cdot)`.

*Proof.* Two routes agree, which is the point. (Marginals route)
Bayes at each level, `p_t(x|y) = p_t(x)h_t(x)/p(y)`, gives
`\nabla\log p_t(x|y) = s_t + \nabla\log h_t`; Anderson's theorem
(`score_foundations/03`) applied to the conditional path yields
exactly the displayed drift. (Process route) The h-transform lemmas:
reweighting the reverse process's path measure by the terminal-ish
functional `p(y|x_0)` is, by the martingale lemma, the h-transform
with `h_t`, whose generator adds `g^2\nabla\log h` — and its
time-`t` marginals are `p_t(x)h_t(x)/p(y) = p_t(x|y)` by
construction. ∎

Classifier guidance (`score_foundations/05`) is this theorem
implemented with a learned `h`; "guidance is exact in principle" is
the statement that the CLASS of drift-corrected samplers contains the
exact conditional sampler — the question is only ever which `h` got
plugged in.

## The Plug-In Error, Identified Exactly

Practice cannot compute `h_t = E[p(y|x_0)|x_t]` and substitutes
plug-ins; the generic one evaluates the clean-data likelihood at the
denoised point:
`\hat h_t(x) = p\big(y\,\big|\,\hat x_0(x, t)\big)` (DPS-style,
`03`).

**Proposition.** The plug-in sampler's drift error is exactly

```math
g^2\,\nabla\log\frac{h_t(x)}{\hat h_t(x)}
\;=\;
g^2\,\nabla\log\ \frac{\mathbb{E}\big[p(y|x_0)\,\big|\,x_t=x\big]}{p\big(y\,\big|\,\mathbb{E}[x_0|x_t=x]\big)} :
```

an EXPECTATION-versus-PLUG-IN ratio — zero iff `p(y|\cdot)` is
effectively affine over the denoising posterior, small when the
posterior is concentrated (late `t`), largest at high noise where the
posterior is diffuse. *Proof.* Subtract the theorem's drift from the
plug-in's. ∎ The same Jensen-type structure as
`score_foundations/05`'s CFG theorem — and that is not a coincidence:
every guidance approximation in this family errs by moving an
expectation through a nonlinearity, and the noise level controls how
much that costs. `03` computes the gap in closed form for linear-
Gaussian measurements; `02` uses its geometry to explain when CFG
should be gated.

## Load-Bearing Audit

```text
Markov structure          the martingale lemma — conditioning on
                          terminal-sigma-field events is what h-
                          transforms do; conditions on the PATH
                          (e.g. trajectory constraints) need the
                          general version (statement);
h > 0                     division everywhere; measure-zero
                          conditioning (exact observations) makes h a
                          delta-limit — 03's sigma_y -> 0 fence;
same diffusion            the transform corrects drift only; that
                          guidance never needs to touch g is a
                          THEOREM here, not a design choice;
learned h                 the theorem is exact; every gap in practice
                          is the proposition's ratio, not the
                          framework.
```

## Position In The Coordinate System

Conditioning enters as a modification of the estimand `s` — and this
file proves the modification has canonical form: add the gradient of
a reverse-martingale `h`. The `\lambda`-dial, the solvers, the phase-A
error theory all apply verbatim to the corrected drift. `04` reveals
the same structure with `p(y|x_0)` replaced by `e^{r(x_0)/\beta}`:
conditioning and reward-tilting are one transform.

## What Remains Open

Learning `h_t` directly (noise-conditional classifiers/likelihoods)
versus plug-in corrections: the proposition prices the plug-in's
error but no result compares it to the estimation error of learned
`h_t` at equal compute — the practically decisive comparison;
h-transforms for path-level conditioning (inpainting across time,
trajectory constraints in video/control models) beyond the terminal
case; and the compounding of the plug-in error with the solver
dynamics — `samplers_and_convergence/05` says the SDE partially heals
drift error and the ODE transports it, which predicts
guidance-solver interactions that are observed and untheorized.
