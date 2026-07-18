# Stochastic Interpolants: The General Family

## The Question

`01` proved the two theorems for an arbitrary interpolant; this file
classifies the design space. The stochastic-interpolant framework
(Albergo–Boffi–Vanden-Eijnden) adds one ingredient — an independent
Gaussian latent — and with it, everything in this repository becomes a
special case of a single three-function family: diffusion, flow
matching, and bridges are points in `(\beta_t, \gamma_t, \Pi)`-space.
The file proves the two structural theorems the generality rests on:
the latent gives every interior marginal a score (with an exact
formula — the insert-the-kernel manipulation's third appearance), and
score plus velocity together reconstruct the full `\lambda`-family of
samplers on ANY interpolant path.

## The Family

```math
x_t \;=\; a_t\,x_0 + \beta_t\,x_1 + \gamma_t\,z,
\qquad
(x_0, x_1)\sim\Pi,\quad z\sim N(0,I)\ \text{independent},
```

with boundary conditions making `x` equal the source at `t=0` and the
data at `t=1` (`a_0 = 1, \beta_0 = \gamma_0 = 0`;
`a_1 = \gamma_1 = 0, \beta_1 = 1`). Instances:

```text
flow matching (01)      gamma = 0, linear a, beta
diffusion (VP), 04's    a = 0 (source absorbed into the latent),
  dictionary               beta = alpha-bar, gamma = sigma —
                           score_foundations/01 in FM time
Brownian-bridge-like    gamma_t ~ sqrt(t(1-t)): the two-sided
                           pinned-noise interpolants
one-sided interpolants  gamma vanishing at one endpoint only
```

(For Gaussian `p_0` and independent coupling, the `a_t x_0` term and
the `\gamma_t z` term are interchangeable — a Gaussian source IS a
latent — which is why plain flow matching from noise and VP diffusion
are the same subfamily in different coordinates; `04` completes that
dictionary.)

## Theorem 1: The Latent Buys A Score

**Theorem.** Wherever `\gamma_t > 0`, the marginal `p_t` has a
density and a well-defined score, given exactly by

```math
\nabla\log p_t(x) \;=\; -\,\frac{\mathbb{E}\big[z\,\big|\,x_t = x\big]}{\gamma_t}.
```

*Proof.* Condition on the pair `(x_0, x_1)`: `x_t` is Gaussian with
mean `m = a_t x_0 + \beta_t x_1` and covariance `\gamma_t^2 I`, so
`p_t(x) = E_{\Pi}\,N(x; m, \gamma_t^2 I)` — a Gaussian mixture, hence
smooth and positive. Differentiate under the expectation:

```math
\nabla\log p_t(x)
= \frac{\mathbb{E}\big[\tfrac{m - x}{\gamma_t^2}\,N(x; m, \gamma_t^2 I)\big]}{p_t(x)}
= \mathbb{E}\Big[\frac{m - x}{\gamma_t^2}\,\Big|\,x_t = x\Big]
= -\frac{\mathbb{E}[z\,|\,x_t = x]}{\gamma_t},
```

since `(x - m)/\gamma_t = z` and the ratio of kernel times prior to
marginal is the posterior — Tweedie's manipulation
(`score_foundations/02`), verbatim, third framework. ∎

And the score is TRAINABLE the same way everything else is: `z` is
known at simulation time, so `E\|s_\theta(x_t) + z/\gamma_t\|^2` is a
CFM-type objective whose projection (Vincent, again) recovers the
displayed conditional expectation. One latent, one denominator, and
the interpolant family carries both estimands at once.

## Theorem 2: Every Interpolant Has A `\lambda`-Dial

**Theorem.** Let `v^*` be the marginal velocity (`01`) and `s` the
score (Theorem 1) of an interpolant path `\{p_t\}`. For every
`\epsilon_t \ge 0`, the SDE

```math
\mathrm{d}x \;=\; \big[v^*(x,t) + \epsilon_t\, s(x,t)\big]\,\mathrm{d}t + \sqrt{2\epsilon_t}\;\mathrm{d}W_t
```

has marginals `p_t` — one exact sampler per noise schedule
`\epsilon`, with `\epsilon = 0` the `01` flow.

*Proof.* Fokker–Planck of the displayed SDE:
`\partial_t p = -\nabla\cdot(v^* p) - \epsilon\,\nabla\cdot(s\,p) +
\epsilon\,\Delta p`, and `s\,p = \nabla p` (the score identity, fifth
theorem in this repo to run on it), so the last two terms cancel,
leaving `01`'s continuity equation, which `p_t` satisfies. ∎

This is `score_foundations/03`'s `\lambda`-family theorem, re-proved
in one line for arbitrary paths — and with it, the ENTIRE solver
theory of phase A transfers: exponential integrators where the
interpolant is affine, the Girsanov decomposition against the
`\epsilon > 0` members, the ODE-transports/SDE-contracts dichotomy
(`samplers_and_convergence/05`) as the design guidance for choosing
`\epsilon`. Path, estimand, solver: fully decoupled, which is the
coordinate system of this repository earning its keep.

## The Classification, Stated Plainly

What the two theorems jointly say about the design space:

```text
gamma > 0 in the interior   both estimands exist, all samplers
                            available, phase-A theory applies: the
                            SAFE region;
gamma = 0 (pure FM)         velocity exists wherever the flow is
                            well-posed; the score may not (atomic
                            data: no density) — deterministic-only
                            sampling, and 01's well-posedness audit
                            is the binding constraint;
endpoint behavior           gamma must vanish at endpoints to hit
                            the prescribed laws; the RATE at which it
                            vanishes controls the score blowup
                            (gamma^{-1} in Theorem 1) — the t_min
                            story of the foundations, now a free
                            design parameter rather than a fixed
                            liability.
```

## Load-Bearing Audit

```text
independence of z           Theorem 1's mixture structure and the
                            posterior identification; correlated
                            latents break the formula (and the
                            trainability);
gamma > 0                   the density; the FM row of the
                            classification is exactly its failure;
score identity s p = grad p Theorem 2, as everywhere;
boundary conditions         what makes the family a NOISE-to-DATA
                            path at all; interpolants missing them
                            solve a different (bridge) problem —
                            fine, but a different contract.
```

## Position In The Coordinate System

The path coordinate `P` in its final generality: three functions and
a coupling, with theorems saying exactly which members carry which
estimands and which solvers. The repository's earlier objects are
relocated rather than replaced: VP diffusion is the
`a = 0` subfamily, FM the `\gamma = 0` subfamily, and the foundations'
`\lambda`-dial is Theorem 2's `\epsilon`.

## What Remains Open

Design selection inside the safe region: `\gamma`'s interior profile
trades score-target conditioning against velocity-target variance —
the Gaussian caricature is now CLOSED (`statistical_theory/07`: the
trigonometric path uniquely minimizes the CFM floor among VP
interpolants, and the endpoint rates obey a strict dichotomy at
`\gamma \sim \sqrt{1-t}`, the bridge rate); the optimum for general
data classes stays open (the family's version of the schedule
question, third appearance); couplings beyond
independence interact with everything here (`03`, `05`); and the
one-sided interpolants — noise only at the data end, where it
regularizes exactly the `t_{\min}` singularity — lack the sharp
statement of what they buy in phase-F terms (score estimation on
manifold data), which is where the design freedom would actually pay.
