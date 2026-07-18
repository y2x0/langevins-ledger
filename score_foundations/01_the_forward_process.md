# The Forward Process

## The Object

Everything in this repository starts from one stochastic differential
equation, the variance-preserving Ornstein–Uhlenbeck process:

```math
\mathrm{d}x = -\tfrac{\beta_t}{2}\,x\,\mathrm{d}t + \sqrt{\beta_t}\,\mathrm{d}W_t,
\qquad x_0 \sim p_0 .
```

The design intent: destroy the data measure `p_0` into a known prior,
smoothly, with closed-form transitions. This file proves the three
facts that make the destruction usable — the Gaussian transition
kernel, the exponential convergence to the prior, and the exactness of
the discrete-time DDPM chain as its discretization.

## The Transition Kernel, Proved

**Theorem.** Conditional on `x_0`,

```math
x_t \;=\; \alpha_t\,x_0 + \sigma_t\,\varepsilon,
\qquad
\alpha_t = e^{-\frac12\int_0^t\beta_s\,ds},
\quad
\sigma_t^2 = 1-\alpha_t^2,
\quad \varepsilon\sim N(0,I).
```

*Proof.* Integrating factor. Let `A_t = e^{\frac12\int_0^t\beta}` (so
`\alpha_t = 1/A_t`). Itô on `A_t x_t`:
`d(A_t x_t) = A_t\,dx + \tfrac{\beta_t}{2}A_t x\,dt
= A_t\sqrt{\beta_t}\,dW` — the drift cancels exactly. Integrate:
`x_t = \alpha_t x_0 + \alpha_t\int_0^t A_s\sqrt{\beta_s}\,dW_s`. The
stochastic integral is Gaussian, mean zero, with variance (Itô
isometry)

```math
\alpha_t^2\int_0^t A_s^2\,\beta_s\,ds
= \alpha_t^2\Big[e^{\int_0^s\beta}\Big]_0^t
= \alpha_t^2\big(A_t^2 - 1\big)
= 1-\alpha_t^2 . \qquad\blacksquare
```

So the path coordinate `P` is fully explicit:
`p_t = (\text{scale by }\alpha_t \text{ then convolve with }
N(0,\sigma_t^2 I))\,\#\,p_0` — data shrunk toward the origin and
blurred, with `\alpha_t^2 + \sigma_t^2 = 1` (the "variance preserving"
in the name: a unit-variance input stays unit-variance forever). The
variance-exploding convention drops the drift and lets `\sigma_t`
grow; the two are related by the deterministic rescaling `x \mapsto
x/\alpha_t`, and nothing in this repository depends on the choice
beyond bookkeeping (`04` proves the discrete dictionary).

## Convergence To The Prior, Proved

**Theorem.** For any two initial laws `p_0, q_0` with finite second
moment,

```math
W_2\big(p_t,\ q_t\big) \;\le\; \alpha_t\;W_2\big(p_0,\ q_0\big),
```

and in particular, taking `q_0 = \gamma = N(0, I)` (which is
stationary: check `m=0, v=1` in the moment ODEs),

```math
W_2\big(p_t,\ \gamma\big) \;\le\; \alpha_t\;W_2\big(p_0,\ \gamma\big)
\;\xrightarrow[t\to\infty]{}\; 0
\quad\text{exponentially in } \textstyle\int_0^t\beta .
```

*Proof.* Synchronous coupling: run two copies of the SDE with the SAME
Brownian motion, started from the `W_2`-optimal coupling of
`(p_0, q_0)`. Their difference `\delta_t = x_t - y_t` obeys the
noise-free ODE `\dot\delta = -\tfrac{\beta_t}{2}\delta`, so
`\|\delta_t\| = \alpha_t\|\delta_0\|` pathwise. A coupling's cost
upper-bounds `W_2`:
`W_2(p_t,q_t)^2 \le E\|\delta_t\|^2 = \alpha_t^2\,W_2(p_0,q_0)^2`. ∎

Two readings worth recording. First, the theorem is the entire
justification for initializing the sampler at the prior: the "prior
mismatch" term in every convergence bound (`samplers_and_convergence/
03`) is this `\alpha_T`, and choosing the horizon is choosing how many
digits of it to pay. Second, the contraction is UNIFORM over data —
no log-concavity, no smoothness: forgetting is a property of the
forward process alone, which is exactly why the hard direction
(reversal, `03`) carries all the conditions.

## DDPM Is The Exact Discretization, Proved

**Proposition.** The DDPM chain
`x_k = \sqrt{1-\beta^{(k)}}\,x_{k-1} + \sqrt{\beta^{(k)}}\,\varepsilon_k`
(independent Gaussians) satisfies

```math
x_k = \sqrt{\bar\alpha_k}\,x_0 + \sqrt{1-\bar\alpha_k}\,\varepsilon,
\qquad
\bar\alpha_k = \prod_{j\le k}\big(1-\beta^{(j)}\big),
```

i.e. it has EXACTLY the transition kernel of the theorem with
`\alpha^2 = \bar\alpha_k` — the discrete chain is not an approximation
of the SDE's marginals; it reproduces them at its grid points.

*Proof.* Induction: a Gaussian scaled and added to an independent
Gaussian is Gaussian; the mean coefficient multiplies to
`\sqrt{\bar\alpha_k}`; the variance obeys
`v_k = (1-\beta^{(k)})v_{k-1} + \beta^{(k)}`, and `v = 1-\bar\alpha`
satisfies it: `(1-\beta)(1-\bar\alpha_{k-1}) + \beta =
1-\bar\alpha_k`. ∎

The practical corollary, stated plainly: the FORWARD side of diffusion
contains no approximation anywhere. Every error in the entire
enterprise — and phases A and F price them — lives in the reverse
direction: estimating the score and integrating with it.

## Load-Bearing Audit

```text
linear drift          the integrating factor and the synchronous
                      coupling both need drift linear in x; general
                      forward processes lose closed-form kernels (and
                      with them, cheap training targets — 02);
Ito isometry          the variance computation;
same-noise coupling   the contraction proof; it also shows the bound
                      is about the DETERMINISTIC part of the flow —
                      noise cancels in the difference;
finite second moment  W2 finiteness; heavy-tailed data weakens the
                      statement to weaker metrics.
```

## Position In The Coordinate System

This file fixes `P`: the path is scale-and-blur with an explicit
schedule, the prior is reached exponentially, and the discrete and
continuous descriptions agree exactly. `02` fixes the estimand on this
path; `03` earns the right to run it backward.

## What Remains Open

Nothing about the theorems — the forward direction is closed
mathematics. The open questions are design questions living one level
up: which schedule `\beta_t` minimizes downstream (reverse-side)
error for a given data class — partial answers via the error
decompositions of phase A and the log-SNR reparameterizations of
`04` — and whether non-OU forward processes (e.g. data-adapted or
anisotropic blurring) buy anything a schedule cannot, where the
honest answer is that closed-form kernels have so far been worth more
than path optimality.
