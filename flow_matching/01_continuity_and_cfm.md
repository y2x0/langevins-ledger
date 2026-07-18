# The Continuity Equation And Conditional Flow Matching

## The Question

Flow matching frees the path coordinate `P` from the Ornstein–
Uhlenbeck process: pick ANY interpolation between noise and data,
learn the velocity field that transports along it, sample by ODE. Two
theorems make this a method rather than a wish: the marginal velocity
of an interpolant generates its marginal path (the continuity
theorem), and that velocity — defined through an intractable
conditional expectation — is learnable by plain regression against
per-pair targets (the CFM identity, which is Vincent's trick from
`score_foundations/02` in its second appearance). Both proved here.

Convention note, once: this family uses flow-matching time — `t = 0`
noise, `t = 1` data — the REVERSE of the diffusion convention;
`04` maintains the dictionary.

## The Setup

A coupling `(x_0, x_1) \sim \Pi` (independent by default: noise
`\times` data) and a differentiable interpolant
`x_t = I_t(x_0, x_1)` (canonical: `I_t = (1-t)x_0 + t x_1`). The
CONDITIONAL velocity is the time derivative along a pair,
`\dot I_t(x_0, x_1)` (`= x_1 - x_0` in the linear case); the
MARGINAL velocity is its conditional expectation on the current
position:

```math
v^*(x, t) \;=\; \mathbb{E}\big[\,\dot I_t(x_0, x_1)\ \big|\ x_t = x\,\big].
```

## Theorem 1: The Marginal Velocity Generates The Path

**Theorem.** The marginals `p_t = \mathrm{Law}(x_t)` satisfy the
continuity equation

```math
\partial_t p_t + \nabla\cdot\big(p_t\,v^*\big) = 0
```

(weakly): the deterministic flow `\dot z = v^*(z, t)` started from
`z_0 \sim p_0` has law `p_t` at every `t` — provided the flow is
well-posed (see the audit; `03` exhibits the failure when it is not).

*Proof.* Test against smooth compactly supported `\varphi`:

```math
\frac{\mathrm{d}}{\mathrm{d}t}\,\mathbb{E}\,\varphi(x_t)
= \mathbb{E}\big[\nabla\varphi(x_t)\cdot\dot I_t\big]
= \mathbb{E}\big[\nabla\varphi(x_t)\cdot\mathbb{E}[\dot I_t\,|\,x_t]\big]
= \int \nabla\varphi\cdot v^*\ p_t
= -\int \varphi\ \nabla\cdot(v^* p_t),
```

by the chain rule (differentiation under the integral, declared), the
tower property, and integration by parts — which is the weak form of
the displayed equation. The flow statement follows because a
well-posed ODE flow pushes `p_0` forward along the unique solution of
the same continuity equation. ∎

The proof is four steps of bookkeeping, and its one idea is the tower
property: the ONLY thing the marginal law can feel about the
conditional velocities is their conditional mean. Everything finer —
which pair a particle "belongs to" — is invisible to `p_t`, and `03`
is about what that invisibility does to couplings.

## Theorem 2: The CFM Identity

`v^*` involves a conditional expectation under an unknown posterior —
untrainable as written. The repair is verbatim
`score_foundations/02`:

**Theorem.** For any square-integrable `v_\theta`,

```math
\mathbb{E}_{t,\,(x_0,x_1),\,x_t}\big\|v_\theta(x_t, t) - \dot I_t(x_0,x_1)\big\|^2
\;=\;
\mathbb{E}_{t,\,x_t}\big\|v_\theta(x_t,t) - v^*(x_t,t)\big\|^2 \;+\; C,
```

with `C \ge 0` independent of `\theta`: regressing on the PER-PAIR
velocity — trivially computable, `x_1 - x_0` in the linear case —
has the same minimizer and the same gradients as regressing on the
true marginal velocity.

*Proof.* Expand both sides; the `\|v_\theta\|^2` terms agree; the
cross terms agree by the tower property
(`E[\langle v_\theta(x_t), \dot I_t\rangle] = E[\langle v_\theta(x_t),
E[\dot I_t|x_t]\rangle]`); the residue is the conditional variance of
`\dot I_t` given `x_t`, which is `\theta`-free and nonnegative. ∎

This is the same theorem as denoising score matching with
`\nabla\log k_t` replaced by `\dot I_t` — the series' recurring
instrument (conditional-expectation projection under `L^2`) now
carrying its second framework. The practical delta over score
matching is worth stating precisely, because it is why the field
moved: the TARGET `x_1 - x_0` is bounded wherever the data is, while
the score target `-\varepsilon/\sigma_t` blows up at the data end —
FM's regression is well-conditioned at exactly the endpoint where
score regression is singular (`04` proves the corresponding statement
for the fields themselves), and nothing about `p_0` needs to be
Gaussian: any source with samples works.

## Load-Bearing Audit

```text
tower property             both theorems' entire engine;
differentiability of I_t   Theorem 1's chain rule; kinks in t make
                           the path piecewise and the flow must be
                           restarted at corners (fine in practice,
                           stated for honesty);
well-posedness of the flow the passage from continuity equation to
                           ODE flow needs uniqueness (Lipschitz v*,
                           or densities + DiPerna–Lions/Ambrosio-type
                           conditions, cited): for ATOMIC data the
                           field is singular and 03 exhibits genuine
                           failure — this hypothesis is the family's
                           t_min analogue;
L2 loss                    the CFM projection, as in Vincent;
x_t must determine t?      no — t is an input to the network; the
                           regression is joint over (x_t, t).
```

## Position In The Coordinate System

`P` generalized and `s` re-derived: the path is now any interpolant's
marginal family, and the estimand is its velocity, learnable exactly.
`02` classifies which interpolants also carry scores (hence
`\lambda`-dials); `03` studies what iterating the flow does to the
coupling; `04` proves the whole family is affinely related to score
matching on Gaussian paths — the coordinates rotate, the content is
shared.

## What Remains Open

The optimal-path question, now properly posable: among interpolants
with the same endpoints, which minimizes learned-sampler error at a
compute budget — the FM analogue of the schedule question
(`samplers_and_convergence/01`), open in the same way; and the
coupling question `03` opens: CFM is exact for ANY coupling `\Pi`,
so the coupling is a free design variable, and its principled
selection (beyond independence) is where the OT story — and its
fences (`05`) — begins.
