# Tweedie And The Estimands

## The Question

Diffusion training regresses ONE function of `(x, t)` from data. The
literature calls it four different things — the score, the noise, the
denoiser, the velocity — and the folklore says they are "equivalent."
This file proves the folklore precisely: Tweedie's formula makes the
four estimands affine bijections of each other ALONG THE PATH, and
Vincent's theorem makes each learnable by plain regression with no
access to `p_t` — while the audit shows what is NOT equivalent about
them: the induced loss weightings.

## Tweedie's Formula, Proved

**Theorem.** With `p_t = \int k_t(\cdot|x_0)\,p_0(dx_0)`,
`k_t(x|x_0) = N(x; \alpha_t x_0, \sigma_t^2 I)`:

```math
\mathbb{E}\big[x_0\,\big|\,x_t = x\big]
\;=\;
\frac{x + \sigma_t^2\,\nabla\log p_t(x)}{\alpha_t}.
```

*Proof.* Differentiate the mixture under the integral:

```math
\nabla\log p_t(x)
= \frac{\int \nabla_x k_t(x|x_0)\,p_0(dx_0)}{p_t(x)}
= \frac{\int \frac{\alpha_t x_0 - x}{\sigma_t^2}\,k_t(x|x_0)\,p_0(dx_0)}{p_t(x)}
= \frac{\alpha_t\,\mathbb{E}[x_0|x_t=x] - x}{\sigma_t^2},
```

using that `k_t(x|x_0)p_0(dx_0)/p_t(x)` is exactly the posterior of
`x_0` given `x_t = x`. Rearrange. ∎

The formula is the repository's hinge: it says the score — a property
of the unknown marginal — equals an affine function of a CONDITIONAL
EXPECTATION, and conditional expectations are what squared-loss
regression computes. Everything trainable about diffusion follows.

**The dictionary (corollary, all one-line rearrangements).** Writing
`\hat x_0 = E[x_0|x_t]`, `\hat\varepsilon = E[\varepsilon|x_t]`, and
the velocity `\hat v = \alpha_t\hat\varepsilon - \sigma_t\hat x_0`
(the `v`-prediction convention):

```math
s_t(x) = -\frac{\hat\varepsilon}{\sigma_t},
\qquad
\hat x_0 = \frac{x - \sigma_t\hat\varepsilon}{\alpha_t},
\qquad
\begin{pmatrix}\hat x_0\\ \hat\varepsilon\end{pmatrix}
=
\begin{pmatrix}\alpha_t & \sigma_t\\ \sigma_t & -\alpha_t\end{pmatrix}^{-1}
\!\begin{pmatrix}x\\ -\hat v\end{pmatrix}\ \text{-type rotations,}
```

each pair related by an invertible affine map with
`(t)`-dependent coefficients (from `x = \alpha\hat x_0 +
\sigma\hat\varepsilon` applied inside the posterior — which holds
because conditional expectation is linear). Four names, one function.

## Vincent's Theorem, Proved

The score involves `p_t`, which is unknown. Denoising score matching
says: regress against the CONDITIONAL score instead, which is
explicit.

**Theorem.** For any square-integrable `s_\theta`,

```math
\mathbb{E}_{x_t\sim p_t}\big\|s_\theta(x_t) - \nabla\log p_t(x_t)\big\|^2
=
\mathbb{E}_{x_0,\,x_t}\big\|s_\theta(x_t) - \nabla_{x_t}\log k_t(x_t|x_0)\big\|^2
\;-\; C_t,
```

with `C_t \ge 0` independent of `\theta`. Hence the minimizers
coincide, and the practical loss — with
`\nabla\log k_t(x_t|x_0) = -(x_t - \alpha_t x_0)/\sigma_t^2 =
-\varepsilon/\sigma_t` — is plain noise-prediction regression.

*Proof.* Expand both squares; the `\|s_\theta\|^2` terms agree, so it
suffices to compare the cross terms. Condition on `x_t`:

```math
\mathbb{E}\big[\nabla\log k_t(x_t|x_0)\,\big|\,x_t\big]
= \frac{\int \nabla_x k_t(x_t|x_0)\,p_0(dx_0)}{p_t(x_t)}
= \nabla\log p_t(x_t)
```

— the same computation as Tweedie's proof read once more. So
`E\langle s_\theta, \nabla\log k\rangle = E\langle s_\theta,
\nabla\log p_t\rangle`: the cross terms are EQUAL, and the two
objectives differ by the `\theta`-free constant
`C_t = E\|\nabla\log k\|^2 - E\|\nabla\log p_t\|^2` (nonnegative by
conditional Jensen: the conditional score is the conditional mean of
the per-sample scores, and means have smaller second moment). ∎

Two sentences of appreciation, because this is the field's load-
bearing trick: **the regression target is allowed to be a noisy,
per-sample surrogate, because conditional expectation projects it
onto the right answer.** The identical trick — replace an
intractable marginal quantity by a conditional one and let the
`L^2` geometry average it — reappears as conditional flow matching
(`flow_matching/01`) and discrete score entropy
(`discrete_diffusion/03`). One trick, three families.

## What Is NOT Equivalent: The Weighting Audit

The dictionary is a pointwise bijection of MINIMIZERS. The LOSSES
differ: regressing `\varepsilon` with unit weight equals regressing
the score with weight `\sigma_t^2`, equals regressing `x_0` with
weight `\alpha_t^2/\sigma_t^2`-type factors — and under a finite-
capacity model that cannot achieve the minimizer at all `t`, the
weighting DECIDES where the error goes:

```text
eps-prediction     de-emphasizes t -> 0 score accuracy (weight
                   sigma^2): fine detail lives there;
x0-prediction      de-emphasizes t -> T (weight ~ SNR): global
                   structure lives there;
v-prediction       the balanced rotation, stable at both ends;
ELBO weights       what variational correctness wants (04) — and NOT
                   what sample quality wants, empirically.
```

Under infinite capacity all choices agree; in practice the estimand
IS a weighting prior over noise levels. That is why `s` is a
coordinate of this repository and not a footnote.

## Load-Bearing Audit

```text
Gaussian kernel          Tweedie differentiates it explicitly; on
                         discrete spaces the formula has a different
                         face (discrete_diffusion/03);
L2 loss                  Vincent's projection IS the conditional-
                         expectation property of squared loss; L1 or
                         adversarial losses void the theorem;
square-integrability     both expansions;
finite capacity          the weighting audit's premise — with it, the
                         four estimands stop being interchangeable.
```

## Position In The Coordinate System

This file fixes `s`: one function, four parametrizations, learnable
exactly by regression (Vincent), convertible pointwise (Tweedie),
inequivalent in emphasis (the audit). `03` shows this one function is
also sufficient: it is the only unknown in the time-reversed dynamics.

## What Remains Open

The optimal weighting question — which emphasis over noise levels
minimizes SAMPLE-quality metrics rather than likelihood — has
empirical winners and no theory connecting them to data properties;
and the finite-capacity interaction between estimand choice and
architecture (what `\varepsilon`- vs `v`-networks spend their
capacity on) is measured, not derived.
