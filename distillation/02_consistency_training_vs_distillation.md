# Consistency Training vs Consistency Distillation

## The Question

`01` budgeted the consistency error; this file prices how the two
training schemes SPEND it. Consistency distillation (CD) builds
adjacent-point pairs with a teacher ODE step; consistency training
(CT) builds them teacher-free, from the forward process itself. The
folklore says CT is "biased but self-contained." The theorems say
something sharper: CD's fixed point is exactly the teacher's map with
the solver's order inherited as bias; and CT's target has a
conditional mean EQUAL to the exact-score DDIM step — its bias is not
in the input but a Jensen gap of the network over the posterior's
fluctuation, the fourth appearance of this series' favorite error
mechanism.

## CD: The Fixed Point And The Inherited Order

CD's pair: from `x_{t_{k+1}}` (on the forward path), step the TEACHER
ODE backward one grid step to `\hat x_{t_k}`, and regress

```math
f_\theta\big(x_{t_{k+1}}, t_{k+1}\big) \;\approx\; f_{\theta^-}\big(\hat x_{t_k}, t_k\big).
```

**Theorem.** (i) If the teacher step is exact (the true PF-ODE flow),
then `f = f^*` gives ZERO loss at every pair: the exact map is CD's
fixed point. (ii) If the teacher step has local error `O(h^{p+1})`
(a `p`-th order solver, `samplers_and_convergence/02`) and
`f_{\theta^-}` is `L`-Lipschitz in `x`, the induced per-pair target
error is `\le L\cdot O(h^{p+1})`, and by `01`'s accumulation theorem
the learned map inherits total bias

```math
O\big(N\,h^{p+1}\big) = O\big((T-t_{\min})\,h^{p}\big) :
```

**the student inherits the teacher solver's order as its bias
floor.** *Proof.* (i): with an exact step, `\hat x_{t_k}` lies on the
same trajectory as `x_{t_{k+1}}`, and `f^*` is constant there
(`01`, (SC)). (ii): the target differs from the same-trajectory value
by `f_{\theta^-}` evaluated `O(h^{p+1})` away; Lipschitz; telescope. ∎

So CD with a DDIM teacher (order 0 in the quadrature reading, locally
`O(h^2)` in the smooth regime) has bias `O(h)`; with a second-order
teacher, `O(h^2)` — the practical instruction "distill from a good
solver" is this theorem, and the accumulation is why the constant in
front is the trajectory length, not the step count.

## CT: The Conditional-Mean Identity

CT deletes the teacher: draw ONE pair `(x_0, \varepsilon)`, place
BOTH points on the same forward path,

```math
x_{t_k} = \alpha_{t_k}x_0 + \sigma_{t_k}\varepsilon,
\qquad
x_{t_{k+1}} = \alpha_{t_{k+1}}x_0 + \sigma_{t_{k+1}}\varepsilon,
```

and regress `f_\theta(x_{t_{k+1}}) \approx f_{\theta^-}(x_{t_k})`.
The input to the target is now RANDOM given `x_{t_{k+1}}` — the
shared-`(x_0, \varepsilon)` coupling. What does it average to?

**Theorem (the CT input is centered on the DDIM step).**

```math
\mathbb{E}\big[x_{t_k}\ \big|\ x_{t_{k+1}}\big]
\;=\;
\alpha_{t_k}\,\hat x_0(x_{t_{k+1}}) + \sigma_{t_k}\,\hat\varepsilon(x_{t_{k+1}})
```

— exactly the DDIM update with the TRUE score
(`score_foundations/04`).

*Proof.* Conditional expectation is linear:
`E[x_{t_k}|x_{t_{k+1}}] = \alpha_{t_k}E[x_0|x_{t_{k+1}}] +
\sigma_{t_k}E[\varepsilon|x_{t_{k+1}}]`, and the two conditional
means are the denoiser and noise estimands of
`score_foundations/02`'s dictionary. ∎

CT thus needs no teacher because THE FORWARD COUPLING IS THE TEACHER:
the exact-score DDIM step is the conditional mean of a zero-cost
simulation. The price is the fluctuation around that mean:

**Proposition (the CT bias is a Jensen gap).** The CT target's
conditional expectation differs from the CD-with-exact-DDIM target by

```math
\mathbb{E}\big[f_{\theta^-}(x_{t_k})\,\big|\,x_{t_{k+1}}\big]
- f_{\theta^-}\big(\mathbb{E}[x_{t_k}|x_{t_{k+1}}]\big)
\;=\;
\tfrac12\,\nabla^2 f_{\theta^-}\big[\mathrm{Cov}(x_{t_k}|x_{t_{k+1}})\big] + \text{h.o.t.},
```

and the conditional covariance is `O(h^2)`: writing
`x_{t_k} = x_{t_{k+1}} - h(\dot\alpha x_0 + \dot\sigma\varepsilon) +
O(h^2)`, the randomness given `x_{t_{k+1}}` enters at order `h`, so
`\mathrm{Cov} = h^2\,\mathrm{Cov}\big(\dot\alpha x_0 +
\dot\sigma\varepsilon\,\big|\,x_{t_{k+1}}\big) + O(h^3)`. Per-step
bias `O(h^2)\|\nabla^2 f\|`; accumulated (`01`), total `O(h)` — the
SAME order as CD-with-DDIM, with the curvature-of-the-ODE constant
replaced by curvature-of-the-network times posterior variance. ∎
(second-order Taylor of `f_{\theta^-}` around the conditional mean)

Beyond the bias, the single-sample target has VARIANCE — gradient
noise scaling with the same posterior fluctuation — and here is the
schedule logic (Song–Dhariwal's "improved techniques," now with the
mechanism visible): coarse grids (small `N`, large `h`) have few
accumulation terms and high per-step bias/noise; fine grids the
reverse; and since bias falls as `h` while target noise per step also
changes, the optimal `N` GROWS over training as the network's
curvature `\|\nabla^2 f\|` and the tolerable noise floor drop —
train coarse to fine (their schedule; statement, mechanism above).
Their second fix — replace `f_{\theta^-}` by `f_\theta` without EMA
in the target limit — removes a fixed-point mismatch orthogonal to
this file's accounting (statement).

## The Comparison, Stated Plainly

```text
CD    bias = teacher solver order, inherited exactly (theorem);
      needs the teacher network per pair (compute) but has
      deterministic targets (low gradient noise);
CT    same asymptotic bias order via a different constant
      (network curvature x posterior variance — the Jensen gap);
      teacher-free, but pays target variance and needs the
      coarse-to-fine schedule to manage the tradeoff;
both  live under 01's linear accumulation: N segments, N leaks —
      neither scheme escapes the budget, they allocate it.
```

## Load-Bearing Audit

```text
exact score inside the identity  the conditional-mean theorem uses
                                 the TRUE estimands; in CT practice
                                 there is no score anywhere — the
                                 identity says the coupling SIMULATES
                                 it, which is the honest magic;
smoothness of f_theta-minus      the Jensen expansion; near t_min on
                                 rough data curvature blows up
                                 (01's audit) — CT's bias constant
                                 degrades exactly there;
shared (x0, eps) coupling        the whole CT construction; independent
                                 draws would center the target on the
                                 WRONG point (the posterior mean path,
                                 not the trajectory);
Lipschitz f in CD (ii)           converting input error to target
                                 error.
```

## Position In The Coordinate System

The estimand-acquisition question for the learned solution map: CD
consumes a solver (`S` feeding `s`), CT consumes the forward coupling
(`P` feeding `s`), and the theorems say the two routes deliver the
same object at the same order with different constants and noise.
The family's Jensen-gap instrument now has four sites: CFG
(foundations/05), plug-in guidance (`guidance/01`), DPS
(`guidance/03`), and CT targets (here) — one mechanism, four
methods.

## What Remains Open

End-to-end guarantees with OPTIMIZATION in the loop (both theorems
characterize targets and fixed points; no result bounds what SGD on
either loss converges to); the optimal grid schedule as a theorem
(the mechanism above is qualitative; the improved-techniques schedule
is empirical); and variance-reduced CT (multi-sample or
antithetic-coupled targets would shrink the Jensen gap's fluctuation
at known cost — an obvious untested corollary of the
conditional-mean identity).
