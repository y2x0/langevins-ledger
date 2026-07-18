# The Composite Theorem

## The Question

`statistical_theory/05` listed it as cross-cutting problem 2: one
bound for what a GUIDED, imperfect-score, `\lambda`-noised sampler
outputs — the composition of drift-error profiles
(`01`–`03`), solver dynamics (`samplers_and_convergence/03, 05`), and
estimation error (`statistical_theory/01`), which five files flagged
and none assembled. This file does the assembly. The result is a
single KL bound against the TRUE CONDITIONAL with four additive error
budgets and one explicit solver factor
`C_\lambda = \tfrac{3(1+\lambda^2)^2}{8\lambda^2}` — whose unique
minimum at `\lambda = 1` is a theorem: the Anderson SDE is the
KL-robust point of the dial. Constants are not optimized; the
structure is the contribution.

## The Setup

Fix the condition `y`. The reference process is the EXACT conditional
`\lambda`-sampler (foundations/03 applied to the conditional path,
drift correction from `01`):

```math
P^c_\lambda:\quad
\mathrm{d}x = \Big[-f + \tfrac{1+\lambda^2}{2}g^2\big(s_t + \nabla\log h_t\big)\Big]\mathrm{d}\tau + \lambda g\,\mathrm{d}\bar W,
\qquad x_{T}\sim p_T(\cdot|y),
```

with marginals `p_t(\cdot|y)`. The implemented sampler `Q_\lambda`
runs the same equation with the practice triple substituted: learned
score `\hat s`, plug-in guidance `\hat h`, strength `\omega`:

```math
Q_\lambda:\quad
\mathrm{d}x = \Big[-f + \tfrac{1+\lambda^2}{2}g^2\big(\hat s_t + \omega\,\nabla\log \hat h_t\big)\Big]\mathrm{d}\tau + \lambda g\,\mathrm{d}\bar W,
\qquad x_T\sim\gamma.
```

## The Theorem

**Theorem (composite bound, continuous time, `\lambda > 0`).** Under
Novikov's condition for the drift difference (audited below),

```math
\mathrm{KL}\big(p_{t_{\min}}(\cdot|y)\ \big\|\ q_{t_{\min}}\big)
\;\le\;
\mathrm{KL}\big(p_T(\cdot|y)\,\big\|\,\gamma\big)
\;+\;
C_\lambda \int_{t_{\min}}^{T} g_t^2\ \mathbb{E}_{p_t(\cdot|y)}\Big[
\underbrace{\|s_t-\hat s_t\|^2}_{\text{score}}
+\underbrace{\big\|\nabla\log\tfrac{h_t}{\hat h_t}\big\|^2}_{\text{plug-in}}
+\underbrace{(\omega-1)^2\big\|\nabla\log\hat h_t\big\|^2}_{\text{guidance budget}}
\Big]\mathrm{d}t,
```

```math
C_\lambda \;=\; \frac{3\,(1+\lambda^2)^2}{8\,\lambda^2},
```

plus the discretization term attaching exactly as in
`samplers_and_convergence/03` (freeze the drift at grid points,
add-and-subtract, one more square in the bracket).

*Proof.* Girsanov between the two path measures (common diffusion
coefficient `\lambda g`; the chain-rule + data-processing scaffolding
verbatim from `samplers_and_convergence/03`):

```math
\mathrm{KL}(P^c\|Q)
= \mathrm{KL}\big(p_T(\cdot|y)\|\gamma\big)
+ \frac12\int \frac{\mathbb{E}_{P^c}\|\Delta b_t\|^2}{\lambda^2 g_t^2}\,\mathrm{d}t,
```

with drift difference

```math
\Delta b
= \tfrac{1+\lambda^2}{2}\,g^2\Big[(s - \hat s) + \nabla\log\tfrac{h}{\hat h} - (\omega-1)\,\nabla\log\hat h\Big]
```

(add and subtract `\nabla\log\hat h`, then split `\omega = 1 +
(\omega-1)`). The elementary inequality
`\|a+b+c\|^2 \le 3(\|a\|^2+\|b\|^2+\|c\|^2)` gives the three budgets;
collecting constants:
`\tfrac12\cdot\big(\tfrac{1+\lambda^2}{2}\big)^2 \tfrac{g^4}{\lambda^2 g^2}\cdot 3
= C_\lambda\,g^2`. Marginals under `P^c` are `p_t(\cdot|y)`
(exactness of the conditional reversal, `01`), which is where the
expectations live. Data processing to the endpoint. ∎

**Corollary (the dial's KL-robust point).** `C_\lambda` is uniquely
minimized at `\lambda = 1`, with `C_1 = 3/2`:

```math
\frac{\mathrm{d}}{\mathrm{d}u}\frac{(1+u)^2}{u}
= \frac{(1+u)(u-1)}{u^2}
\qquad (u = \lambda^2)
```

— negative for `u<1`, positive for `u>1`; and `C_\lambda \to \infty`
at BOTH ends. ∎ The two divergences are the two known failure modes,
now in one formula: `\lambda \to 0` is the ODE, where Girsanov's
control degenerates (the transport regime —
`samplers_and_convergence/05`'s no-decay propagator is the same fact
in `W_2` clothing); `\lambda \to \infty` is churn overwhelming the
drift's information. The reverse SDE is not merely "more robust than
the ODE": it is the exact optimum of this bound's solver factor.

## Reading The Budgets

Each term is an earlier file, now integrated:

```text
score        Vincent's objective under the CONDITIONAL path — the
             trained loss, reweighted by conditioning (F/01's
             interface, conditional edition);
plug-in      01's pointwise error ratio, now integrated: the price of
             DPS-style h-hats, computable in closed form for linear-
             Gaussian measurements (03's S_t formula plugs in
             directly);
guidance     NEW, and the composite's dividend: the KL distance from
             the true conditional grows as (omega-1)^2 times the
             classifier's Fisher information along the conditional
             path — the quantitative form of foundations/05's "CFG
             has no target": it HAS a distance from the natural
             target, and this is its rate;
prior        e^{-Theta(T)} as always (foundations/01);
solver       C_lambda, plus the A/03-style discretization square.
```

The `(\omega-1)^2`-budget's order is right: `02`'s exact Gaussian
family gives
`\mathrm{KL}(\text{conditional}\,\|\,\omega\text{-tilt}) =
\tfrac{(\omega-1)^2\mu^2}{2v}`-type expressions — quadratic in
`(\omega-1)`, matching the bound's structure, with the bound's
constant `C_\lambda\int g^2 E\|\nabla\log h\|^2` an integrated Fisher
term dominating the exact answer's single-level constant
(order-check, honestly labeled: constants compared only in structure;
the Gaussian instantiation of the Fisher integral is displayed
arithmetic left to the reader-with-an-afternoon).

The bound also disciplines schedule design in one glance: interval
guidance (`02`) is the choice `\omega_t \equiv 1` outside a band —
which zeroes the guidance budget exactly where
`E\|\nabla\log \hat h\|^2` is large (early, by `02`'s gap profile and
`statistical_theory/04`'s geometry) — and the composite bound turns
that from a heuristic into the minimization of a displayed integral.

## The ODE Complement

At `\lambda = 0` the theorem is void (as it must be — no noise to
reweight), and the `W_2` route of `statistical_theory/01` takes over:
the synchronous-coupling/Grönwall bound applies with the drift error
enlarged by the same three budgets and the Lipschitz budget enlarged
by `\omega\,\mathrm{Lip}(\nabla\log\hat h)` — the guided ODE pays the
guidance budget WITHOUT decay (propagator `\equiv 1`) and with an
exponential Grönwall constant that `\omega` inflates. Assembled
statement, same ingredients; the two routes bracket the dial from its
two ends, and their crossover is the design content of
`samplers_and_convergence/05`, now with guidance included.

## Load-Bearing Audit

```text
lambda > 0                the Girsanov route's jurisdiction; the ODE
                          endpoint is covered only by the (weaker,
                          exponential-constant) W2 complement;
Novikov with guidance     the integrand now includes omega^2 E||grad
                          log h-hat||^2, which blows up near t_min on
                          supported data (statistical_theory/04) —
                          the t_min-times-omega interaction is THE
                          binding regularity constraint, made
                          explicit here for the first time;
errors under p_t(.|y)     the conditional path: trained losses
                          estimate the UNconditional integrand; the
                          reweighting gap is real and inherited from
                          A/03's audit, conditional edition;
constant 3, C_lambda      the triangle split is crude; the STRUCTURE
                          (four additive budgets, one solver factor,
                          lambda = 1 optimal) is the claim — constants
                          are not.
```

## Position In The Coordinate System

The assembly file: `(s)`-errors (score, plug-in, strength) and
`(S)`-choices (`\lambda`, grid) in one inequality against the one
defensible target (the true conditional). Cross-cutting problem 2 of
`statistical_theory/05` is hereby closed at the level it was posed —
"one composed bound, all ingredients in place" — and remains open at
the level it deserves: tight constants, the trainable-norm/strong-
metric mismatch, and the `\lambda`-schedule optimum under the
composite integrand.

## What Remains Open

Optimizing `\lambda_t` AGAINST the composite integrand (the bound
makes it a calculus-of-variations problem with all terms displayed —
solvable in the Gaussian caricature, unattempted); empirical
calibration: whether the four budgets' measured magnitudes rank as
the geometry predicts (plug-in and guidance dominating early, score
late) — a measurement this bound finally makes well-posed; and the
mixed-metric problem inherited from `statistical_theory/01`,
unmoved by the assembly and still the phase's deepest gap.
