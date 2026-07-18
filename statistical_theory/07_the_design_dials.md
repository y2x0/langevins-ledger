# The Design Dials

## The Question

Cross-cutting problem 6 of `05`: three design dials recur through the
repository without an optimality theorem anywhere — the noise
schedule (priced in `samplers_and_convergence/01`, made
training-irrelevant by `discrete_diffusion/02`, free at sampling
time), the training-loss weighting across noise levels
(`score_foundations/02`'s audit, with `01`'s propagator named as the
missing criterion), and the interpolant's `\gamma_t`
(`flow_matching/02`'s safe region). Each is a one-dimensional
variational problem whose objective this repository has already
computed; this file solves all three in the Gaussian caricature. Two
findings organize it. First, the grid and the weight have the SAME
answer: one measure, `\beta_t e^{-\lambda^2 B(t)/2}\,\mathrm{d}t` —
uniform in `\alpha_t^{\lambda^2}` — optimizes both. Second, the
interpolant dial splits into a solved interior problem (the
trigonometric path) and an endpoint DICHOTOMY: no interpolant
carries both a trainable velocity and a trainable score into the
data.

## The Shared Objects

Throughout, `B(t) := \int_{t_{\min}}^{t}\beta_r\,\mathrm{d}r` (the
schedule integrated from the OUTPUT end), so that
`e^{-B(t)/2} = \alpha_t/\alpha_{t_{\min}}`. Data is `N(0,1)` per
coordinate (`p_t = N(0,1)`, `s = -x`,
`score_foundations/06`), and the sampler is the `\lambda`-family. Two
exact facts, extending `samplers_and_convergence/01` and
`statistical_theory/01`:

**Lemma (the `\lambda`-family EM map).** On Gaussian data the
`\lambda`-family drift is `-\tfrac{\lambda^2\beta}{2}x`, so one EM
step of size `h` maps variances by
`v \mapsto (1-\tfrac{\lambda^2\beta h}{2})^2 v + \lambda^2\beta h`,
and the variance ERROR `e = v - 1` by

```math
e \;\mapsto\; \Big(1-\tfrac{\lambda^2\beta h}{2}\Big)^2 e
\;+\; \frac{\lambda^4\beta^2h^2}{4}.
```

*Proof.* Drift: `\tfrac{\beta}{2}x + \tfrac{1+\lambda^2}{2}\beta s =
\tfrac{\beta}{2}(1-(1+\lambda^2))x`; the map is
`samplers_and_convergence/01`'s computation with `\beta \to
\lambda^2\beta`; substitute `v = 1 + e` and use that `v = 1` is the
exact flow's fixed point at every `t`. ∎

So variance errors contract at rate `e^{-\lambda^2 B}` over the
remaining schedule, while MEAN errors contract at
`e^{-\lambda^2 B/2}` (`statistical_theory/01`'s propagator) — mean
multiplier versus its square. Both dials below reduce to weighting an
injection `\propto \beta` against one of these propagators, and both
optimizations land on the same object, the **design measure**

```math
\mathrm{d}\mu_\lambda \;:=\; \beta_t\,e^{-\lambda^2 B(t)/2}\,\mathrm{d}t
\;\propto\; \mathrm{d}\big(\alpha_t^{\lambda^2}\big),
```

uniform in `\alpha^{\lambda^2}`: at `\lambda = 1`, equal steps of the
signal coefficient `\alpha`; at `\lambda = 0`, the flat `\beta`-clock
`\mathrm{d}B`.

## Dial 1: The Sampling Grid

`K` steps from `T` down to `t_{\min}`; step-size profile `h(t)`;
exact score; output error = the accumulated variance bias. By the
Lemma, to leading order in `\max h`,

```math
e_K \;=\; \frac{\lambda^4}{4}\int_{t_{\min}}^{T}
\beta_t^2\,h(t)\,e^{-\lambda^2 B(t)}\,\mathrm{d}t,
\qquad
K \;=\; \int_{t_{\min}}^{T}\frac{\mathrm{d}t}{h(t)} .
```

**Theorem (the optimal grid).** Over all step profiles with `K`
steps,

```math
e_K \;\ge\; \frac{1}{K}\Big(1 - \big(\alpha_T/\alpha_{t_{\min}}\big)^{\lambda^2}\Big)^{2},
```

with equality iff the grid density is `\propto \mathrm{d}\mu_\lambda`
— uniform in `\alpha^{\lambda^2}`. The optimal error is
SCHEDULE-INVARIANT: it depends on `\beta` only through the total
`B(T)`.

*Proof.* Write `\varphi := \tfrac{\lambda^4}{4}\beta^2 e^{-\lambda^2 B}`,
so `e_K = \int\varphi h`. Cauchy–Schwarz:

```math
\Big(\int\sqrt{\varphi}\Big)^{2}
= \Big(\int \sqrt{\varphi h}\cdot h^{-1/2}\Big)^{2}
\le \Big(\int \varphi h\Big)\Big(\int h^{-1}\Big)
= e_K\,K,
```

with equality iff `\varphi h^2` is constant, i.e. density
`1/h \propto \sqrt{\varphi} = \tfrac{\lambda^2}{2}\beta
e^{-\lambda^2 B/2}`. Evaluate: `\int\sqrt{\varphi}\,\mathrm{d}t =
\tfrac{\lambda^2}{2}\int\beta e^{-\lambda^2B/2}\mathrm{d}t =
1 - e^{-\lambda^2 B(T)/2}` (substitute `\mathrm{d}B`), and
`e^{-B(T)/2} = \alpha_T/\alpha_{t_{\min}}`. ∎

Readings. (i) The rule quantifies `samplers_and_convergence/01`'s
qualitative one: spend steps where `\beta` is large AND the output
still remembers. (ii) Schedule invariance is the sampling-side cousin
of `discrete_diffusion/02`'s `du/u` theorem: with the grid chosen
optimally, the schedule's shape is a reparametrization, and only
total integrated noise matters. In KL, `d` coordinates cost
`\approx d\,e_K^2/4` — monotone in `e_K`, same optimizer. (iii) At
`\lambda = 0` the caricature degenerates (the Gaussian PF-ODE drift
vanishes; EM is exact): the grid problem is a stochastic-sampler
problem here, and the honest general-`\lambda` statement is the
displayed one.

## Dial 2: The Training Weight

Training certifies exactly one thing: the value of the WEIGHTED score
loss (Vincent, `score_foundations/02`). Model the design problem as
the game that certificate defines. The trainer picks a weight profile
`w \ge 0`, `\int_{t_{\min}}^{T} w\,\mathrm{d}t = 1` (overall scale is
a learning rate, not a design freedom); nature picks the score-bias
profile `\delta_t` subject to the certificate
`\int w_t\,\delta_t^2\,\mathrm{d}t \le \varepsilon^2`; the payoff is
the output mean error, which for `x`-independent bias is EXACTLY
(`statistical_theory/01`'s propagator)

```math
|m| \;=\; \Big|\int_{t_{\min}}^{T} A(t)\,\delta_t\,\mathrm{d}t\Big|,
\qquad
A(t) \;=\; \frac{1+\lambda^2}{2}\,\beta_t\,e^{-\lambda^2 B(t)/2}
\;\propto\; \frac{\mathrm{d}\mu_\lambda}{\mathrm{d}t}.
```

**Theorem (the minimax weight).** The game has the saddle point

```math
\min_{w}\ \max_{\delta}\ |m|
\;=\; \varepsilon\int_{t_{\min}}^{T} A(t)\,\mathrm{d}t,
\qquad
w^*(t) \;=\; \frac{A(t)}{\int A}
\;\propto\; \mathrm{d}\mu_\lambda,
```

and at `w^*` the worst case is the EQUALIZED profile
`\delta^* \equiv \varepsilon`: the optimal weight is the one that
makes nature indifferent to where it puts the error.

*Proof.* Inner maximum: `\int A\delta = \int (A/\sqrt{w})
(\sqrt{w}\delta) \le (\int A^2/w)^{1/2}(\int w\delta^2)^{1/2}`, tight
at `\delta \propto A/w`. Outer minimum: `\int A = \int (A/\sqrt w)
\sqrt w \le (\int A^2/w)^{1/2}(\int w)^{1/2}`, so `\int A^2/w \ge
(\int A)^2`, tight at `w \propto A`; at `w^*`, `A/w^*` is constant,
so nature's maximizer is flat and the certificate forces
`\delta^* \equiv \varepsilon`. ∎

This is `statistical_theory/01`'s closing conjecture ("reweight by
the propagator's profile per intended sampler"), now a theorem — and
`w^* \propto \mathrm{d}\mu_\lambda` is the SAME measure as dial 1's
grid. The coincidence is square-root bookkeeping: the grid weights a
squared injection against the squared (variance) propagator and takes
a square root through Cauchy–Schwarz; the weight uses the injection
against the mean propagator directly. One measure, two optimality
problems. Against `score_foundations/02`'s menu (in score units, VP
clock):

```text
eps-prediction        score-weight sigma_t^2: VANISHES at low noise —
                      exactly where the lambda = 1 propagator forgives
                      nothing; in this caricature its worst-case
                      output error diverges (verification/);
w* (lambda = 1)       score-weight beta alpha_t: finite at the data
                      end, exponentially forgiving at high noise;
w* (lambda = 0)       flat in B: the ODE transports every level's
                      error intact (F/01), so every level weighs the
                      same;
ELBO (foundations/04) likelihood's choice — a different objective,
                      honestly incomparable.
```

## Dial 3: The Interpolant's `\gamma`

One-sided Gaussian family (`flow_matching/02` with the source
absorbed into the latent; the interpolant's signal coefficient is
written `b_t` here to keep `\beta` for the schedule):

```math
x_t = b_t\,x_1 + \gamma_t\,z,\qquad x_1 \sim N(0,1),\ z\sim N(0,I)\
\text{indep.},\qquad
b_0 = 0,\ \gamma_0 = 1,\ b_1 = 1,\ \gamma_1 = 0,
```

`\rho_t^2 := b_t^2 + \gamma_t^2`, polar coordinates
`b = \rho\cos\theta`, `\gamma = \rho\sin\theta`
(`\theta(0) = \pi/2`, `\theta(1) = 0`).

**Theorem (the floor identity).** The CFM target
`v_{\mathrm{c}} = \dot b\,x_1 + \dot\gamma\,z` satisfies, per
coordinate,

```math
\mathbb{E}\big[v_{\mathrm{c}}\,\big|\,x_t\big] = \frac{\dot\rho}{\rho}\,x_t = v^*(x_t),
\qquad
\mathrm{Var}\big(v_{\mathrm{c}}\,\big|\,x_t\big)
= \frac{(\dot b\gamma - b\dot\gamma)^2}{\rho^2}
= \rho^2\dot\theta^2 ,
```

so the CFM loss at its own minimizer — the FLOOR no network can
remove, and the scale of its gradient noise — is the angular energy
`\int_0^1 \rho^2\dot\theta^2\,\mathrm{d}t`. Radial motion of
`(b, \gamma)` is learnable signal; angular motion is irreducible
target noise.

*Proof.* Rotate: `u := x_t/\rho` and `w := (\gamma x_1 - b z)/\rho`
are independent standard normals, with `x_1 = (b u + \gamma w)/\rho`,
`z = (\gamma u - b w)/\rho`. Substitute:
`v_{\mathrm{c}} = \dot\rho\,u + \rho^{-1}(\dot b\gamma - b\dot\gamma)\,w`
(the `u`-coefficient is `(b\dot b + \gamma\dot\gamma)/\rho =
\dot\rho`). Conditioning on `x_t` is conditioning on `u`; the
Wronskian in polar form is `\dot b\gamma - b\dot\gamma =
-\rho^2\dot\theta`. Sanity: `\dot b^2 + \dot\gamma^2 = \dot\rho^2 +
\rho^2\dot\theta^2` is the law of total variance, algebraically. ∎

**Theorem (the VP optimum).** Among variance-preserving interpolants
(`\rho \equiv 1`), the floor `\int_0^1\dot\theta^2` is uniquely
minimized (in `H^1`) by constant angular speed —

```math
\theta(t) = \frac{\pi}{2}(1-t):
\qquad
b_t = \sin\frac{\pi t}{2},\quad \gamma_t = \cos\frac{\pi t}{2},
\qquad
\text{floor} = \frac{\pi^2}{4}\ \text{per coordinate}
```

— the trigonometric path (the cosine schedule, in diffusion time).

*Proof.* `(\pi/2)^2 = (\int_0^1\dot\theta\,\mathrm{d}t)^2 \le
\int_0^1\dot\theta^2\,\mathrm{d}t` by Cauchy–Schwarz, with equality
iff `\dot\theta` is a.e. constant; the endpoints fix the constant. ∎

**Theorem (the endpoint dichotomy).** Suppose
`\gamma_t \sim c\,(1-t)^q` at the data end (exact power profile,
`b, \rho \to 1`). The score target `-z/\gamma` has conditional
variance `b^2/(\gamma^2\rho^2)`, and:

```text
velocity floor  int rho^2 thetadot^2 dt  finite  iff  q > 1/2
score target    int b^2/(gamma^2 rho^2)  finite  iff  q < 1/2
q = 1/2 (the Brownian-bridge rate):      BOTH log-divergent
```

No `\gamma` makes both finite: an interpolant chooses which estimand
it carries into the data.

*Proof.* Near `t = 1`, `\theta \sim \gamma/\rho \sim c(1-t)^q`, so
`\dot\theta^2 \sim c^2q^2(1-t)^{2q-2}`, integrable iff `2q-2 > -1`;
the score-target variance `\sim c^{-2}(1-t)^{-2q}`, integrable iff
`2q < 1`. For the bridge, exactly: `b = t`,
`\gamma^2 = t(1-t)`, `\rho^2 = t` give
`b^2/(\gamma^2\rho^2) = 1/(1-t)`. ∎

Readings. (i) The interior optimum is maximally score-blind: the
trigonometric path has `q = 1`, and its score target's variance
diverges like `(1-t)^{-2}` — the CFM-optimal path is a
velocity-only design. (ii) This sharpens `flow_matching/02`'s
classification: the endpoint rate is indeed a free design parameter,
but it obeys a conservation law, and the bridge rate
`\sqrt{t(1-t)}` is not a convention — it is the unique shared
critical point. (iii) The dichotomy is target-side and present
already for Gaussian data; `04`'s `\sigma^{-2}` blowup is data-side.
The two compound.

## The Three Clocks

```text
dial              objective                 optimum
sampling grid     variance bias, K steps    uniform in alpha^{lambda^2};
                                            error (1-(alpha_T/alpha_tmin)
                                            ^{lambda^2})^2 / K
training weight   minimax output error      w* = the SAME measure
                                            d mu_lambda
interpolant       the CFM floor             uniform in theta (VP: the
                                            trigonometric path); the
                                            q = 1/2 dichotomy at the
                                            data end
```

Dials 1 and 2 share one clock because they weight the same propagated
injection; dial 3's objective is training noise — a different
quantity, and honestly a different clock. There is no universal
design measure; there are exactly two.

## Load-Bearing Audit

```text
Gaussian data            every closed form; beyond it the grid
                         functional becomes the linearized flow's
                         fundamental matrix and the floor becomes
                         data-dependent (F/01's statement) —
                         statements, not theorems;
leading order in h       dial 1 optimizes the O(h) functional; the
                         discrete map shifts the optimum at O(h^2)
                         (checked numerically at K = 500,
                         verification/);
EM, lambda-family        dial 1 is solver-specific: exponential
                         integrators (A/02) change the injection phi —
                         same variational problem, different measure;
the minimax model        dial 2's caricature: nature is adversarial
                         within the trained certificate; real
                         estimators are not — but any other model
                         needs a delta(w) response theory nothing in
                         this repository proves;
x-independent bias       dial 2's nature lives in F/01's exact
                         propagator class; richer perturbations need
                         the fundamental matrix;
unit total weight        the normalization int w = 1 — scale is a
                         learning rate; without SOME normalization the
                         game is void;
rho = 1 (dial 3)         rho free makes the floor degenerate (interior
                         collapse to rho = 0); VP is the natural
                         constraint, not the only one;
exact power endpoints    the dichotomy as proved; regularly varying
                         gamma adds slowly varying factors that cannot
                         cross the q = 1/2 threshold.
```

## Position In The Coordinate System

Problem 6 of `05`, closed at the level posed: each coordinate's free
dial — `\mathcal{S}`'s grid, `s`'s weighting, `\mathcal{P}`'s
interior profile — now has an optimality theorem in the caricature
where its objective is exact. The organizing surprise is economy: the
first two dials collapse into one measure
(`\mathrm{d}\mu_\lambda`, the propagated injection), and the third
is governed by one angle, whose endpoint behavior is a strict
dichotomy rather than a tunable trade.

## What Remains Open

Everything beyond the caricature, and specifically: the
data-dependent design measure (plug a MEASURED error or curvature
profile into the same two Cauchy–Schwarz arguments — the empirical
execution both dials license); the optimal grid for exponential
integrators, where the injection involves denoiser curvature in the
`\rho`-clock (`distillation/04`'s governing quantity) rather than
`\beta^2`; the joint problem (grid, weight, and path interact at
second order; nothing here prices the interaction); the right
constraint for two-sided bridges, where `\rho \equiv 1` is not
available; the discrete cousins (the sampling-time reveal schedule —
problem 5's overlap — and the masking measure's grid); and the metric
gap (problem 4) standing between all three theorems and sample
QUALITY: these dials optimize output error, and the field's
empirical winners are chosen against a currency no theorem here
speaks.
