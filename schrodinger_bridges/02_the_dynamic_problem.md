# The Dynamic Problem: Bridges And The h-Transform Pair

## The Question

`01` solved the static problem: the optimal COUPLING of two
marginals against a Gibbs kernel. Diffusion methods need a PATH — a
process to simulate. This file lifts the static answer to path
space and proves three things: the dynamic Schrödinger problem
collapses onto the static one (the bridge term of the KL
decomposition can always be zeroed), its solution is a **two-sided
Doob h-transform** of the reference — `guidance_and_control/01`'s
machinery applied at both ends simultaneously — and with an OU
reference the whole construction degenerates, exactly and at a
computable rate, into this repository's VP diffusion path. The
ledger's original forward process was a Schrödinger bridge all
along; this file proves in which sense.

## The Decomposition

Reference path measure `R` (a diffusion on `[0,T]`, generator `L`,
drift `b`, diffusion `g`), prescribed marginals `\mu` at `0`, `\nu`
at `T`. The dynamic problem:
`\min\ \mathrm{KL}(Q\|R)` over path measures `Q` with
`Q_0 = \mu, Q_T = \nu`.

**Theorem (static KL + bridge mismatch).** For every `Q \ll R`,

```math
\mathrm{KL}(Q\,\|\,R)
\;=\;
\mathrm{KL}\big(q_{0T}\,\|\,r_{0T}\big)
\;+\;
\mathbb{E}_{(x_0,x_T)\sim q_{0T}}\,
\mathrm{KL}\big(Q^{x_0,x_T}\,\|\,R^{x_0,x_T}\big),
```

where `q_{0T}, r_{0T}` are the endpoint couplings and
`Q^{x_0,x_T}, R^{x_0,x_T}` the conditional path laws (bridges).
Consequently

```math
\min_{Q:\,Q_0=\mu,\,Q_T=\nu} \mathrm{KL}(Q\|R)
\;=\;
\min_{\pi\in\Pi(\mu,\nu)} \mathrm{KL}(\pi\,\|\,r_{0T}),
```

attained uniquely by `Q^* = \int R^{x_0,x_T}\,\mathrm{d}\pi^*` —
the reference's own bridges, remixed by `01`'s static plan.

*Proof.* The identity is the KL chain rule under the measurable map
path `\mapsto` (endpoints, bridge): disintegrate both measures over
the endpoint pair. The second term is nonnegative and vanishes iff
`Q` uses `R`'s bridges `q_{0T}`-a.s.; the endpoint constraint
touches only the first term. Minimize each independently: the
candidate `Q^*` zeroes the second and hands the first to `01`,
whose minimizer `\pi^*` exists and is unique (Donsker–Varadhan +
strict convexity, proved there). Uniqueness of `Q^*`: both terms
are uniquely minimized. ∎

This is the theorem that makes the dynamic problem TRACTABLE: all
the process-level difficulty lives in objects the reference already
owns (its bridges), and everything data-dependent is a static
coupling. `03`'s IMF algorithm is this decomposition run as a
fixed-point iteration.

## The Two-Sided h-Transform

`01` gives `\mathrm{d}\pi^* = f(x_0)\,g(x_T)\,\mathrm{d}r_{0T}`, so

```math
\frac{\mathrm{d}Q^*}{\mathrm{d}R} \;=\; f(x_0)\,g(x_T)
```

— an initial tilt times a TERMINAL functional. Define the
**potential pair**

```math
\varphi_t(x) := \mathbb{E}_R\big[g(x_T)\,\big|\,x_t=x\big],
\qquad
\hat\varphi_t(x) := \int f(x_0)\,r_{0t}(x_0, x)\,\mathrm{d}x_0 ,
```

(`r_{0t}` = the reference's joint density of `(x_0, x_t)`; so
`\hat\varphi_t = r_t\cdot\mathbb{E}_R[f(x_0)|x_t]`).

**Theorem (the factorization and the pair of equations).**

```math
q^*_t(x) \;=\; \varphi_t(x)\,\hat\varphi_t(x),
\qquad\text{hence}\qquad
\nabla\log q^*_t = \nabla\log\varphi_t + \nabla\log\hat\varphi_t ,
```

with `\varphi` solving the backward Kolmogorov equation of `R`
(`\partial_t\varphi + L\varphi = 0`, terminal data `\varphi_T = g`)
and `\hat\varphi` the forward (Fokker–Planck) equation
(`\partial_t\hat\varphi = L^*\hat\varphi`, initial data
`\hat\varphi_0 = f\,r_0`), coupled only through the boundary
conditions

```math
\varphi_0\,\hat\varphi_0 = \mu,
\qquad
\varphi_T\,\hat\varphi_T = \nu
```

— the **dynamic Schrödinger system**, `01`'s `(f, g)` equations
with the kernel now a semigroup.

*Proof.* Factorization: by the Markov property of `R`, `x_0` and
`x_T` are conditionally independent given `x_t`, so
`q^*_t(x) = r_t(x)\,\mathbb{E}_R[f(x_0)g(x_T)|x_t=x]
= r_t\cdot\mathbb{E}[f|x_t]\cdot\mathbb{E}[g|x_t]
= \hat\varphi_t\,\varphi_t`. Equations: `\varphi_t(x_t)` is a
martingale under `R` (tower property — `guidance_and_control/01`'s
first lemma verbatim, with `g(x_T)` the positive terminal
functional), and a martingale of a Markov diffusion solves the
backward equation; `\hat\varphi_t` is the kernel action
`\hat\varphi_t = \int f r_0\,\cdot r_{t|0}`, i.e. an (unnormalized)
initial law pushed by the reference semigroup, which is what the
forward equation propagates. Boundary: at `t=0`,
`\varphi_0 f r_0 = q^*_0 = \mu`; at `t=T`,
`\hat\varphi_T\,g = q^*_T = \nu`. ∎

**Corollary (the SDE pair — the bridge is a sampler).** `Q^*` is
the diffusion

```math
\mathrm{d}x \;=\; \big[\,b(x,t) + g^2\,\nabla\log\varphi_t(x)\,\big]\,\mathrm{d}t + g\,\mathrm{d}W,
\qquad x_0 \sim \mu,
```

and its time reversal is the reversal of `R` drift-corrected by
`+g^2\nabla\log\hat\varphi_t`, initialized from `\nu`.

*Proof.* Forward: `\mathrm{d}Q^*/\mathrm{d}R = f(x_0)g(x_T)`; the
`f(x_0)` factor only reweights the initial law (it is
`\mathcal{F}_0`-measurable), and the `g(x_T)` factor is a positive
terminal functional — `guidance_and_control/01`'s h-transform
theorem adds exactly `g^2\nabla\log\varphi_t` to the drift.
Reverse: run the same argument in reverse time, where the roles of
`f` and `g` swap; Anderson (`score_foundations/03`) gives the
reversal of `R`, and the factorization
`\nabla\log q^* = \nabla\log\varphi + \nabla\log\hat\varphi` makes
the two descriptions marginal-consistent (the reverse drift
`-b + g^2\nabla\log q^* + g^2\cdot(-\nabla\log\varphi)` — the
score splits between the two potentials). ∎

Conditioning tilted ONE end (`guidance_and_control/01`); reward
fine-tuning tilted one end (`guidance_and_control/04`); the
Schrödinger bridge is the same transform with BOTH ends pinned, and
the score's split into `\varphi\cdot\hat\varphi` is the structural
payoff: the estimand of bridge models is a potential, not a score,
and the two coincide only when one potential is constant.

## The OU Reference, Exactly

Now the ledger's own reference: `R` = the VP forward process
(`\mathrm{d}x = -\tfrac12 x\,\mathrm{d}t + \mathrm{d}W`, `\beta=1`)
started from `\mu`, horizon `T`; `\alpha = e^{-T/2}`,
`\sigma^2 = 1-\alpha^2` (`score_foundations/01`). Its endpoint
kernel is `r_{T|0}(y|x) = \mathcal{N}(y;\ \alpha x,\ \sigma^2)`, so:

**Observation (the dynamic problem is `01` with a shrunk cost).**
The static problem against `r_{0T}` is entropic OT with cost
`c(x,y) = (y-\alpha x)^2/2` and temperature `\varepsilon = \sigma^2`.
For Gaussian marginals `\mu = \mathcal{N}(0,s^2)`,
`\nu = \mathcal{N}(0,1)`, `01`'s closed form applies to the pair
`(\alpha x,\ y)`:

```math
\alpha\,c^* \;=\; \mathrm{Cov}(\alpha x_0,\,x_T)
\;=\; \sqrt{\alpha^2 s^2 + \tfrac{\sigma^4}{4}}\; -\; \frac{\sigma^2}{2}.
```

**Theorem (the diffusion path is a Schrödinger bridge — exactly).**
If the target is the forward process's OWN marginal,
`\nu = p_T = \mathcal{N}(0,\ \alpha^2 s^2 + \sigma^2)`, then
`Q^* = R` for every horizon `T`.

*Proof.* `f \equiv 1, g \equiv 1` satisfies the Schrödinger system
with these marginals (`\varphi \equiv 1`,
`\hat\varphi_t = r_t`, boundary conditions read `r_0 = \mu`,
`r_T = p_T` — true by construction). The system's solution is
unique up to the scalar trade (`01`), and the induced `Q^*` is
unique (the decomposition theorem). ∎

**Theorem (the degenerate limit, with rate).** If instead
`\nu = \gamma = \mathcal{N}(0,1)` — the PRIOR practice actually
samples from — then the bridge coupling deviates from the
reference's by

```math
c^*  \;=\; \alpha s^2 \;+\; O(\alpha^3),
\qquad \alpha = e^{-T/2},
```

i.e. the Schrödinger correction to the VP path vanishes at the rate
`e^{-3T/2}` in the covariance (reference value: `c = \alpha s^2`).

*Proof.* Expand the closed form with `\sigma^2 = 1-\alpha^2`:
`\alpha c^* = \tfrac12\sqrt{(1-\alpha^2)^2 + 4\alpha^2 s^2} - \tfrac12(1-\alpha^2)
= \tfrac12\big[1 + \alpha^2(2s^2-1) + O(\alpha^4)\big] - \tfrac12(1-\alpha^2)
= \alpha^2 s^2 + O(\alpha^4)`. ∎

Two readings, both earned. Short horizon: as `T \to 0`
(`\alpha \to 1, \sigma^2 \to 0`) the temperature dies and the
bridge converges to the OT coupling (`01`'s
`\varepsilon \to 0` endpoint) — Schrödinger bridges INTERPOLATE
between optimal transport (`T = 0`) and the ledger's diffusion
(`T = \infty`), with the horizon as the dial. Long horizon: the
`O(\alpha^3)` correction is the bridge-side twin of the prior
mismatch term `e^{-T}`-ish that `samplers_and_convergence/03`
charges for initializing at `\gamma` instead of `p_T` — one
discrepancy, priced in two currencies: the Girsanov decomposition
pays it as a KL surcharge and keeps the process; the bridge
ABSORBS it into the optimal coupling and keeps the marginals. The
numerical check in `verification/verify.py` verifies the closed
form against log-domain Sinkhorn on the OU kernel at three
horizons, and the `f = g = 1` exactness directly.

## Load-Bearing Audit

```text
R Markov                the factorization q = phi phihat rests on
                        conditional independence given x_t; non-
                        Markov references (memory kernels) lose the
                        potential pair entirely;
bridges of R reusable   the decomposition zeroes the bridge term
                        only if R's bridges are simulable/learnable
                        — for OU they are Gaussian in closed form,
                        for general references this is itself a
                        modeling assumption (03's algorithms);
positive densities      f, g > 0 and r_{0T} > 0 inherited from 01:
                        disjoint-support marginals at T = 0 (exact
                        OT) sit outside the framework;
Gaussian marginals      the OU section's closed forms; the exactness
                        theorem (nu = p_T) needs NO Gaussianity —
                        f = g = 1 solves the system for any data mu;
common g                the SDE pair modifies drift only, guidance/
                        01's inheritance — bridges never touch the
                        diffusion coefficient.
```

## Position In The Coordinate System

The path coordinate `\mathcal{P}`, now fully two-dimensional:
marginal family TIMES coupling, with the horizon `T` interpolating
the coupling from OT to independence. The estimand coordinate
acquires a new entry — the potential `\varphi` (equivalently its
log-gradient), which REDUCES to the score when one end is free:
`\nu = p_T` forces `\varphi \equiv 1` and the bridge sampler is
Anderson's reversal verbatim. The solver coordinate is untouched:
the SDE pair is simulated by every phase-A method, and pays the
same discretization bills.

## What Remains Open

Quantitative stability of the potential pair: how error in
`(f, g)` — estimated, in practice, by `03`'s iterations or a
network — propagates through `\varphi, \hat\varphi` into the
sampled law (the analogue of `statistical_theory/01`'s propagation
theory, absent for bridges); the non-Gaussian rate for the
`T \to \infty` degeneration (the `O(\alpha^3)` is proved here for
Gaussian pairs; the general statement should follow from the
semigroup's spectral gap and does not exist in usable form); and
data-dependent reference design — the reference process is a free
parameter of the whole framework, OU is chosen for its closed-form
bridges, and no theorem ranks references by the estimation or
sampling difficulty of the resulting potentials (the phase-G
version of the schedule-design question).
