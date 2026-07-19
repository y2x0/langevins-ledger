# The Schrödinger System: Entropic Optimal Transport

## The Question

`flow_matching/05` proved that FM with the independent coupling does
not produce the OT map, and `flow_matching/03` improves couplings
iteratively without ever naming what the improvement converges TO.
This file names the target: the **static Schrödinger problem** —
entropic optimal transport — whose solution is the unique coupling
that is simultaneously as cheap as OT wants and as spread as entropy
demands, with a dial `\varepsilon` interpolating between them. The
independent coupling of CFM is the `\varepsilon = \infty` endpoint;
the OT coupling is `\varepsilon = 0`. Everything downstream (the
dynamic problem of `02`, the IPF/IMF algorithms of `03`, bridge
matching in `04`) is built on the objects constructed here.

## The Two Forms

Marginals `\mu, \nu` on `\mathbb{R}^d`, cost
`c(x,y) = |x-y|^2/2`, temperature `\varepsilon > 0`. Define the
Gibbs kernel measure and its normalization

```math
\mathrm{d}\bar K_\varepsilon(x,y) \;=\;
\frac{1}{Z_\varepsilon}\,e^{-c(x,y)/\varepsilon}\,
\mathrm{d}\mu(x)\,\mathrm{d}\nu(y),
\qquad
Z_\varepsilon = \int e^{-c/\varepsilon}\,\mathrm{d}\mu\,\mathrm{d}\nu .
```

**Lemma (the two forms agree).** Over couplings
`\pi \in \Pi(\mu,\nu)`,

```math
\int c\,\mathrm{d}\pi \;+\; \varepsilon\,\mathrm{KL}(\pi\,\|\,\mu\otimes\nu)
\;=\;
\varepsilon\,\mathrm{KL}(\pi\,\|\,\bar K_\varepsilon)\;-\;\varepsilon\log Z_\varepsilon .
```

*Proof.* Expand
`\mathrm{KL}(\pi\|\bar K_\varepsilon) = \int\log\frac{\mathrm{d}\pi}{\mathrm{d}\mu\otimes\nu}\,\mathrm{d}\pi + \frac{1}{\varepsilon}\int c\,\mathrm{d}\pi + \log Z_\varepsilon`. ∎

So regularized transport IS a KL projection: find the coupling
closest to the Gibbs kernel. This is Schrödinger's 1931 question
(most likely evolution of a particle cloud given both endpoints),
and it is the form every algorithm in `03` attacks.

## Existence And Uniqueness

**Theorem.** If `\mu,\nu` have finite second moments, the minimizer
`\pi^*_\varepsilon` of `\mathrm{KL}(\cdot\,\|\,\bar K_\varepsilon)`
over `\Pi(\mu,\nu)` exists and is unique.

*Proof.* `\Pi(\mu,\nu)` is tight (both marginals fixed), hence
weakly compact by Prokhorov, and convex. The map
`\pi \mapsto \mathrm{KL}(\pi\|\bar K_\varepsilon)` is weakly lower
semicontinuous: by the Donsker–Varadhan representation (the named
lemma this proof leans on),
`\mathrm{KL}(\pi\|R) = \sup_{\varphi\in C_b}\{\int\varphi\,\mathrm{d}\pi - \log\int e^\varphi\,\mathrm{d}R\}`,
a supremum of weakly-continuous affine functionals. An lsc function
on a compact set attains its minimum, and the minimum is finite
because the product coupling gives
`\mathrm{KL}(\mu\otimes\nu\|\bar K_\varepsilon) = \frac{1}{\varepsilon}\int c\,\mathrm{d}\mu\,\mathrm{d}\nu + \log Z_\varepsilon < \infty`
(second moments). Uniqueness: `t\log t` is strictly convex, so KL is
strictly convex on the convex set where it is finite. ∎

## The Product Structure And The System

**Theorem (finite case, proved in full).** Let `x` range over `m`
states, `y` over `n`, with `\mu_i, \nu_j > 0` and `K_{ij} > 0`. The
minimizer of `\sum_{ij}\pi_{ij}\log(\pi_{ij}/K_{ij})` over the
transport polytope has the form

```math
\pi^*_{ij} \;=\; f_i\, g_j\, K_{ij},
\qquad
f_i \sum_j g_j K_{ij} = \mu_i,
\qquad
g_j \sum_i f_i K_{ij} = \nu_j
```

— the **Schrödinger system** — with `(f, g)` unique up to the scalar
trade `(\lambda f, g/\lambda)`.

*Proof.* The optimum is interior: if `\pi_{ij} = 0` somewhere, mix
with the product coupling — the one-sided derivative of `t\log t` at
`0^+` is `-\infty`, so `F((1-\lambda)\pi + \lambda\,\mu\otimes\nu)`
strictly decreases for small `\lambda`, contradiction. At an
interior optimum, Lagrange multipliers `\varphi_i, \psi_j` for the
row/column constraints give stationarity
`\log(\pi_{ij}/K_{ij}) + 1 = \varphi_i + \psi_j`, i.e.
`\pi_{ij} = f_i g_j K_{ij}` with `f_i = e^{\varphi_i - 1/2}`,
`g_j = e^{\psi_j - 1/2}`. The marginal constraints are the displayed
system. Uniqueness of the ratio: if `fg K` and `f'g'K` have equal
entries then `f_i/f'_i = g'_j/g_j` for all `i,j`, forcing both sides
constant. ∎

In the continuum the same structure holds
(`\mathrm{d}\pi^* = f(x)g(y)\,\mathrm{d}K_\varepsilon`,
Rüschendorf–Thomsen; Csiszár) — statement, the finite proof carries
the idea. `02` will recognize `f` and `g` as a Doob pair: the
h-transform machinery of `guidance_and_control/01`, applied at BOTH
ends.

## Sinkhorn = Alternating I-Projection

Let `\Gamma_\mu` (resp. `\Gamma_\nu`) be the couplings with correct
`x`- (resp. `y`-) marginal. Sinkhorn alternately fixes each marginal
by rescaling — i.e. alternately I-projects onto `\Gamma_\mu` and
`\Gamma_\nu`.

**Lemma (I-projection in closed form + the Pythagorean identity).**
For a joint `Q` with `x`-marginal `Q_x`, the minimizer of
`\mathrm{KL}(\cdot\,\|\,Q)` over `\Gamma_\mu` is
`\mathrm{d}P = \mathrm{d}\mu(x)\,Q(\mathrm{d}y|x)` (swap the
marginal, keep the conditional), and for EVERY `\pi \in \Gamma_\mu`:

```math
\mathrm{KL}(\pi\,\|\,Q) \;=\; \mathrm{KL}(\pi\,\|\,P) \;+\; \mathrm{KL}(\mu\,\|\,Q_x).
```

*Proof.* Chain rule for KL:
`\mathrm{KL}(\pi\|Q) = \mathrm{KL}(\pi_x\|Q_x) + \mathbb{E}_{\pi_x}\mathrm{KL}(\pi(\cdot|x)\|Q(\cdot|x))`.
For `\pi \in \Gamma_\mu` the first term is `\mathrm{KL}(\mu\|Q_x)`;
the second equals `\mathrm{KL}(\pi\|P)` exactly (P has marginal
`\mu`, conditional `Q(\cdot|x)`, so its own chain rule has zero
first term). Identity proved; minimizing over `\pi` makes `P`
optimal since the correction term is `\pi`-free. ∎

**Corollary (monotone convergence, proved).** Along Sinkhorn's
iterates `Q^{(0)} = \bar K_\varepsilon, Q^{(1)}, Q^{(2)}, \dots`,
the value `\mathrm{KL}(\pi^*\|Q^{(k)})` decreases by EXACTLY the
marginal defect `\mathrm{KL}(\mu\|Q^{(k)}_x)` (or `\nu`-defect) at
each step; the defects are summable, hence `\to 0`: the iterates'
marginals converge to `(\mu,\nu)` in KL. *Proof.* Apply the identity
with `\pi = \pi^*` (which lies in both constraint sets) and
telescope. ∎

The LINEAR rate — contraction of Sinkhorn in the Hilbert projective
metric with factor governed by
`\theta = \sup_{ijkl} \frac{K_{ik}K_{jl}}{K_{jk}K_{il}}` — is
Birkhoff–Hopf via Franklin–Lorenz: named, not reproduced (the heavy
lemma this file assembles around). What the audit must retain: the
factor degrades to `1` as `\varepsilon \to 0` (the kernel loses
uniform positivity), which is exactly the regime practice wants.

## The Gaussian Case, Exactly

`\mu = \mathcal{N}(0, a^2)`, `\nu = \mathcal{N}(0, b^2)` in one
dimension.

**Step 1 (the optimizer is Gaussian, proved).** For Gaussian
`\mu\otimes\nu`, both `\int c\,\mathrm{d}\pi` and
`\mathbb{E}_\pi[\log(\mu\otimes\nu)]` depend on `\pi` only through
second moments, so the objective is
`(\text{function of second moments}) - \varepsilon\, h(\pi)` with
`h` the differential entropy. At fixed second moments the Gaussian
uniquely maximizes `h`; the optimum is Gaussian. ∎

**Step 2 (the scalar reduction, solved).** A zero-mean Gaussian
coupling is determined by `c = \mathrm{Cov}(x,y)`. Then
`\int c\,\mathrm{d}\pi = (a^2 + b^2 - 2c)/2` and (both marginal
entries of the covariance being pinned)
`\mathrm{KL}(\pi\|\mu\otimes\nu) = -\tfrac12\log(1 - c^2/a^2b^2)`.
Setting the `c`-derivative of the objective to zero:
`1 = \varepsilon c/(a^2b^2 - c^2)`, i.e. `c^2 + \varepsilon c - a^2b^2 = 0`:

```math
c_\varepsilon \;=\; \sqrt{a^2b^2 + \tfrac{\varepsilon^2}{4}}\;-\;\frac{\varepsilon}{2}.
```

The plan's conditional is
`\pi^*(y|x) = \mathcal{N}\big(\tfrac{c_\varepsilon}{a^2}x,\ b^2 - \tfrac{c_\varepsilon^2}{a^2}\big)`:
**the entropic plan is the OT map, shrunk and blurred**, with
`\varepsilon` the blur.

**Step 3 (both endpoints, derived).** `c_\varepsilon` is strictly
decreasing in `\varepsilon`. As `\varepsilon \to 0`:
`c_\varepsilon \to ab`, correlation `\to 1`, conditional variance
`\to 0` — the plan concentrates on `y = (b/a)x`, the monotone map
that `flow_matching/05` proved is THE 1-D OT map. As
`\varepsilon \to \infty`:
`c_\varepsilon = a^2b^2/\varepsilon + O(\varepsilon^{-3}) \to 0` —
the independent coupling. Worked numbers (`a=1, b=2`):
`\varepsilon = 1` gives `c = \sqrt{4.25} - 0.5 \approx 1.562`
(correlation `0.78`), against `c = 2` at the OT end and `c = 0` at
the CFM end. ∎

The matrix version
(`C_\varepsilon = A^{1/2}(A^{1/2}BA^{1/2} + \tfrac{\varepsilon^2}{4}I)^{1/2}A^{-1/2} - \tfrac{\varepsilon}{2}I`,
the entropic Bures formula, Janati et al. 2020) reduces to Step 2 on
each 1-D block: statement, the scalar case carries the proof. The
Sinkhorn simulation in `verification/verify.py` checks Step 2's
formula at three temperatures against the discretized algorithm.

## The Fence, Placed

`flow_matching/05` ended at a fence: independent-coupling FM is not
OT, and minibatch-OT couplings buy improvement "at unpriced cost."
This file prices the family. The coupling dial is `\varepsilon`:
CFM sits at `\varepsilon = \infty`; OT at `\varepsilon = 0` is
unattainable by any KL-based algorithm (the contraction factor
degrades, and at `\varepsilon = 0` the plan escapes the
absolutely-continuous class entirely); minibatch OT is an implicit
choice of effective `\varepsilon` set by batch size (statement —
Pooladian et al.); rectified flow (`flow_matching/03`) walks the
coupling toward straightness without ever selecting an
`\varepsilon`. The Schrödinger problem is the member of the family
with a variational characterization, closed-form Gaussian answer,
and a convergent algorithm — which is why it, and not OT itself, is
the practical target `02`–`04` build toward.

## Load-Bearing Audit

```text
finite second moments      existence: makes KL(mu x nu || K) finite —
                           the compactness half never needs it, the
                           finiteness half does;
K > 0 everywhere           the product structure's interior argument
                           and Sinkhorn's well-definedness; costs
                           taking the value +infinity (constraints)
                           break both;
quadratic cost + Gaussian  Step 1's max-entropy argument needs log of
                           the reference to be quadratic; any other
                           (c, reference) pair loses Gaussianity of
                           the plan and the closed form with it;
Birkhoff rate              stated not proved, and its factor blows up
                           as eps -> 0: the linear-rate guarantee
                           evaporates exactly where OT is approached
                           — the honest reading of "Sinkhorn is fast."
```

## Position In The Coordinate System

The path coordinate `\mathcal{P}` in PLAN.md was a family of
marginals; this phase adds the COUPLING as a second, independent
dial of `\mathcal{P}`, and this file solves its static version: the
`\varepsilon`-family of couplings between the endpoints, with CFM
and OT as its boundary points. The estimand and solver coordinates
are untouched here — `02` lifts the static plan to a path measure
(where the estimand question returns), and `03` turns this file's
alternating projections into the dynamic algorithms.

## What Remains Open

Quantitative Sinkhorn rates in the `\varepsilon \to 0` regime that
practice occupies (the Birkhoff factor is vacuous there; the known
`\varepsilon`-dependent complexity bounds are polynomial in
`1/\varepsilon` and believed loose); the sample complexity of the
PLUG-IN entropic plan as a coupling for downstream FM training —
entropic OT estimation rates exist (Genevay et al.; Mena–Weed), but
no result tracks the estimated coupling's error through the CFM
regression into the sampler (the composition this repository would
demand); and the selection problem: no criterion — statistical,
perceptual, or optimization-theoretic — selects `\varepsilon` for a
generative model, the family's version of the schedule-design
question that `statistical_theory/07` closed only in the Gaussian
caricature.
