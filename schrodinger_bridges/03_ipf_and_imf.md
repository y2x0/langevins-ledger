# IPF And IMF: The Two Projection Algorithms

## The Question

`02` characterized the bridge; nothing yet computes it. The two
algorithm families both iterate KL projections, onto DIFFERENT
constraint sets: **IPF** (Fortet 1940; Sinkhorn; Diffusion
Schrödinger Bridge) alternately fixes the two endpoint marginals;
**IMF** (Peluchetti; Shi et al.; DSBM) alternates between the
Markov class and the reciprocal class. This file proves what each
projection IS (an h-transform; a conditional expectation — the
repository's two favorite objects), proves monotone convergence for
IPF by the same telescope as `01`, solves the Gaussian case in full
— the iteration collapses to a scalar Möbius map whose contraction
factor per round is EXACTLY the fourth power of the optimal plan's
correlation — and audits where log-concavity carries the general
rates (2025 results, assembled not reproduced).

## IPF: Alternating Marginal Fixing

Constraint sets on path measures:
`\Gamma_\mu = \{Q : Q_0 = \mu\}`, `\Gamma_\nu = \{Q : Q_T = \nu\}`.
Start `P^0 = R`; alternately I-project onto `\Gamma_\nu`, then
`\Gamma_\mu`.

**Lemma (the projection is an h-transform).** The minimizer of
`\mathrm{KL}(\cdot\,\|\,P)` over `\Gamma_\nu` is

```math
\mathrm{d}P' \;=\; \mathrm{d}P\cdot\frac{\mathrm{d}\nu}{\mathrm{d}P_T}(x_T),
```

and for every `Q \in \Gamma_\nu` the Pythagorean identity holds:
`\mathrm{KL}(Q\|P) = \mathrm{KL}(Q\|P') + \mathrm{KL}(\nu\|P_T)`.

*Proof.* Disintegrate over the terminal point: the KL chain rule
gives `\mathrm{KL}(Q\|P) = \mathrm{KL}(Q_T\|P_T) +
\mathbb{E}_{Q_T}\,\mathrm{KL}(Q^{x_T}\|P^{x_T})` (endpoint law plus
conditional path laws). For `Q \in \Gamma_\nu` the first term is
the constant `\mathrm{KL}(\nu\|P_T)`; the second equals
`\mathrm{KL}(Q\|P')` exactly, because `P'` has terminal law `\nu`
and P's conditionals. Identity proved; `P'` (zero second term at
`Q = P'`) is the minimizer. ∎

This is `01`'s I-projection lemma verbatim, lifted to path space —
and the reweighting is by a terminal functional, so by
`guidance_and_control/01` each IPF iterate is a bona fide DIFFUSION:
the h-transform of the previous one by
`h_t(x) = \mathbb{E}\big[\tfrac{\mathrm{d}\nu}{\mathrm{d}P_T}(x_T)\,\big|\,x_t = x\big]`,
drift-corrected by `g^2\nabla\log h_t`. (The `\Gamma_\mu` step is
the mirror statement in reverse time — the "half-bridges" of DSB,
which is why practice alternates fitting a backward and a forward
process.)

**Corollary (monotone convergence).** With
`Q^*` the Schrödinger bridge (`\in \Gamma_\mu \cap \Gamma_\nu`),
each IPF step decreases `\mathrm{KL}(Q^*\|P^n)` by exactly the
current marginal defect (`\mathrm{KL}(\mu\|P^n_0)` or
`\mathrm{KL}(\nu\|P^n_T)`); the defects sum to at most
`\mathrm{KL}(Q^*\|R) < \infty`, hence `\to 0`: the iterates'
endpoint marginals converge to `(\mu,\nu)` in KL. *Proof.* Apply
the identity at `Q = Q^*` and telescope — `01`'s corollary, word
for word. ∎

## The Gaussian Case, Solved In Full

One dimension, `\mu = \mathcal{N}(0,a^2)`,
`\nu = \mathcal{N}(0,b^2)`; by `02`'s reduction it suffices to run
the STATIC iteration against the Gibbs coupling with cross-precision
`q` (for cost `c/\varepsilon` on `\mathbb{R}^2`, `q = 1/\varepsilon`;
for the OU reference, `q = \alpha/\sigma^2`).

**Step 1 (the invariant).** Every IPF iterate has the form
`f_n(x)\,g_n(y)\,K(x,y)` (`01`'s family is preserved: marginal
fixing keeps conditionals, and conditionals of `fgK` stay in the
family). For Gaussian `f, g, K` the joint precision matrix is

```math
\begin{pmatrix} p_n & -q \\ -q & r_n \end{pmatrix}:
```

`f` moves only `p`, `g` moves only `r`, and the OFF-DIAGONAL
precision `-q` is untouchable by any IPF step. One number per
potential; the algorithm is a map `(p_n, r_n) \to (p_{n+1}, r_{n+1})`.

**Step 2 (the map).** Fixing the `y`-marginal keeps the conditional
`x|y` — in precision coordinates, keeps `(p, q)` — and sets
`\mathrm{Var}(y) = p/(pr - q^2) = b^2`, i.e.
`r \leftarrow 1/b^2 + q^2/p`. The mirror step sets
`p \leftarrow 1/a^2 + q^2/r`. One full round is the scalar Möbius
iteration

```math
p_{n+1} \;=\; \frac{1}{a^2} \;+\; \frac{q^2}{\;\dfrac{1}{b^2} + \dfrac{q^2}{p_n}\;}.
```

**Step 3 (fixed point = the Schrödinger system).** At a fixed
point, `\mathrm{Var}(x) = r/(pr-q^2) = a^2` and
`\mathrm{Var}(y) = b^2` simultaneously, and the covariance
`c = q/(pr - q^2)` satisfies `a^2b^2 - c^2 = c/q`: with
`q = 1/\varepsilon` this is `c^2 + \varepsilon c - a^2 b^2 = 0` —
`01`'s quadratic, recovered from the algorithm's statics. ∎

**Theorem (the contraction factor, exactly).** The map of Step 2
has derivative

```math
\frac{\mathrm{d}p_{n+1}}{\mathrm{d}p_n}
\;=\; \left(\frac{q^2}{p_* r_*}\right)^{\!2}
\;=\; \rho^4,
\qquad
\rho \;=\; \mathrm{Corr}_{\pi^*}(x, y) \;=\; \frac{c_\varepsilon}{ab},
```

at the fixed point: IPF converges locally at the linear rate
`\rho^4` per full round (`\rho^2` per half-step), where `\rho` is
the correlation of the optimal plan.

*Proof.* Chain rule on Step 2's composition:
`\mathrm{d}p_{n+1}/\mathrm{d}p_n = q^4/(p_*^2 r_*^2)`. The
identification: under precision `[[p,-q],[-q,r]]` the covariance is
`[[r,q],[q,p]]/(pr-q^2)`, so
`\mathrm{Corr}^2 = q^2/(pr)`. Monotone global convergence: the map
is increasing and concave in `p_n > 0` with a unique positive fixed
point, so iterates converge monotonically from any start. ∎

Worked numbers (`a=1, b=2, \varepsilon=1`, `01`'s example):
`c_\varepsilon = 1.5616`, `\rho = 0.781`, contraction
`\rho^4 = 0.372` — one order of magnitude of error per ~2.3 rounds.
The two audits this quantifies: as `\varepsilon \to \infty`,
`\rho \to 0` — instant convergence to the product coupling (CFM's
endpoint is free); as `\varepsilon \to 0`, `c_\varepsilon \to ab`,
`\rho \to 1` — the rate degrades to nothing EXACTLY as the OT
endpoint is approached, which is `01`'s Birkhoff-factor warning
made into a formula: the price of transport is paid in iterations,
`(1-\rho^4)^{-1} \sim ab/(2\varepsilon)` rounds per e-fold as
`\varepsilon \to 0`. The recursion, its fixed point, and the
`\rho^4` rate are checked against `01`'s closed form in
`verification/verify.py`.

## IMF: Markov Meets Reciprocal

Two classes: `\mathcal{M}` = Markov path measures;
`\mathcal{R}(R)` = the reciprocal class, measures sharing `R`'s
bridges (`Q = \int R^{x_0,x_T}\,\mathrm{d}q_{0T}`). By `02`'s
decomposition theorem, the Schrödinger bridge is the unique member
of `\mathcal{M} \cap \mathcal{R}(R)` with marginals `(\mu,\nu)`.
IMF alternates projections, PRESERVING the endpoint marginals at
every step (the practical advantage over IPF, which violates one
marginal at every step).

**Lemma (reciprocal projection).** For Markov `Q` with endpoint
coupling `q_{0T}`:
`\Pi_{\mathcal{R}}(Q) := \arg\min_{P\in\mathcal{R}(R)}\mathrm{KL}(Q\|P)
= \int R^{x_0,x_T}\,\mathrm{d}q_{0T}` — keep the coupling, swap in
`R`'s bridges. Moreover
`\mathrm{KL}(Q^*\|\Pi_{\mathcal{R}}(Q)) \le \mathrm{KL}(Q^*\|Q)`.

*Proof.* First claim: decompose
`\mathrm{KL}(Q\|P) = \mathrm{KL}(q_{0T}\|p_{0T}) +
\mathbb{E}_{q_{0T}}\mathrm{KL}(Q^{x_0,x_T}\|R^{x_0,x_T})` (`P`'s
bridges are `R`'s); the second term is `P`-free, the first is
minimized (to zero) at `p_{0T} = q_{0T}`. Second claim: since `Q^*`
has `R`'s bridges,
`\mathrm{KL}(Q^*\|Q) = \mathrm{KL}(\pi^*\|q_{0T}) +
\mathbb{E}_{\pi^*}\mathrm{KL}(R^{x_0,x_T}\|Q^{x_0,x_T})
\ge \mathrm{KL}(\pi^*\|q_{0T}) = \mathrm{KL}(Q^*\|\Pi_{\mathcal{R}}(Q))`. ∎

**Lemma (Markovian projection = conditional expectation).** Let
`Q` be a (generally non-Markov) path measure with drift process
`u_t` and diffusion `g` (its Doob–Meyer/Girsanov representation).
Then

```math
\Pi_{\mathcal{M}}(Q) \;:=\; \arg\min_{M\in\mathcal{M}}\mathrm{KL}(Q\|M)
\quad\text{has drift}\quad
m^*(x,t) \;=\; \mathbb{E}_Q\big[\,u_t\,\big|\,x_t = x\,\big],
```

and `\Pi_{\mathcal{M}}(Q)` has the SAME time-marginals as `Q`.

*Proof.* Girsanov (`samplers_and_convergence/03`):
`\mathrm{KL}(Q\|M) = \mathrm{KL}(Q_0\|M_0) +
\tfrac{1}{2}\mathbb{E}_Q\!\int_0^T \|u_t - m(x_t,t)\|^2/g^2\,\mathrm{d}t`.
Minimizing the quadratic over Markov drifts `m` pointwise in
`(x,t)` is an L2 projection onto `\sigma(x_t)`-measurable
functions: the conditional expectation — the Bregman projection
lemma (`discrete_diffusion/03`), in its `\phi = \|\cdot\|^2`
clothing, fifth appearance. Marginal preservation: `Q`'s marginals
satisfy the continuity equation driven by
`\mathbb{E}_Q[u_t|x_t = \cdot]` — `flow_matching/01`'s
marginal-velocity theorem, same computation — which is exactly the
Fokker–Planck flow of `\Pi_{\mathcal{M}}(Q)`. ∎

So IMF's inner step IS conditional flow matching against the
current coupling: simulate `R`-bridges between coupled endpoint
pairs, regress the bridge drift on `x_t` — DSBM's bridge-matching
loss is this lemma's regression, and `04` prices what happens when
the regression is inexact.

**Theorem (fixed points; monotone KL).** IMF's iterates keep
marginals `(\mu,\nu)`; the value `\mathrm{KL}(Q^*\|Q_n)` is
non-increasing (reciprocal step: the lemma above; Markovian step:
statement — Shi et al. 2023, Peluchetti 2023); and any fixed point
lies in `\mathcal{M}\cap\mathcal{R}(R)` with marginals `(\mu,\nu)`,
hence EQUALS `Q^*` by `02`'s uniqueness. ∎

The asymmetry in that theorem is deliberate and honest: the
reciprocal half's inequality is two lines (above); the Markovian
half's needs `Q^*`'s Markovianity played against the projection and
is assembled from the sources, not reproduced. What IS proved here
suffices for the fixed-point characterization — an IMF stationary
point has no freedom left.

## The 2024–25 Rates, Audited

Quantitative convergence, as currently known:

```text
IPF/Sinkhorn        linear at rho^4 per round — proved above for
                    Gaussians; general marginals: Birkhoff-Hopf
                    contraction under bounded cost (01), degrading
                    as eps -> 0;
IMF, Gaussian       exponential KL convergence, explicit factor
                    (IPMF line: Gushchin et al. 2024) — the
                    Gaussian family is closed under both
                    projections, so the iteration is again a matrix
                    recursion (statement; our scalar method extends
                    but the bookkeeping is theirs);
IMF, log-concave    exponential KL convergence for sufficiently
                    long horizon T (Gentiloni Silveri-Conforti-
                    Durmus 2025): the engine is a new contraction
                    property of the Markovian projection, and BOTH
                    hypotheses are load-bearing — log-concavity
                    enters through gradient-flow/functional-
                    inequality machinery for the projected drift,
                    and large T makes the reference's bridges
                    forget their endpoints (02's alpha -> 0
                    regime); weakly log-concave marginals: same
                    paper, degraded constants;
IMF, finite state   exponential with explicit factor (Sokolov-
                    Korotin 2025) — the discrete analogue, where
                    the ledger's discrete_diffusion machinery
                    would apply.
```

None of these track ESTIMATION error: every rate above is for the
exact projections. The gap between "the iteration converges" and
"the trained network's iteration converges" is `04`'s subject, and
it is the same gap phase A left to phase F.

## Load-Bearing Audit

```text
KL(Q* || R) < infty     the IPF telescope's budget; fails iff the
                        static problem is infeasible (01's
                        existence theorem is the guard);
Gaussian family closure the entire solved case: both IPF steps
                        preserve Gaussianity BECAUSE conditionals
                        of Gaussians are Gaussian — the same
                        closure that made 01's Step 1 work;
q fixed under IPF       the invariant that reduces the iteration
                        to one scalar per potential; it is the
                        algorithmic shadow of 01's uniqueness-up-
                        to-scalar for (f, g);
Girsanov representation the Markovian projection lemma assumes Q
                        has a drift representation w.r.t. the same
                        g — reciprocal mixtures of R-bridges do
                        (their bridges are R's own SDE bridges);
                        arbitrary path measures need not;
exactness of projections all four rate results; a learned
                        regression replaces m* with an estimate
                        and NO current rate survives that
                        substitution (04).
```

## Position In The Coordinate System

The solver coordinate `\mathcal{S}`, but at the OUTER level: IPF
and IMF do not integrate an SDE, they iterate on which SDE to
integrate — a fixed-point loop wrapped around the entire
`(\mathcal{P}, s, \mathcal{S})` triple, converging to the bridge
that `02` characterized. The estimand inside the loop is the one
the repository already owns: an h-transform potential (IPF) or a
conditional-expectation drift (IMF/CFM). Nothing new is trained;
something old is trained repeatedly.

## What Remains Open

A global (not fixed-point-local) rate for Gaussian IPF is easy
(the Möbius map is monotone concave — the argument is sketched
above); the general sharp rate as `\varepsilon \to 0` is not —
the known bounds are polynomial in `1/\varepsilon` and the
Gaussian formula `(1-\rho^4)^{-1} \sim ab/2\varepsilon` suggests
the right conjecture; finite-iteration + estimation-error
composition (each IMF round fits a regression; how do `n` rounds
of `\epsilon`-inexact Markovian projections compound? — linear
accumulation as in `distillation/01`, or does the reciprocal
projection heal drift error the way the SDE heals score error in
`samplers_and_convergence/05`?); and IPMF — the
forward/backward-alternating scheme practice actually runs — has
Gaussian-case convergence only (its combination of the two
projection types resists both proof strategies; the 2024 paper
conjectures general convergence and proves none).
