# Reward Fine-Tuning Is KL Control

## The Question

Fine-tune a diffusion model to increase a reward `r(x_0)` — aesthetic
score, human preference, protein fitness — without destroying the
pretrained model. The standard objective is the standard one:
maximize reward minus a KL to the reference. This file proves that on
path space the problem solves in closed form (the SAME one-line proof
as Bellman's Ledger `rlhf_mathematics/02`), that the optimum is a
Doob h-transform of the pretrained sampler (`01`'s machinery, with
`e^{r/\beta}` in place of a likelihood), and that its value function
satisfies an HJB equation that LINEARIZES under the log transform —
Bellman's `lqr_and_continuous_control/04` (linearly solvable
control), landing in a third repository. One theorem, three ledgers;
this is the bridge file of the trilogy.

## The Problem And Its Closed Form

Let `P` be the pretrained reverse process's PATH measure (from prior
to sample), and optimize over path measures `Q` of controlled
diffusions (same `g`, drift freely modified):

```math
\max_{Q}\quad \mathbb{E}_{Q}\big[r(x_0)\big] \;-\; \beta\,\mathrm{KL}\big(Q\,\|\,P\big).
```

**Theorem (the tilted path measure).** The unique optimum is

```math
\frac{\mathrm{d}Q^*}{\mathrm{d}P} \;=\; \frac{e^{\,r(x_0)/\beta}}{\mathbb{E}_P\big[e^{\,r(x_0)/\beta}\big]},
\qquad
\text{optimal value} \;=\; \beta\,\log\ \mathbb{E}_P\big[e^{\,r(x_0)/\beta}\big].
```

*Proof.* Verbatim `rlhf_mathematics/02`, with completions replaced by
paths: for any `Q \ll P`,

```math
\mathbb{E}_Q[r] - \beta\,\mathrm{KL}(Q\|P)
= -\beta\,\mathbb{E}_Q\Big[\log\frac{\mathrm{d}Q}{\mathrm{d}P\,e^{r/\beta}}\Big]
= -\beta\,\mathrm{KL}\big(Q\,\|\,Q^*\big) + \beta\log \mathbb{E}_P e^{r/\beta},
```

nonnegativity of KL, equality iff `Q = Q^*`. ∎

Everything `rlhf_mathematics/02` derived downstream transfers
verbatim because the proof did: the reward–KL frontier with slope
`\beta`, the temperature limits (`\beta \to \infty`: the pretrained
model; `\beta \to 0`: reward argmax on the support), and **support is
destiny** — `Q^* \ll P`: KL-anchored fine-tuning reweights the
pretrained sampler's path support, never leaves it. (Algorithms that
leave support are violating their own objective, not refuting the
theorem — `05`.)

## The Optimum Is An h-Transform: The Sampler Exists

A tilted path measure is only useful if it is again a SAMPLER — a
Markov diffusion with a computable drift. It is, because the tilt is
by a function of the ENDPOINT:

**Theorem.** Define `h_t(x) := \mathbb{E}_P\big[e^{r(x_0)/\beta}\,\big|\,x_t = x\big]`.
Then `h_t(x_t)` is a reverse-time martingale under `P`, and `Q^*` is
the Doob h-transform of `P` by `h`: the diffusion with drift

```math
\tilde b_t(x) \;=\; b_t^{P}(x) \;+\; g^2\,\nabla\log h_t(x),
```

initialized from the `h_T`-reweighted prior.

*Proof.* Martingale: tower property, exactly `01`'s lemma with
`p(y|x_0)` replaced by `e^{r(x_0)/\beta}` — the two are the same
mathematical object (a positive terminal functional), which is the
precise sense in which CONDITIONING AND REWARD-TILTING ARE ONE
TRANSFORM: guidance is fine-tuning with `r = \log p(y|x_0)` at
`\beta = 1`. The generator computation and marginal identification
are `01`'s second lemma and theorem unchanged. ∎

## The Value Function, And The Third Ledger

**Theorem (Hopf–Cole / linearly solvable control).** Let
`V_t(x) := \beta\log h_t(x)`. Then: (i) `V` is the value function of
the control problem — the optimal reward-minus-KL-to-go from state
`x` at time `t`; (ii) `h` satisfies the LINEAR backward Kolmogorov
equation of `P`, `\partial_t^{\leftarrow} h + L_P h = 0` with terminal
data `h_0 = e^{r/\beta}`; and (iii) `V` therefore satisfies the
nonlinear HJB equation

```math
\partial_t^{\leftarrow} V + L_P V + \frac{g^2}{2\beta}\,\|\nabla V\|^2 = 0,
\qquad V_0 = r,
```

whose optimal control is `u^* = \tfrac{g^2}{\beta}\nabla V =
g^2\nabla\log h` — and whose nonlinearity is REMOVED by the
exponential substitution `h = e^{V/\beta}`.

*Proof.* (ii) is the martingale property differentiated (a martingale
of a Markov process solves the backward equation). (iii): substitute
`V = \beta\log h` into (ii): `\partial h = h\,\partial V/\beta`,
`Lh = \tfrac{h}{\beta}(LV) + \tfrac{h}{2\beta^2}g^2\|\nabla V\|^2`
(Itô/diffusion chain rule for the generator), divide by `h/\beta`.
(i): the KL of path measures with common `g` is the expected
quadratic control cost — Girsanov
(`samplers_and_convergence/03`): `\mathrm{KL}(Q\|P) =
\tfrac{1}{2}\mathbb{E}_Q\!\int\|u_t\|^2/g^2`, so the problem is a
stochastic control problem with running cost
`\tfrac{\beta}{2g^2}\|u\|^2` and terminal reward `r`; its HJB is
(iii), and the verification argument (Itô on `V_t(x_t^{Q^*})` plus
the martingale property) confirms `V` attains it. ∎

The trilogy's accounting, in one place: this theorem is
**Bellman's `rlhf/02`** (the closed form — step one), **this
repository's `01`** (the h-transform — step two), and **Bellman's
`lqr/04`** (Todorov's linearly solvable control: KL control cost
`\Leftrightarrow` linear equation for `z = e^{V/\beta}` — step
three). The desirability function `z`, the RLHF partition function
`Z`, and the guidance likelihood `h_t` are one object; the soft
value `\beta\log E\,e^{r/\beta}` is its logarithm in every ledger.

## What Algorithms Approximate, Stated

`h_t` is an expectation over the model's own future — computable in
closed form never, approximable many ways, and each fine-tuning
method is a choice:

```text
value learning / adjoint matching   regress the optimal drift
                                    g^2 grad log h directly (Domingo-
                                    Enrich et al.: matching the
                                    h-transform's drift with unbiased
                                    targets; statement);
policy-gradient (DDPO, DPOK)        treat the K denoising steps as an
                                    MDP episode and run PPO-style
                                    updates — Bellman's policy_gradient
                                    machinery on the path MDP
                                    (statements; the per-step KL is
                                    Girsanov's integrand);
Diffusion-DPO                       rlhf/03's partition-cancellation
                                    trick on the per-step Gaussian
                                    policies (statement) — the DPO
                                    derivation transfers because THIS
                                    file's closed form does;
best-of-n                           rlhf/04 verbatim: sample n, pick
                                    max reward — the zeroth-order
                                    competitor, with the same
                                    log n - (n-1)/n KL price.
```

## Load-Bearing Audit

```text
terminal reward             the h-transform needs r = r(x_0); running
                            rewards (on the trajectory) change h's
                            equation but not the framework (statement);
common diffusion g          Girsanov's KL formula = control cost; the
                            dictionary breaks if fine-tuning touches g;
Q << P                      support-is-destiny; also where every
                            "reward hacking via off-support" story
                            must locate itself (05);
beta > 0                    the frontier's currency, as in every
                            ledger: beta -> 0 is argmax — 02's
                            omega -> infinity endpoint, same limit,
                            third clothing.
```

## Position In The Coordinate System

The estimand modified by a terminal functional, the solver unchanged,
the path measure re-weighted: reward fine-tuning lives entirely in
the `(s)`-slot as `+g^2\nabla\log h`, exactly like conditioning,
provably. The file is also the trilogy's keystone: decisions
(Bellman), attention (the softmax that computes these very
expectations), and diffusion meet in one exponential tilt with three
names.

## What Remains Open

Approximation guarantees for ANY of the practical methods (adjoint
matching's targets are unbiased; end-to-end error bounds against
`Q^*` do not exist); multi-reward and constrained variants (the
frontier geometry is convex-duality-friendly and undeveloped for
paths); the interaction of the learned `h` with guidance-era
pathologies (`05`); and the honest scoreboard question inherited from
RLHF itself: whether the KL anchor measures the right divergence for
perceptual quality — path KL is a proxy for sample KL
(data-processing, `samplers_and_convergence/03`), and the gap between
them under fine-tuning is unmeasured.
