# What Remains Open

## The Contract For This File

Every notebook in this repository ends with its own honest frontier;
this closing file is for what cuts ACROSS them — the open problems
that recur in multiple families' final sections, stated once,
sharply, with what is already in place to attack them. It is the
repository's summary by way of its gaps, and the trilogy's closing
entry.

## The Six Cross-Cutting Problems

**1. The deep generalization channel.** `03`'s trichotomy isolates
it: which smoother of the empirical-attention score does a given
architecture-plus-optimizer implement, and where is the memorization
transition `n_{\mathrm{mem}}`? Solved: the linear projection case
(`03`), the bandwidth channel (`02`). In place: the empirical score
is EXACTLY attention over the training set
(`score_foundations/06`) — so attention-ledger's machinery
(capacity, inductive bias toward sparse retrieval) is formally
applicable to diffusion generalization, an unexploited bridge. This
is simultaneously the copyright, privacy, and creativity question;
it is the field's most consequential open problem and this
repository's first.

**2. The guidance-times-solver theorem.** CLOSED at the level posed,
by retrofit: `guidance_and_control/06` assembles the composed bound —
KL to the true conditional `\le` prior + `C_\lambda\int g^2\,[`score
error + plug-in error + `(\omega-1)^2\times`classifier Fisher`]` with
`C_\lambda = 3(1+\lambda^2)^2/8\lambda^2` uniquely minimized at
`\lambda = 1`, discretization attaching as in `A/03`, and the ODE
endpoint covered by the `W_2` complement. Still open at the level it
deserves: tight constants, the `\lambda_t`-schedule optimum against
the composite integrand, and the trainable-norm/strong-metric
mismatch the assembly inherits unchanged.

**3. Mode weights.** CLOSED at the level posed, by retrofit: `06`
proves the bound — the mode posterior is a reverse-time martingale
under the `\lambda = 1` sampler (and only it: the `\lambda^2-1`
defect computed), giving `|\hat w - w| \le` score error times the
MODE-FISHER BUDGET, which for the two-point case is a dimensionless
`O(1)` constant concentrated at the merge scale `\sigma \approx a`:
one bit of mode identity, one unit of Fisher, spent in one window.
Still open there: the general-`K` merge-hierarchy theorem and the
window-localized empirical measurement the bound licenses.

**4. The metric gap.** The theory's currencies (KL, TV, `W_2`) and
generation's currency (perceptual/semantic quality) are known to
diverge; the guarantee chain (`distillation/04`) is complete in the
former and silent in the latter. Even a task-restricted theorem
connecting a proved divergence to a perceptual metric would change
what the whole theory can promise. Nothing exists.

**5. Discrete estimation.** CLOSED at the level posed, by retrofit:
`08` supplies phase F's discrete chapter. The discrete excess-risk
identity — masked-diffusion loss minus data entropy = the
order-averaged conditional KL, the discrete Vincent norm; the
estimation rate — per-context multinomial floor `(|V|-1)/2n_R`, with
the any-order model's conditional count exceeding a fixed order's by
EXACTLY `L(1+|V|^{-1})^{L-1}` (`\approx L` for large vocabularies,
exponential for small — the capacity cost of `E/04`'s constraint,
by counting); and the scheduling identity — the reveal tax is
`T = C - \sum_\ell I(x_\ell;\mathrm{pred}(\ell))`, serialization
buying back each token's predecessor information, with TC-optimal
scheduling a clean information maximization (`E/05` closed). Still
open there: the covering-number rate for a REAL denoiser (the phase-H
bridge, now with an explicit target), the ragged-context regime
`n \lesssim m\log m`, submodularity of the schedule objective, and
the uniform-chain (score-entropy) estimand where the telescope dies.

**6. Design against the ledgers.** CLOSED at the level posed, by
retrofit: `07` solves each dial in the Gaussian caricature. The
sampling grid: optimal step density `\propto \beta_t
e^{-\lambda^2B(t)/2}` (uniform in `\alpha^{\lambda^2}`), with
schedule-INVARIANT optimal error
`(1-(\alpha_T/\alpha_{t_{\min}})^{\lambda^2})^2/K`; the training
weight: the minimax-optimal `w^*` is THE SAME measure — `01`'s
propagator profile, its closing conjecture now a saddle-point
theorem; the interpolant: the CFM floor is the angular energy
`\int\rho^2\dot\theta^2`, uniquely minimized among VP paths by the
trigonometric schedule (`\pi^2/4`), and the endpoint DICHOTOMY — the
floor is finite iff `\gamma` vanishes faster than `\sqrt{1-t}`, the
score target is integrable iff slower: no path carries both
estimands into the data, and the bridge rate is the shared critical
point. Still open there: everything beyond the caricature, and the
joint (grid `\times` weight `\times` path) problem.

## What The Repository Claims To Have Settled

For balance, the closed column — results this ledger treats as done
mathematics: the forward calculus and its exact discreteness
(`foundations/01`); the estimand equivalences and their trainability
(the Bregman lemma closing Vincent/CFM/CT/score-entropy into one
theorem); reversal, the `\lambda`-family, and the quadrature
identity; the Girsanov decomposition and the
transport-vs-contraction dichotomy; the h-transform trilogy
(conditioning = measurements = rewards) with its Hopf–Cole bridge to
Bellman's Ledger; consistency's characterization and budget; the
discrete Anderson theorem, the ELBO collapse, and the any-order
equivalence; memorization, the KDE reduction, and the `\sigma^{-2}`
geometry. None of these will move; everything in the section above
is built on top of them.

## The Trilogy, Closed

Three repositories, one method: fix the coordinates, prove what is
provable in full, price every hypothesis, and let the open problems
be named rather than blurred. The recurring objects turned out to be
shared across all three — the exponential tilt (RLHF's closed form,
the soft Bellman operator, CFG and reward-tilted paths), the softmax
(attention's mixing, the empirical score, the Hopfield update, the
masked posterior), the conditional-expectation projection (every
trainable estimand in all three ledgers), and the
contraction-vs-transport dichotomy (Bellman operators, token
dynamics, sampler error). The mathematics of decisions, of
sequence models, and of generation is, to a degree the writing of
these ledgers made hard to miss, one subject — and its honest
frontier, in every volume, is the same: the theorems price the
components; the trained system's behavior at scale remains the
open integral.

## Load-Bearing Audit

```text
this file's claims       organizational, not mathematical: each
                         "proved" pointer resolves to a file whose own
                         audit governs; each "open" claim is falsified
                         the day someone closes it, which is the point;
the trilogy reading      a synthesis, honestly labeled: the shared
                         objects are theorems, the "one subject" is a
                         judgment.
```

## Position In The Coordinate System

The origin: `(P, s, S)` was the claim that diffusion methods differ
in three coordinates; forty-one files later the claim has carried
every theorem in the repository, and its residue — what the
coordinates do NOT determine, the trained system's emergent
behavior — is exactly the six-problem list above. A coordinate
system is judged by what it makes visible; this one is left to that
judgment.
