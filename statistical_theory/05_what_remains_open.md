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

**2. The guidance-times-solver theorem.** Flagged in five files: the
drift-error profiles are proved (`guidance/01–03`), the
transport-vs-contraction dynamics are proved (`A/05`, `01` here — the
propagator), the displacement constants are proved (`04`). Missing:
one composed bound — what a guided, discretized, `\lambda`-noised
sampler outputs, as a function of `(\omega, \lambda, K, t_{\min})`.
Every ingredient exists in this repository; the assembly is a paper
nobody has written.

**3. Mode weights.** `A/06` proved the mechanism (weights decided at
high noise, where merging modes make the score maximally
estimation-sensitive) and `01`'s propagator says high-noise errors
survive ODE sampling untouched. Missing: a bound on relative mode
masses under score error `\varepsilon_{sc}` — the quantity behind
every "diversity" claim, unpriced.

**4. The metric gap.** The theory's currencies (KL, TV, `W_2`) and
generation's currency (perceptual/semantic quality) are known to
diverge; the guarantee chain (`distillation/04`) is complete in the
former and silent in the latter. Even a task-restricted theorem
connecting a proved divergence to a perceptual metric would change
what the whole theory can promise. Nothing exists.

**5. Discrete estimation.** Phase E built the discrete objects and
their exact identities; phase F's estimation theory has no discrete
chapter — rates for masked-denoiser learning over `|V|^L` states,
the capacity cost of any-order consistency
(`discrete_diffusion/04`'s constraint), and TC-optimal reveal
scheduling (`05` there) are all open, all well-posed, and all
attackable with the tools already on the shelf (covering numbers
from attention-ledger's phase H, the Bregman lemma, the
total-correlation identity).

**6. Design against the ledgers.** Three dials recur without an
optimality theorem anywhere: the noise schedule / masking measure
(priced in `A/01`, fixed by the ELBO in `E/02`, free at sampling),
the loss weighting across noise levels (foundations/02, `01`'s
propagator as the missing criterion), and the interpolant's
`\gamma_t` (flow_matching/02, `04`'s blowup management). Each is a
one-dimensional design problem with its objective now computable
from this repository's theorems; none has been solved even in the
Gaussian caricature.

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
