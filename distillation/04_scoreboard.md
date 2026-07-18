# The Distillation Scoreboard

## The Question

One-step generation is the family's destination and marketing's
favorite word. This closing file separates the columns: what is
proved (a guarantee CHAIN from data to student, each link priced by
an earlier file), what is empirical (the adversarial and
architectural post-training that currently wins benchmarks), and
what the guided-teacher question — deferred here by
`guidance_and_control/05` and `score_foundations/05` — actually
resolves to.

## The Guarantee Chain

Assembled from the repository, link by link, each with its price tag:

```text
data --(score training)--> score error eps_sc
        priced by: Vincent's objective = the Girsanov term
        (foundations/02, samplers/03); estimation theory: phase F;

score --(teacher sampler)--> teacher output law
        priced by: A/03-04 (KL <= prior + eps_sc^2-integral +
        discretization); deterministic teachers add A/02's
        quadrature error, transported not contracted (A/05);

teacher --(distillation)--> one-step student map
        priced by: 01's linear accumulation (consistency), 02's
        fixed-point/bias theorems (CD inherits solver order; CT's
        Jensen-gap constant), 03's round recursion (progressive);

student map --(evaluation)--> student output law
        free: pushforward of the prior through the learned map —
        no further stochastic analysis needed, which is the
        analytical PAYOFF of one-step: the last link, for once,
        costs nothing.
```

The chain's moral is worth one plain sentence: **a distilled model's
guarantee is never better than teacher-quality plus
distillation-budget, and every term in both is already a theorem in
this repository** — there is no separate "distillation theory" to
wait for; there is constant-tracking nobody has done end to end.

## What Is Empirical, Kept Honest

The current best one-step and few-step systems add ingredients the
chain does not cover, and the scoreboard should say so plainly:

```text
adversarial post-training     GAN-style losses on top of consistency
                              or PD students: improves perceptual
                              metrics, exits every guarantee above
                              (the student's law is no longer tied to
                              the teacher's by any bound; statements);
continuous-time CMs           sCM/TrigFlow-style reformulations:
                              cleaner parametrizations of 01's
                              transport equation with better
                              conditioning — engineering ON the
                              theory, consistent with it (statements);
score/distribution matching   distribution-level objectives (score
                              distillation, moment matching): replace
                              the map-matching target by a law-
                              matching one — different geometry,
                              partial theory, outside this file's
                              chain (statements).
```

None of this is a criticism: the chain bounds laws in KL-like
divergences, benchmarks measure perceptual quality, and the two are
known to diverge (`samplers/04`'s reading 5). The honest statement is
that PROVED one-step sampling exists (the chain), and STATE-OF-THE-ART
one-step sampling is currently paid for with unpriced components.

## The Guided-Teacher Question, Resolved

Deferred twice, answered now. Distilling a CFG teacher looks
paradoxical after `score_foundations/05`: the guided family is not a
diffusion path of any data law — what is the student even learning?
Resolution, in two proved observations:

**(i) The guided flow map exists.** CFG defines a bona fide vector
field (`s_\omega` is a smooth function; the guided PF-ODE is a
well-posed ODE under the same regularity as ever), so its solution
map `f^*_\omega` exists and `01`'s characterization applies verbatim:
consistency distillation of a guided teacher is EXACTLY as
well-founded as of an unguided one. Nothing in phase-D theory used
the field being a score.

**(ii) What is undefined is the LAW's pedigree, not the map.** The
student faithfully samples (pushforward of the prior through)
`f^*_\omega` — the same law the guided teacher samples. The open
question was and remains foundations/05's: characterizing THAT law
relative to the data. Distillation neither worsens nor resolves it;
it inherits it. The practical corollaries follow: an
`\omega`-conditioned student is learning a FAMILY of flow maps
(dimension added to the estimand, cost priced by capacity, no new
theory needed); and "CFG-free" distilled models that bake in one
`\omega` have simply chosen a point of the family — their
diversity-quality position is `guidance/02`'s frontier, frozen.

So the scoreboard entry reads: guided distillation — theoretically
sound as map-learning (proved), semantically open as sampling
(inherited, foundations/05), empirically standard (statements).

## The Comparison Table

```text
method            guarantee                   binding constant
many-step SDE     A/03-04: KL bound, robust   eps_sc + sqrt-h terms
many-step ODE     A/02: quadrature error      denoiser curvature
                  (transported, A/05)
consistency (CD)  01+02: teacher map + N-leak solver order x length
consistency (CT)  01+02: same order,          network curvature x
                  teacher-free                posterior variance
progressive       03: exact-linear base +     absorbed curvature vs
                  round accumulation          capacity
+ adversarial     none                        (benchmarks)
```

One column deserves the last word: every deterministic row's binding
constant is the SAME OBJECT — the denoiser's variation along the
noise clock — surfacing as quadrature error, map regularity, or
absorbed curvature. Phase D's summary in one line: few-step
generation is possible exactly to the extent that
`\hat x_0` is smooth along trajectories, and that is a property of
the DATA (`samplers/02`), estimated by the network, spent by the
solver or the student.

## Load-Bearing Audit

```text
KL-vs-perceptual        the chain's currency mismatch with the
                        scoreboard's — stated, unresolved, and the
                        reason "proved" and "best" name different
                        systems;
teacher floor           all distillation rows sit on the teacher's
                        A-phase error; "the student beat the teacher"
                        claims involve the unpriced components, by
                        construction;
capacity                invisible to every theorem in the phase and
                        decisive in practice — the audit item this
                        family shares with attention-ledger's
                        universal-approximation file: existence
                        theorems, exponential fine print.
```

## Position In The Coordinate System

Phase D closes the solver coordinate: `S` now ranges from
thousand-step SDEs to a single learned map, with one governing
smoothness quantity and a complete (if constant-untracked) guarantee
chain. What the repository has NOT yet priced is the first link —
`\varepsilon_{sc}` itself, the learned score's distance from truth —
and that is phase F, after the discrete world of phase E.

## What Remains Open

End-to-end constant tracking through the chain for one real
configuration (every link is a theorem; their composition has never
been evaluated numerically against a trained system — a paper waiting
to be written); a lower bound for one-step maps (03's open item, the
family's missing floor); the law-level theory of
distribution-matching distillation (the one route that might evade
the teacher floor, currently the least theorized); and the
perceptual-metric gap, which this repository can name but not close.
