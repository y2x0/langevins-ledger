# Memorization vs Generalization

## The Question

`score_foundations/06` proved the uncomfortable theorem: the training
objective's unrestricted optimum is the empirical score, and the
exact sampler built on it outputs training points. Yet trained
diffusion models generalize. This file closes the arc: a trichotomy
theorem — every non-memorizing sampler differs from the empirical
optimum through exactly three channels, each now quantified by an
earlier file — a proved toy for the interesting channel (restricted
function classes: generalization = the projection residual), and the
empirical phenomenology placed. The scheduled retrofit to
`score_foundations/06` lands with this file.

## The Trichotomy, Formalized

**Proposition.** Let a sampler be built from a score model
`\hat s` trained on `n` samples, run by any phase-A solver, stopped
at `t_{\min}`. Its output law differs from the memorizing law (the
empirical measure) only through:

```text
(a) smoothing      t_min > 0: even with s-hat = empirical score, the
                   output is the KDE at bandwidth sigma/alpha at
                   t_min — 02's reduction, exactly;
(b) estimation     s-hat != empirical score: capacity, optimization,
                   and regularization keep the model away from the
                   objective's minimizer — the inductive-bias channel;
(c) solver error   discretization and stochasticity (A/01's bias is
                   itself a smoother; injected noise likewise).
```

*Proof.* By definition of the three stages: with `t_{\min} = 0`,
`\hat s =` empirical score, and exact integration, the output is the
empirical measure (`score_foundations/06`'s memorization theorem);
each departure from that triple is one listed channel, and there are
no other inputs to the sampler. ∎

Trivial to prove and clarifying to state: **"why do diffusion models
generalize" decomposes into three separately-answerable questions**,
of which (a) is solved (bandwidth statistics, `02`), (c) is priced
(phase A), and (b) — the channel that actually distinguishes deep
models from KDEs — is the open frontier, with one solvable case:

## The Projection Theorem: Channel (b) In The Linear Case

**Theorem.** Restrict the score model to a linear class
`\hat s(x, t) = \Theta\,\phi(x, t)` (fixed features `\phi`, trained
weights `\Theta`). The population-of-the-empirical-objective
minimizer is the `L^2(\hat p_t)`-projection of the EMPIRICAL score
onto the feature span:

```math
\hat s^*_t = \Pi_{\mathrm{span}(\phi)}\ \big[\,\hat s^{\,\mathrm{emp}}_t\,\big],
```

and the sampler's deviation from memorization through channel (b) is
exactly the projection RESIDUAL — the component of the empirical
score (a sharp, `n`-spike attention field,
`score_foundations/06`) that the features cannot represent.

*Proof.* Vincent's objective (`score_foundations/02`) is an `L^2`
regression whose unrestricted minimizer is `\hat s^{\mathrm{emp}}`;
over a linear class, the minimizer of
`\mathbb{E}\|\Theta\phi - \text{target}\|^2` is the orthogonal
projection of the unrestricted minimizer onto the class (normal
equations; the Bregman lemma of `discrete_diffusion/03` at
`\varphi = \|\cdot\|^2`, restricted). ∎

The reading is the file's thesis: **generalization through channel
(b) is representational failure, oriented by the architecture.** A
smooth feature class cannot build the empirical score's
`O(\sigma^{-1})`-scale spikes around each training point (`04`'s
regularity ladder), so the projection returns a smoothed field — an
architecture-shaped kernel estimate, with the "bandwidth" set by the
features rather than the schedule. In the kernel/NTK reading of
training (statement), early stopping of OPTIMIZATION plays the same
role at finite width: gradient descent fits the smooth components of
the empirical score first, and stopping before the spikes is channel
(b) implemented by the training loop.

## The Phenomenology, Placed

The measured facts align with the trichotomy's reading, each cited as
statement: extraction attacks recover training images from models
trained on small/duplicated data (channel (b) too weak to smooth:
memorization observed — Carlini et al.); at large `n`, two models of
the same architecture trained on DISJOINT halves of a dataset
converge to near-identical samplers (Kadkhodaie et al.) — the
signature of an architecture-determined smoother dominating the
empirical fluctuations, i.e. channel (b) behaving like a stable
projection, exactly the theorem's picture at scale; and duplication
sits where the trichotomy puts it (duplicates concentrate empirical
mass the smoother cannot spread). The transition
`n_{\mathrm{mem}}(architecture, data)` between the regimes is
measurable, architecture-dependent, and untheorized beyond the linear
case above — the phase's honest frontier, named in `05`.

## Load-Bearing Audit

```text
unrestricted-optimum anchor   the trichotomy is relative to the
                              empirical score being THE minimizer —
                              Vincent's theorem, load-bearing;
linear class                  the projection theorem; deep networks
                              are not projections, and the NTK
                              reading is a regime statement, honestly
                              fenced;
L2(p-hat_t) geometry          which projection: the weighting over
                              noise levels enters the inner product —
                              foundations/02's weighting audit,
                              statistical edition;
disjoint-splits evidence      correlational support for the
                              projection picture at scale, not a
                              proof of it.
```

## Position In The Coordinate System

The estimand's finite-`n` truth: `s` as trained is neither the true
score (unreachable) nor the empirical score (undesirable) but an
architecture-projected smoothing of the latter — and the sampler's
celebrated creativity is, in this ledger's accounting, the projection
residual of channel (b) plus the bandwidth of channel (a). Retrofit
landed: `score_foundations/06`'s open interpolation question now has
its solved endpoints (KDE for (a), projection for (b)-linear) and its
open middle (deep (b)) on the record.

## What Remains Open

The deep channel-(b) theory — which smoother a given architecture
implements, and the memorization transition `n_{\mathrm{mem}}` as a
function of capacity (the field's most consequential open problem:
it is simultaneously the copyright question, the privacy question,
and the creativity question, and the linear theorem above is its
entire settled portion); per-sample memorization risk certificates
(detection is empirical; certification would need the projection
theory made quantitative); and the interaction of channels — solver
noise (c) as implicit regularization of (b) is folklore with no
statement.
