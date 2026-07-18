# What Flow Matching Is Not

## The Question

Flow matching arrived wrapped in optimal-transport language —
"OT paths," straight lines, displacement interpolation — and the
vocabulary outruns the theorems. This file draws the fence from both
sides: two positive results proved (in one dimension the flow map IS
optimal transport; the family's cost theorem from `03` holds
generally), two negative results proved (the marginal PATH of
independent-coupling FM is not the OT geodesic — worked example with
atoms and numbers; straightness, the actual fixed-point property, is
weaker than optimality), and the minibatch-OT literature placed as
statements. The purpose is calibration: FM is an excellent
regression-based sampler family whose OT content is real but
specific, and smaller than the branding.

## Positive, Proved: One Dimension Is Optimal

**Theorem.** In `d = 1`, under `01`'s well-posedness hypotheses
(densities, unique flow), the FM transport map `T: z_0 \mapsto z_1`
is nondecreasing — trajectories of a 1-D ODE cannot cross — and a
monotone map between given marginals is THE optimal transport map for
every convex cost (the classical rearrangement theorem, cited). So in
one dimension, flow matching from ANY coupling with a well-posed flow
delivers optimal transport, whatever coupling it started from.

*Proof of the monotonicity.* Two solutions of the same
scalar ODE with `z_0 < z_0'` satisfy `z_t < z_t'` for all `t`:
equality at some first time would violate uniqueness through that
point. A pointwise limit of order-preserving maps preserves order on
the support; hence `T` is nondecreasing. ∎

This theorem is why small examples mislead: every 1-D experiment
"confirms" the OT story, and the story is TRUE there — for the
structural reason that non-crossing and monotonicity coincide in one
dimension and monotonicity is 1-D optimality. None of that survives
`d \ge 2`, where non-crossing is cheap and optimality is not.

## Negative, Proved: The Path Is Not The Geodesic

Displacement interpolation (the OT geodesic) between `p_0` and `p_1`
moves mass along the OPTIMAL coupling; FM's path moves it along the
COUPLING IT WAS GIVEN. These differ even when the endpoints coincide:

**Counterexample (exact, with atoms and numbers).** Let
`p_0 = p_1 = \tfrac12\delta_{-1} + \tfrac12\delta_{+1}` in `d = 1`,
independent coupling. The OT coupling is the identity; its geodesic
is CONSTANT: `p_t^{OT} = p_0` for all `t`, cost `0`. The FM path is
the law of `(1-t)x_0 + t x_1` with independent signs:

```math
p_t^{FM}
= \tfrac14\,\delta_{-1} + \tfrac14\,\delta_{+1}
+ \tfrac14\,\delta_{1-2t} + \tfrac14\,\delta_{2t-1},
```

which at `t = \tfrac12` puts mass `\tfrac12` at the ORIGIN — a point
neither endpoint charges. The path bulges through configurations the
transport problem never visits; its kinetic energy is
`E|x_1 - x_0|^2 = 2` against the geodesic's `0`. ∎

The independent coupling is the mechanism, not the atoms: smooth the
atoms and the same bulge appears in mollified form. "OT path" in the
FM literature names the per-PAIR straight line — conditional
displacement interpolation between a point and a point, which is
trivially the OT geodesic BETWEEN THOSE TWO POINTS — not optimality
of anything marginal. The phrase is not wrong; it is about a
different, much smaller statement than the reader assumes.

## Negative, Fenced: Straight Is Not Optimal

`03` proved rectification's fixed points are the straight couplings
and that every convex cost weakly decreases along the loop. The
tempting completion — "therefore rectification converges to OT" — is
FALSE in general for `d \ge 2`: straightness is a NON-CROSSING-type
property (each particle travels its chord), while `W_2`-optimality is
cyclical monotonicity — a strictly stronger global constraint; the
straight couplings form a large set containing the optimal one, the
rectification map is only guaranteed to land IN the set, and Liu et
al. are explicit that reflow targets straightness (= fast sampling),
not optimality (statement, with their separating examples). The
correct summary of `03` + this file:

```text
rectification provably buys   one-step samplability (straightness)
                              and weak cost improvement;
rectification does not buy    the OT map, the OT cost, or the
                              geodesic path — except in d = 1, where
                              the positive theorem above collapses
                              the distinction.
```

**Minibatch OT (statements).** Replacing the independent coupling by
per-batch OT pairings (Pooladian et al., Tong et al.) biases the
coupling toward optimality with quantified batch-size effects: the
implied coupling converges to OT as the batch grows, with the
variance/bias tradeoffs their papers price; at practical batch sizes
it is a straightening accelerant (fewer reflows, shorter chords)
rather than an OT guarantee. Consistent with, not contradicting, the
fences above.

## Why The Fence Matters

Because the OT reading suggests properties FM does not have:
canonical pairings (creative applications — editing, interpolation,
"natural" correspondences — inherit the coupling's arbitrariness, not
OT's canonicity), path uniqueness (different couplings, different
flows, same marginals), and cost optimality of the learned map.
What FM actually guarantees is `01`–`04`: exact trainable velocity
for any chosen path, full solver freedom, and — via rectification —
convergence toward one-step samplability. Those are the theorems;
they are enough; the extra story is loanwords.

## Load-Bearing Audit

```text
uniqueness (1-D theorem)     non-crossing IS uniqueness; the atomic
                             crossing case of 03 breaks both;
rearrangement theorem        cited classical input for 1-D optimality;
independent coupling         the bulge counterexample's mechanism —
                             better couplings shrink the bulge, which
                             is exactly the minibatch-OT program;
d >= 2 for the gap           straight-vs-optimal separation needs
                             room to be non-crossing yet cyclically
                             non-monotone; in d = 1 the fence closes.
```

## Position In The Coordinate System

The path coordinate's honest boundary: `P` is a free design choice
with exact theorems (`01`–`04`) and a transport-flavored improvement
loop (`03`); it is not a solution of the Monge problem, and the
repository's contract — say what the sampler provably outputs —
requires the distinction stated once, sharply, here.

## What Remains Open

Whether OPTIMALITY (not just straightness) is reachable by any
simulation-free training loop at scale — minibatch-OT with growing
batches says yes asymptotically, at unpriced cost; a rate for the
straight-vs-optimal gap under rectification from generic couplings
(is the gap typically small even if never zero?); and whether
optimality would even help generation — the sampler needs
straightness for speed and marginal fidelity for correctness, and no
result says the OT coupling improves either at fixed compute. The
fence may enclose a question nobody needs answered; proving THAT
would itself be a contribution.
