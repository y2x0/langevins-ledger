# Masked Diffusion Is Any-Order Autoregression

## The Question

Two schools of sequence generation: predict the next token
(autoregression) and fill in the blanks (masked diffusion). This file
proves they are the same family: the masked-diffusion ELBO equals the
EXPECTED ANY-ORDER autoregressive log-likelihood — an average of
chain-rule factorizations over uniformly random generation orders —
via one Beta-integral identity; and ordinary left-to-right AR is
recovered exactly as the degenerate deterministic-order schedule. The
theorem is the phase's centerpiece and the precise content of every
"diffusion LMs are order-agnostic AR models" slogan.

## The Two Objects

From `02`'s schedule-invariant form, the masked-diffusion objective
on a length-`L` sequence:

```math
\mathcal{L}_{\mathrm{MD}}
= \int_0^1 \frac{1}{u}\
\mathbb{E}_{S_u}\Big[\sum_{\ell\in S_u} -\log p_\theta\big(x^\ell\,\big|\,x_{\bar S_u}\big)\Big]\,\mathrm{d}u,
```

`S_u` the masked set (each position independently with probability
`u`), `x_{\bar S}` the visible complement. The any-order AR
objective: draw a uniformly random permutation `\sigma` of the `L`
positions, factorize by the chain rule in that order:

```math
\mathcal{L}_{\mathrm{AO}}
= \mathbb{E}_\sigma\Big[\sum_{k=1}^{L} -\log p_\theta\big(x^{\sigma(k)}\,\big|\,x^{\sigma(1)},\dots,x^{\sigma(k-1)}\big)\Big].
```

Both objectives are sums of terms "predict token `\ell` from visible
set `R`"; the theorem is that the two WEIGHTINGS over `(\ell, R)`
pairs are identical.

## The Theorem, Proved

**Theorem.** `\mathcal{L}_{\mathrm{MD}} = \mathcal{L}_{\mathrm{AO}}`
— term by term: for every position `\ell` and every subset `R` of the
other positions, both objectives weight the term
`-\log p_\theta(x^\ell\,|\,x_R)` by exactly

```math
\frac{|R|!\ \big(L-1-|R|\big)!}{L!}\,.
```

*Proof.* (Masked-diffusion side.) The term appears when `\ell` is
masked and the masked set among the OTHERS is exactly the complement
`S = \text{others}\setminus R`, `|S| = L-1-|R|`:

```math
\int_0^1 \frac1u\ \underbrace{u}_{\ell\ \text{masked}}\ u^{|S|}(1-u)^{|R|}\,\mathrm{d}u
= \int_0^1 u^{|S|}(1-u)^{|R|}\,\mathrm{d}u
= B\big(|S|+1,\ |R|+1\big)
= \frac{(L-1-|R|)!\ |R|!}{L!},
```

the `1/u` weight exactly cancelling `\ell`'s own masking probability,
and the Beta integral evaluating in factorials. (Any-order side.) The
term appears iff, in the random order, the set preceding `\ell` is
exactly `R`: among the `L!` orders, those arranging `R` first (in any
of `|R|!` ways), then `\ell`, then the rest (`(L-1-|R|)!` ways) —
probability `|R|!\,(L-1-|R|)!/L!`. Equal. ∎

The `1/u` measure — forced on `02` by the ELBO — is thereby
explained: it is precisely the measure under which independent
masking simulates uniform random orderings. Nothing was tuned; the
likelihood bound and the chain rule met in the middle.

**Corollary (AR as a degenerate schedule).** Replace independent
masking by the deterministic schedule that masks positions
`L, L-1, \dots` in order (a suffix mask of growing length): the
posterior reveal (`02`'s lemma) always uncovers the LAST masked
position, every term is "predict `\ell` from the exact prefix," and
the objective is the standard left-to-right AR likelihood — one
ordering, probability one. Left-to-right AR is masked diffusion with
a fixed order; masked diffusion is AR averaged over all orders. ∎

## What The Equivalence Does And Does Not Say

It DOES say: a masked-diffusion LM is trained to be a consistent
conditional model for EVERY (context, target) pattern — `2^{L}`-many
conditionals per sequence versus AR's `L` prefix-conditionals — and
its ELBO is an average of `L!` valid likelihood factorizations. Two
consequences follow at theorem level: infilling and
any-order decoding are native (each is one of the averaged
factorizations, already trained); and the model's ELBO can never beat
the best fixed order (Jensen: the average over orders is at least
the minimum over orders — for a model CAPABLE of matching the true
conditionals, all orders give the same value, the true likelihood;
for a finite model, order-averaging is a constraint, and the
observed perplexity gap to AR models is the constraint's price,
stated with the estimation-burden reading: the masked model spreads
its capacity over exponentially many conditionals, the AR model
concentrates on `L`).

It does NOT say the SAMPLERS are equivalent: the equivalence is
between training objectives. AR decoding follows one factorization
exactly; masked-diffusion decoding reveals several tokens per step
from a factorized proposal, and that approximation — absent from the
ELBO entirely — is `05`'s subject and tax.

## Load-Bearing Audit

```text
independent masking       the Beta integral's product form — the
                          proof-level meaning of "masking ratio";
the 1/u measure           from the ELBO (02), not chosen: the theorem
                          would FAIL for any other weighting — 02 and
                          04 are jointly rigid;
one shared network        "consistent across conditionals" presumes
                          one p_theta serving all patterns — the
                          weight-tying that makes any-order training
                          a constraint rather than a free lunch;
exact conditionals        the equal-for-all-orders claim is the
                          infinite-capacity clause, as usual.
```

## Position In The Coordinate System

The path coordinate's discrete degeneration, completed: `02` showed
only the masking-ratio measure matters; this file shows THAT measure
is uniform-random-orderings in disguise — the masking path IS
order-averaging, `P` has become a distribution over factorizations,
and AR sits at its degenerate corner. The attention-ledger bridge
sharpens accordingly: causal attention implements the corner;
bidirectional attention implements the average; the architectural
choice and the probabilistic choice are the same choice.

## What Remains Open

The capacity-allocation question the perplexity gap poses is now
ANSWERED at the lookup-table level (`statistical_theory/08`): the
any-order model must represent `L(1+|V|^{-1})^{L-1}` times as many
conditional cells as a fixed-order model — `\approx L` for large
vocabularies (the perplexity gap is mild for natural text),
exponential in `L` for small alphabets — and the excess-risk identity
there makes "the observed gap is the constraint's price" exact; what
stays open is the covering-number version for a real (non-table)
network; optimal ORDER learning (the theorem averages over
uniform orders; learned or data-adapted order distributions exit the
ELBO equivalence and their objectives are only partially understood —
statements); and whether the equivalence extends to the uniform
chain (it does not as stated — corrupted tokens are not absences, and
the chain-rule reading dies with the absorbing structure; what
replaces it is open).
