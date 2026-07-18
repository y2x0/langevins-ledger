# Discrete Estimation

## The Question

Cross-cutting problem 5 of `05`: phase E built the discrete objects
and their exact identities, and phase F's estimation theory has no
discrete chapter. This file supplies it, in three theorems that use
only tools already on the shelf. First, the discrete EXCESS-RISK
identity: the masked-diffusion training loss minus the data entropy
is exactly the order-averaged conditional KL — the discrete
"trainable norm," the Vincent-loss analogue that `statistical_theory/01`
is the continuous version of. Second, the estimation RATE: a masked
conditional over `|V|` symbols has per-context KL floor
`(|V|-1)/2n_R`, and the number of conditionals a masked model must
represent exceeds an autoregressive model's by exactly
`L(1+|V|^{-1})^{L-1}` — the capacity-allocation question of
`discrete_diffusion/04` and the ragged-context burden of `E/02`,
closed by counting. Third, the reveal SCHEDULE: the parallel-decoding
tax of `E/05` obeys the exact identity `T = C - \sum_\ell I(x_\ell;
\text{predecessors})`, so serialization buys back precisely each
token's mutual information with what precedes it, and TC-optimal
scheduling is a clean combinatorial problem with the serial schedule
as its unconstrained optimum.

## The Setup

Length-`L` sequences, vocabulary `|V|`, data law `p_0` on `V^L`
(entropy `H(x_0)`, all logs natural, "nats"). The masked-diffusion
objective in its any-order form (`discrete_diffusion/04`), with the
Beta weight `w(\ell, R) = |R|!\,(L-1-|R|)!/L!` on the term "predict
position `\ell` from the visible set `R`":

```math
\mathcal{L}(\theta)
\;=\;
\sum_{\ell=1}^{L}\ \sum_{R\subseteq [L]\setminus\{\ell\}} w(\ell, R)\
\mathbb{E}_{x_R\sim p_0}\big[-\log p_\theta(x^\ell\,|\,x_R)\big].
```

`\sum_R w(\ell, R) = 1` for each `\ell` (`04`'s permutation count), so
`\mathcal{L}` is an average of proper cross-entropies.

## Theorem 1: The Discrete Excess-Risk Identity

**Theorem.** For every denoiser `p_\theta`,

```math
\mathcal{L}(\theta) - H(x_0)
\;=\;
\sum_{\ell}\sum_{R} w(\ell, R)\
\mathbb{E}_{x_R}\Big[\mathrm{KL}\big(p_0(x^\ell\,|\,x_R)\ \big\|\ p_\theta(x^\ell\,|\,x_R)\big)\Big]
\;\;=:\;\; \mathcal{E}(\theta)\ \ge 0,
```

the order-averaged conditional KL. The minimum of the training loss
is the DATA ENTROPY, attained exactly at the true conditionals, and
the excess is the estimand error in the only norm training controls.

*Proof.* Add and subtract the true conditional inside each term:

```math
\mathbb{E}_{x_R}\big[-\log p_\theta(x^\ell|x_R)\big]
= \underbrace{\mathbb{E}_{x_R}\big[H(x^\ell|x_R{=}\cdot)\big]}_{H(x^\ell|x_R)}
+ \mathbb{E}_{x_R}\,\mathrm{KL}\big(p_0(x^\ell|x_R)\,\|\,p_\theta(x^\ell|x_R)\big).
```

Sum against `w`. The KL part is `\mathcal{E}(\theta)`. The entropy
part is `\sum_\ell\sum_R w(\ell, R)\,H(x^\ell|x_R) =
\mathbb{E}_\sigma\big[\sum_k H(x^{\sigma(k)}|x^{\sigma(1)},\dots,
x^{\sigma(k-1)})\big]` (the Beta weight IS the uniform-order
predecessor probability, `04`), and for EVERY fixed order the inner
sum telescopes to `H(x_0)` by the entropy chain rule; averaging over
orders leaves `H(x_0)`. ∎

This is the discrete interface to phase A's discrete cousin, and the
exact analogue of the continuous story: `\mathcal{E}(\theta)` is the
weighted estimand error that `E/02`'s ELBO gap equals and that
`E/05`'s decoding tax adds to — the KL from data to samples is at
most `\mathcal{E}(\theta)` (estimation) plus `\sum_j
\mathbb{E}[\mathrm{TC}(x_{S_j}|c_j)]` (the discretization/parallelism
tax, Theorem 3), the token-space form of `A/03`'s "score error plus
discretization." One reading is worth stating: the Bregman lemma
(`discrete_diffusion/03`) says each conditional is learnable by
regression toward `\mathbb{E}[\mathbf{1}_{x^\ell}|x_R]`; Theorem 1
says the price of imperfect learning aggregates as a KL, at the
uniform-order weight — projection and its cost, both discrete.

## Theorem 2: The Estimation Rate And The Capacity Count

Now bound `\mathcal{E}` for the honest nonparametric estimator: the
lookup table (histogram) that stores, per realized context, the
empirical symbol frequencies. Two pieces — the per-context floor
(exact) and the number of contexts (exact) — assemble to the rate.

**Lemma (per-context KL floor).** Fix `(\ell, R)` and a context value
`x_R = v` realized `n_v` times. Estimate `p_0(\cdot|v)` over `|V|`
symbols by the empirical frequencies `\hat p`. Then to leading order

```math
\mathbb{E}\,\mathrm{KL}\big(p_0(\cdot|v)\,\|\,\hat p\big)
= \frac{|V|-1}{2\,n_v} + o\!\big(n_v^{-1}\big),
```

with the leading constant EXACT: the `n_v`-scaled term is Pearson's
statistic, whose expectation is `|V|-1` identically.

*Proof.* Write `\hat p = p_0 + \delta`, `\mathbb{E}\delta = 0`,
`\mathrm{Cov} = \tfrac1{n_v}(\mathrm{diag}\,p_0 - p_0 p_0^\top)`.
Second-order Taylor of KL at `\delta = 0` (first order vanishes,
Fisher metric): `\mathrm{KL}(p_0\|p_0+\delta) = \tfrac12\sum_a
\delta_a^2/p_{0,a} + o(\|\delta\|^2)`. Take expectations:
`\sum_a \mathbb{E}\delta_a^2/p_{0,a} = \tfrac1{n_v}\sum_a
(1-p_{0,a}) = (|V|-1)/n_v`, using `\mathbb{E}\delta_a^2 =
p_{0,a}(1-p_{0,a})/n_v` — this sum is exactly `\mathbb{E}[\chi^2]/n_v`
with `\chi^2` Pearson's, whose mean is `|V|-1` for any `n_v`. ∎

**Lemma (the conditional count).** The number of distinct conditional
cells a model over `V^L` must specify is

```math
N_{\mathrm{any}} = L\,(1+|V|)^{L-1}\quad(\text{any-order/masked}),
\qquad
N_{\mathrm{AR}} = \frac{|V|^{L}-1}{|V|-1}\quad(\text{fixed order}),
```

counting each `(\text{position}, \text{context pattern},
\text{context values})` once. Their ratio is exactly

```math
\frac{N_{\mathrm{any}}}{N_{\mathrm{AR}}}
\ \xrightarrow{\ |V|^{L}\gg 1\ }\
L\Big(1+\tfrac1{|V|}\Big)^{L-1}
\;=\;
\begin{cases}
L\cdot e^{L/|V|}(1+o(1)) & L\ll|V|\ \text{(overhead} \approx L)\\[2pt]
\text{exponential in } L & L\gg|V| .
\end{cases}
```

*Proof.* Any-order: position `\ell` (`L` choices), a context set `R`
among the other `L-1` positions with any values, contributing
`\sum_{k=0}^{L-1}\binom{L-1}{k}|V|^k = (1+|V|)^{L-1}` cells. Fixed
left-to-right order: position `k` conditions on the length-`(k-1)`
prefix, `|V|^{k-1}` values; sum the geometric series. The ratio's
leading term drops the `-1` in `N_{\mathrm{AR}}` and factors
`(1+|V|)^{L-1} = |V|^{L-1}(1+|V|^{-1})^{L-1}`. ∎

**Theorem (the assembled rate, well-sampled regime).** If every
realized context is seen `\Omega(1)` times (the coupon-collector
threshold `n \gtrsim m\log m`, with `m` the number of realized
contexts), the table estimator achieves

```math
\mathcal{E}(\hat\theta)
\;=\;
\frac{|V|-1}{2n}\ \sum_\ell\sum_R w(\ell, R)\,m_{\ell,R}
\;+\;\text{l.o.t.},
```

`m_{\ell,R}` the number of realized values of `x_R` — the weighted
effective conditional count, over `n`. The masked model's rate
exceeds a fixed-order model's by the count ratio above: `\approx L`
when the vocabulary is large, exponential when it is small.

*Proof.* Sum the per-context floor against `w`, with
`\mathbb{E}_{x_R}[\cdot] = \sum_v p_0(v)\,\mathrm{KL}_v` and expected
count `n_v = n\,p_0(v)`: each realized context contributes
`p_0(v)\cdot(|V|-1)/(2 n p_0(v)) = (|V|-1)/(2n)`, so the value-average
returns `(|V|-1)\,m_{\ell,R}/(2n)`. ∎

The three results together ARE the discrete `F/02`: the multinomial
floor is the discrete bias–variance balance (no bandwidth — the
symbol count `|V|-1` plays the variance role, the context count `m`
the effective dimension); the count ratio is the exact price of
`04`'s "capacity spread over exponentially many conditionals," now
`L(1+|V|^{-1})^{L-1}` rather than a slogan; and the collapse to
`\approx L` for `|V|\gg L` is why the any-order perplexity gap is
mild for large-vocabulary natural text and severe for small-alphabet
(e.g. binary, genomic) data — a testable prediction. Covering-number
control of a real (non-table) denoiser class is where
attention-ledger's phase-H machinery enters, replacing `m` by a
capacity that its inductive bias makes far smaller — the bridge
`03` names, now with an explicit target quantity.

## Theorem 3: TC-Optimal Reveal Scheduling

`E/05` proved the per-step tax is a conditional total correlation and
posed TC-optimal scheduling as open. Here is its exact objective. A
schedule is an ordered partition of `[L]` into blocks
`S_1,\dots,S_k`; block `j` is revealed given `c_j = x_{S_1\cup\cdots
\cup S_{j-1}}`; total tax `T = \sum_j \mathrm{TC}(x_{S_j}|c_j)`. Let
`C := \sum_\ell H(x_\ell) - H(x_0)` be the sequence's total
correlation (multiinformation).

**Theorem (the scheduling identity).**

```math
T \;=\; C \;-\; \sum_{\ell=1}^{L} I\big(x_\ell\,;\,\mathrm{pred}(\ell)\big),
```

where `\mathrm{pred}(\ell)` is the set of tokens in blocks strictly
before `\ell`'s. Serialization buys back exactly each token's mutual
information with its predecessors. Consequences:

```text
fully serial (k=L):   pred(pi(m)) = all earlier tokens; sum I
                      telescopes to C; T = 0 — the unique zero-tax
                      schedule (E/05's serial exactness, re-derived);
one-shot (k=1):       pred = empty; T = C — the maximum, the full
                      multiinformation (E/05's two-token 1 bit);
budget k:             min tax = C - max over ordered k-partitions of
                      sum_ell I(x_ell; pred(ell)) — a monotone,
                      SUBMODULAR-flavored gain to maximize.
```

*Proof.* Expand `T = \sum_j\big[\sum_{\ell\in S_j}H(x_\ell|c_j) -
H(x_{S_j}|c_j)\big]`. The second sum telescopes by the chain rule:
`\sum_j H(x_{S_j}|c_j) = H(x_0)` (blocks partition the sequence,
contexts are the running unions). In the first,
`H(x_\ell|c_j) = H(x_\ell|\mathrm{pred}(\ell))` for `\ell\in S_j`.
So `T = \sum_\ell H(x_\ell|\mathrm{pred}(\ell)) - H(x_0)`. Subtract
and add `\sum_\ell H(x_\ell)`: `T = C - \sum_\ell[H(x_\ell) -
H(x_\ell|\mathrm{pred}(\ell))] = C - \sum_\ell I(x_\ell;
\mathrm{pred}(\ell))`. ∎

**Worked example (a Markov chain).** `x_1-x_2-\cdots-x_L` a stationary
chain, per-edge mutual information `I` (so `C = (L-1)I`; higher-order
total correlation vanishes by the Markov property). Compare two
2-block schedules at `k=2`:

```text
adjacency-respecting  S_1 = odds, S_2 = evens: every even token's
(interleaved):        predecessors include BOTH neighbors, so
                      sum I(x_ell; pred) = (L-1)I = C in the large-L
                      limit (each internal even sees two edges' worth,
                      each odd contributes at reveal-2 nothing... )
                      -> tax -> 0 as the block structure captures
                      every edge across the two rounds;
contiguous halves:    S_1 = first L/2, S_2 = second half: only the
                      single cut edge is paid inside no block — the
                      one edge whose endpoints are revealed together
                      in S_1 or S_2 loses its cross-block credit;
                      tax ≈ I (one edge), independent of L.
```

The identity makes the design rule exact and matches intuition:
reveal to MAXIMIZE the dependency each token shares with what is
already revealed; block boundaries are cheap exactly where
`I(x_\ell;\mathrm{pred})` is already large (weak cross-block
dependence), which is `E/05`'s "cut where dependence is local," now
an optimization over an explicit information functional. Confidence-
based heuristics are its greedy surrogate: revealing the highest-
`I`-with-context token first is one step of maximizing the sum.

## Load-Bearing Audit

```text
independent masking /       Theorem 1's weight IS 04's Beta count;
the 1/u measure             any other weighting breaks the entropy
                            telescope and the identity's clean floor;
entropy chain rule          Theorems 1 and 3 both — the telescope that
                            makes the minimum the data entropy and the
                            tax a multiinformation defect;
small-perturbation (T2)     the floor is the LEADING term (Fisher
                            metric); the exact object is the Pearson
                            mean, so the constant |V|-1 is not
                            asymptotic — the o(1/n) is the higher
                            cumulants;
forward KL / smoothing      Theorem 1's excess is KL(truth||model):
                            a raw table assigns 0 to unseen symbols
                            and infinite risk — the floor holds for a
                            Krichevsky-Trofimov/add-1/2 estimator, and
                            the leading constant is unchanged (KT
                            redundancy = (|V|-1)/2 · log n / n per
                            context in the worst case; the well-sampled
                            floor is the second-order term) — stated
                            honestly, the smoothing is load-bearing;
well-sampled regime         the assembled rate ASSUMES n ≳ m log m;
                            below it the ragged-context tail (contexts
                            seen o(1) times, model reverts to its
                            prior, O(1) risk each) dominates — this IS
                            E/02's ragged-context burden, and the
                            table estimator is minimax-bad there:
                            structure (a real denoiser) is not a
                            convenience but a necessity, quantified;
counting = capacity         the count is the TABLE parameter measure;
                            a structured class's effective capacity is
                            its covering number (attention-ledger H) —
                            the count is the ceiling the architecture
                            must beat, not the operative rate;
TC = independence tax only  Theorem 3 credits the marginals as exact
                            (E/05's audit); estimation error (T1-2)
                            adds on top — the two discrete error
                            sources, never entangled in the proofs.
```

## Position In The Coordinate System

The estimand's statistical price on token spaces: `\mathcal{E}` is
the discrete Vincent norm (Theorem 1), its rate is multinomial-times-
context-count (Theorem 2), and the solver's remaining freedom — the
reveal schedule — has an exact information objective (Theorem 3).
Phase F now spans both worlds with one structure: an excess-risk
identity that makes the loss minus an entropy the estimand error
(`01` continuous, `08` discrete), a rate that is the oldest estimator
in the relevant book (KDE bandwidth there, multinomial cells here),
and a design freedom priced by a propagator (`07`) or a
multiinformation (`08`). The `(P, s, S)` coordinates carried the
discrete objects exactly as far as the continuous ones.

## What Remains Open

The covering-number rate for a real (transformer) masked denoiser —
plugging attention-ledger phase H's capacity bounds into Theorem 1 in
place of the table count, the concrete unexploited bridge (`03`);
the ragged-context regime `n \lesssim m\log m`, where the table fails
and the minimax rate is governed by the data's context-occupancy
distribution (a discrete cousin of manifold-support rates, `F/02`'s
riders); the budgeted TC-optimal schedule as a submodular
maximization — whether `\sum_\ell I(x_\ell;\mathrm{pred})` is
submodular in the block structure (it would make the greedy
confidence heuristic a `1-1/e` approximation, promoting folklore to a
guarantee); the score-entropy (uniform-chain) estimand, where the
any-order/entropy-telescope machinery dies (`04`'s fence) and the
excess-risk object is a ratio-matching Bregman divergence
(`03`) whose rate theory is unbuilt; and the same metric gap that
stops every phase-F rate short of sample QUALITY, here between the
ELBO-KL these theorems bound and generation.
