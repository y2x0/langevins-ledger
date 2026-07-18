# The Scoreboard: Masked Diffusion vs Autoregression

## The Question

`04` proved the training objectives are one family; the SAMPLERS are
not, and the practical case for diffusion LMs — parallel decoding —
lives exactly in the difference. This closing file proves the tax:
revealing several tokens in one step samples them independently given
the context, and the per-step cost is EXACTLY the total correlation
of the revealed set — worked to numbers on a two-token example where
half the sampled mass lands on impossible sequences. With the tax
priced, the scoreboard against AR can be honest, including the
attention-ledger serial-ceiling bridge.

## The Parallel-Decoding Tax, Proved

A reverse step reveals the token set `S` given visible context `c`.
The true joint is `p(x_S\,|\,c)`; the factorized denoiser proposal is
`\prod_{\ell\in S} p(x^\ell\,|\,c)` — correct marginals
(`02`–`03`: each factor is the exact posterior), independent
sampling.

**Theorem.** The per-step error is exactly the conditional total
correlation of the revealed set:

```math
\mathrm{KL}\Big(p(x_S|c)\ \Big\|\ \prod_{\ell\in S}p(x^\ell|c)\Big)
\;=\;
\sum_{\ell\in S} H\big(x^\ell\,\big|\,c\big)\;-\;H\big(x_S\,\big|\,c\big)
\;=:\;
\mathrm{TC}\big(x_S\,\big|\,c\big)\ \ge 0,
```

zero iff the revealed tokens are conditionally independent given the
context. *Proof.* Expand the KL:
`E_p[\log p(x_S|c)] - \sum_\ell E_p[\log p(x^\ell|c)]` — the
definition of the entropy difference. ∎

**Worked to numbers.** Two tokens, data uniform on
`\{(A,B), (B,A)\}`, both masked (`c` empty). Marginals: each token
`50/50` between `A` and `B` — the factorized proposal samples the
four pairs uniformly:

```text
one-step parallel:  (A,B) 1/4   (B,A) 1/4   (A,A) 1/4   (B,B) 1/4
                    HALF the mass on sequences of probability zero;
                    TC = H(x1)+H(x2)-H(x1,x2) = 1 + 1 - 1 = 1 bit —
                    the theorem's tax, exactly the KL incurred;
two-step (reveal    step 1 samples token 1 from its true marginal,
one, then the       step 2 samples token 2 from its true conditional
other):             — EXACT. Serialization costs a step and buys the
                    bit back.
```

The general accounting follows by summing the theorem over steps: a
`k`-step schedule revealing sets `S_1, \dots, S_k` incurs total KL
`\sum_j E[\mathrm{TC}(x_{S_j}|c_j)]` — **the price of parallelism is
the dependence structure revealed-per-step**, and fully serial
decoding (`k = L`, singletons) is the unique zero-tax schedule, which
is AR decoding in a (possibly adaptive) order. Speed and exactness
trade at the exchange rate TC-per-step; remasking/corrector variants
(re-mask low-confidence tokens and retry) spend extra steps to buy
back tax already paid — the discrete corrector, same logic as
`samplers_and_convergence/06` (statements for the specific schemes).

## The Attention-Ledger Bridge, Closed

attention-ledger's expressivity phase proved the transformer forward
pass is constant-depth parallel computation, with genuinely serial
tasks (its `S_5`/automaton witness) requiring decode steps — the
serial ceiling. Masked-diffusion decoding does not evade that
theorem: each reveal step is one forward pass, so `k`-step decoding
has `k` serial rounds, and tasks needing `\Omega(L)` serial
dependency (each token computationally determined by the previous)
force either `k \approx L` (no speedup) or a total-correlation tax
that is not small — for deterministic chains, TC of a jointly
revealed pair is the full entropy of the dependent token. The honest
synthesis:

```text
diffusion LMs parallelize SAMPLING BREADTH (weakly dependent tokens
revealed together at low tax), not COMPUTATIONAL DEPTH (attention-
ledger's ceiling is untouched); their win condition is data whose
conditional dependence is local/shallow — which natural text often
is, and reasoning chains often are not.
```

## The Scoreboard

```text
axis                 AR                      masked diffusion
likelihood           concentrated capacity:  order-averaged (04):
                     L conditionals, best    native ELBO gap;
                     current perplexities    observed gap real
conditioning         prefix only; infilling  any pattern, natively
                     by heuristics           trained (04)
decoding cost        L serial steps          k steps + TC tax
                                             (theorem above)
serial computation   n steps available       same ceiling per
(attention E/07)     natively                step; no free depth
error dynamics       exact sampling of its   per-step tax, plus
                     own factorization       remasking correctors
guidance/control     logit steering, RLHF    phase-C machinery
                     (Bellman's ledger)      transfers via 01's
                                             reversal (statements)
```

Neither column dominates; the table's content is that every cell is
now a theorem or a priced statement rather than a vibe — which was
this phase's contract.

## Load-Bearing Audit

```text
exact per-token posteriors   the theorem taxes ONLY the independence,
                             crediting the marginals as exact; a
                             trained denoiser adds estimation error
                             on top (phase F's discrete open item);
context = all visible        the factorization is conditional on the
                             full bidirectional context — the tax
                             would be far worse for weaker
                             conditioning;
schedule adaptivity          confidence-based reveal orders make S_j
                             data-dependent: the theorem still
                             applies per-step (TC of the chosen set);
                             adaptive schedules are TC-minimization
                             heuristics, unproved as such.
```

## Position In The Coordinate System

The solver coordinate on discrete spaces, priced: `S` ranges from
fully serial (AR, exact, slow) to aggressively parallel (taxed by
conditional total correlation, quantified), with correctors buying
back tax at step cost. Phase E closes with the three coordinates all
discrete-instantiated: masking paths (`01`–`02`), posterior
estimands (`02`–`03`), and reveal schedules (here) — and the
attention bridge run in both directions: attention computes the
denoiser; the denoiser's objective is attention-ledger's MLM; the
ceiling on serial computation binds both.

## What Remains Open

TC-optimal reveal schedules (the theorem defines the objective —
minimize summed conditional total correlation under a step budget —
an information-theoretic scheduling problem nobody has attacked
directly; confidence heuristics are its greedy shadow); estimation
rates for the discrete denoiser (F's missing discrete chapter);
principled hybrid decoders (AR over blocks, diffusion within — the
tax theorem says exactly when block boundaries are cheap, unused);
and the reasoning question the bridge sharpens: whether
chain-of-thought-style serial computation can coexist with parallel
reveal — the current answer is architectural folklore, and the
ceiling theorem makes it precise enough to attack.
