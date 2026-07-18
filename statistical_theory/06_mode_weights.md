# Mode Weights

## The Question

Cross-cutting problem 3 of `05`: bound the error in RELATIVE MODE
MASSES — the quantity behind every diversity claim — as a function of
score error. `samplers_and_convergence/06` proved the mechanism
(weights are decided at high noise); this file proves the bound. The
route: the mode posterior is a normalized h-function
(`guidance_and_control/01`), it is a reverse-time martingale under
the `\lambda = 1` sampler AND ONLY under it (a computation with a
clean `\lambda^2 - 1` defect), and its gradient — the mode-Fisher
density — integrates over the whole path to an `O(1)` budget
concentrated at the merge scale. Mode weights cost one bit; the bill
is presented in a specific noise window.

## The Object And The Martingale

Mixture data `p_0 = \sum_k w_k\,p_0^{(k)}`; each component and the
mixture evolve under the same forward process, so
`p_t = \sum_k w_k p_t^{(k)}` for all `t`. The mode posterior:

```math
\pi_k(x, t) \;=\; \mathbb{P}\big(\text{mode} = k\ \big|\ x_t = x\big)
\;=\; \frac{w_k\,p_t^{(k)}(x)}{p_t(x)},
\qquad
\mathbb{E}_{p_t}\big[\pi_k\big] = w_k\ \ \forall t .
```

`\pi_k` is `01`'s h-transform object for the internal condition
"which mode" — and the true output weight is
`w_k = E[\pi_k]` at every level, for EVERY exact `\lambda`-sampler
(the expectation only sees marginals). The finer structure is
`\lambda`-dependent:

**Lemma (the martingale, and its `\lambda`-defect).** Under the
exact `\lambda = 1` reverse process, `\pi_k(x_\tau, \tau)` is a
martingale: `(\partial_\tau + L_{P_1})\pi_k = 0`. *Probabilistic
proof:* the `\lambda=1` reverse process has the law of the true joint
process reversed, and posteriors along a Markov chain are martingales
by the tower property (`guidance_and_control/01`'s lemma, with
"mode" as the conditioned variable). Moreover, the general-`\lambda`
computation (write `\pi = w e^{\ell_k - \ell}`, substitute both
Fokker–Planck equations in log form, cancel):

```math
\big(\partial_\tau + L_{P_\lambda}\big)\pi_k
\;=\;
\big(\lambda^2 - 1\big)\,\frac{g^2}{2}\,\pi_k\Big[\Delta(\ell_k - \ell) + \|u\|^2 + u\cdot\nabla\ell\Big],
\qquad u := \nabla(\ell_k - \ell):
```

zero identically iff `\lambda = 1`. ∎ The SDE endpoint is not merely
robust (`samplers_and_convergence/05`, `guidance_and_control/06`): it
is the unique sampler whose paths CONSERVE mode posteriors — the
right frame for mode accounting, and the file works there (the
`\lambda \ne 1` residual is second-order in errors; statement at the
end).

## The Mode-Weight Bound, Proved

Sampler `Q`: the `\lambda = 1` reverse SDE with learned score
`\hat s`, initialized at the prior. Output weight
`\hat w_k := Q(x_{t_{\min}} \in \text{basin}_k)`.

**Theorem.**

```math
\big|\hat w_k - w_k\big|
\;\le\;
\int_{t_{\min}}^{T} g_t^2\ \mathbb{E}_{Q}\Big[\|\hat s - s\|\,\|\nabla\pi_k\|\Big]\mathrm{d}t
\;+\; \varepsilon_{\mathrm{prior}} \;+\; \varepsilon_{\mathrm{commit}},
```

and by Cauchy–Schwarz the integral is at most
`\big(\int g^2 E_Q\|\hat s - s\|^2\big)^{1/2}\big(\int g^2
E_Q\|\nabla\pi_k\|^2\big)^{1/2}` — score error times the
**mode-Fisher budget**.

*Proof.* Dynkin along `Q` (regularity: `\pi` smooth and bounded on
`[t_{\min}, T]`):
`\frac{d}{d\tau}E_Q[\pi_k] = E_Q[(\partial_\tau + L_Q)\pi_k]
= E_Q[(\partial_\tau + L_{P_1})\pi_k] + E_Q[\Delta b\cdot\nabla\pi_k]
= E_Q[g^2(\hat s - s)\cdot\nabla\pi_k]` by the Lemma
(`\Delta b = g^2(\hat s - s)` at `\lambda = 1`). Integrate from `T`
to `t_{\min}` and bound; the two endpoint terms are
`\varepsilon_{\mathrm{prior}} = |E_Q[\pi_k(x_T, T)] - w_k|` (the
prior carries almost no mode information: bounded by
`\sup_x\|\nabla\pi_k(\cdot, T)\|\cdot E_Q\|x\| = O(\alpha_T)` — the
posterior's slope dies with the signal) and
`\varepsilon_{\mathrm{commit}} = |E_Q[\pi_k(t_{\min})] -
\hat w_k|`, bounded by `E_Q[\min(\pi_k, 1-\pi_k)]` — exponentially
small at separation (next theorem). ∎

## The Budget Is One Bit, Spent At The Merge Scale

Two-point case (`\pm a`, equal weights, noise `\sigma`;
`score_foundations/06`'s tanh world, where
`\pi(x) = (1 + e^{-2ax/\sigma^2})^{-1}`):

**Theorem.** `\nabla\pi = \tfrac{2a}{\sigma^2}\,\pi(1-\pi)`, and:

```text
(separated, sigma <= a/2):  E_{p_t}||grad pi||^2
                            <= (4a^2/sigma^4) * 2 e^{-a^2/8 sigma^2}
                            (split at |x| = a/2: pi(1-pi) <=
                            e^{-2a|x|/sigma^2}, and each Gaussian
                            component puts mass <= e^{-a^2/8sigma^2}
                            inside the split) — exponentially dead;
(general):                  E||grad pi||^2 <= a^2/(4 sigma^4);
(the total budget):         in the VE clock (g^2 dt = d sigma^2),
                            int g^2 E||grad pi||^2 dt
                            <= int_{a^2/4}^infty a^2/(4s^2) ds + exp.
                            = 1 + (exponentially small) = O(1),
                            INDEPENDENT OF a.
```

*Proof.* Differentiate the sigmoid; the tail split as displayed; the
budget integral: substitute `s = \sigma^2`, integrate `a^2/4s^2` from
the window edge `s = a^2/4`, and the sub-window contribution is the
exponential bound integrated — finite and negligible. ∎

The reading is the file's headline: **the total sensitivity of a mode
weight to score error is a dimensionless `O(1)` constant — one bit of
mode identity, one unit-scale Fisher budget — and it is spent almost
entirely in the window `\sigma \approx a`, the scale at which the
modes merge.** Consequences, each now quantitative: score errors at
low noise (however large) CANNOT move mode weights (the budget there
is `e^{-a^2/8\sigma^2}`-dead — low-noise error costs fidelity within
modes, `04`'s regime, never diversity); diversity claims are claims
about score accuracy in one specific window, measurable by evaluating
the trained score against held-out data at `\sigma \approx`
inter-mode distances; and `samplers_and_convergence/06`'s "weights
decided early" mechanism is now two-sided — decided early, and
UNDECIDABLE late.

## Fences

```text
lambda != 1        the defect term makes the ODE's mode accounting
                   second-order-coupled: d/dtau E_Q[pi] gains
                   (lambda^2-1)(g^2/2)(E_Q - E_{p_t})[pi B] with B
                   the bracket — controlled by TV(q_t, p_t) times
                   ||pi B||_inf (Pinsker + A/03), second order in
                   errors; statement, honest;
E_Q vs E_p         the theorem's integrand lives under the sampler's
                   path (F/01's caveat); the same Pinsker conversion
                   trades it for data-path quantities at second-order
                   cost;
K modes            pairwise merges at separations a_{ij}: the budget
                   sums over the merge hierarchy (~ one bit per
                   binary split; the two-point theorem is the
                   transverse model) — statement, with the clean
                   general theorem open;
basin definition   hat-w reads basins at t_min; commit error is the
                   exponential above, provided t_min is below the
                   smallest separation's window.
```

## Position In The Coordinate System

Problem 3 of `05`, closed at the level posed: the estimand's error is
converted to mode-mass error through a proved identity (martingale +
Dynkin), with the conversion factor — the mode-Fisher budget —
computed and found to be `O(1)` and localized. The file also closes a
loop with `guidance_and_control/06`: conditioning-Fisher and
mode-Fisher are the same object (a posterior's gradient energy along
the path), once for external conditions, once for the data's own
latent structure.

## What Remains Open

The general-`K` merge-hierarchy theorem (the budget as a sum over the
mode dendrogram — well-posed, unproved); empirical execution: the
window-localized score-error measurement this file licenses, run
against a trained model's actual diversity failures; and the
`\lambda`-optimal mode accounting (the defect suggests slight
sub-unity `\lambda` could trade mode fidelity for within-mode
fidelity — a dial nobody has touched).
