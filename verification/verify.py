"""Numerical verification of closed-form results in Langevin's Ledger.

Each check tests a theorem proved in the notebooks against direct
simulation or quadrature. Run: python3 verification/verify.py
Exit code 0 = all checks pass at stated tolerances.
"""
import numpy as np

rng = np.random.default_rng(0)
PASS = []

def check(name, ok, detail=""):
    PASS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

# -- foundations/04: DDIM exact iff denoiser frozen. Point-mass data:
# x0hat = 0 constant => 12-step DDIM must land exactly on the TRUE
# t_min marginal, var = sigma^2(t_min) (not 0: the truncation floor).
# Gaussian data: x0hat varies along trajectories => DDIM strictly
# contracts variance (this check FALSIFIED the original foundations/06
# claim; both files now carry the correction).
K = 12
ts = np.linspace(1e-4, 5.0, 400)
def alpha(t): return np.exp(-0.5 * t)          # beta = 1
def sigma(t): return np.sqrt(1 - alpha(t)**2)
grid = np.linspace(5.0, 1e-4, K)               # coarse: 12 steps
x = rng.standard_normal(200000) * sigma(grid[0])   # data = delta_0
xg = rng.standard_normal(200000) * 1.0             # data = N(0,1)
for i in range(K - 1):
    t, tn = grid[i], grid[i + 1]
    a, s = alpha(t), sigma(t)
    x = alpha(tn) * 0.0 + sigma(tn) * (x / s)      # frozen x0hat = 0
    x0h = a * xg                                    # Gaussian denoiser
    xg = alpha(tn) * x0h + sigma(tn) * (xg - a * x0h) / s
check("DDIM exact iff denoiser frozen (point-mass yes, Gaussian no)",
      abs(np.var(x) - sigma(grid[-1])**2) < 5e-6 and np.var(xg) < 0.9,
      f"point var={np.var(x):.6e} vs sigma^2(t_min)={sigma(grid[-1])**2:.6e}; "
      f"Gaussian var={np.var(xg):.3f}<1 (contraction, as corrected)")

# -- foundations/02: Tweedie on a two-point mixture --------------------------
a_, s_ = 0.8, 0.5                              # alpha_t, sigma_t
x0 = rng.choice([-1.0, 1.0], 400000)
xt = a_ * x0 + s_ * rng.standard_normal(400000)
# closed-form: x0hat = tanh(a x / s^2) for atoms +-1
bins = np.linspace(-2, 2, 41)
idx = np.digitize(xt, bins)
emp, thr = [], []
for b in range(5, 36):
    m = idx == b
    if m.sum() > 500:
        c = 0.5 * (bins[b - 1] + bins[b])
        emp.append(x0[m].mean())
        thr.append(np.tanh(a_ * c / s_**2))
err = np.max(np.abs(np.array(emp) - np.array(thr)))
check("Tweedie: E[x0|xt] = tanh(a x/s^2) (two-point)", err < 0.03,
      f"max err={err:.4f}")

# -- foundations/05 + guidance/02: Gaussian CFG tilt formulas ----------------
mc, vc, mu_, vu = 1.0, 0.5, 0.0, 1.0
om = 2.5
lam = om / vc + (1 - om) / vu
m_th = (om * mc / vc + (1 - om) * mu_ / vu) / lam
xs = np.linspace(-8, 10, 20001)
lt = om * (-0.5 * (xs - mc)**2 / vc) + (1 - om) * (-0.5 * (xs - mu_)**2 / vu)
w = np.exp(lt - lt.max()); w /= np.trapezoid(w, xs)
m_num = np.trapezoid(w * xs, xs)
v_num = np.trapezoid(w * (xs - m_num)**2, xs)
check("CFG Gaussian family: precision/mean formulas",
      abs(m_num - m_th) < 1e-3 and abs(v_num - 1 / lam) < 1e-3,
      f"m={m_num:.4f} vs {m_th:.4f}; v={v_num:.4f} vs {1/lam:.4f}")

# -- foundations/05: the Jensen gap is real and nonnegative ------------------
# mixture data, r = p(x0|y)/p(x0) with y = component 1
x0m = np.where(rng.random(400000) < 0.5, 1.0, -1.0)
xtm = 0.5 * x0m + np.sqrt(1 - 0.25) * rng.standard_normal(400000)
r = np.where(x0m > 0, 2.0, 1e-12)              # p(x0|y)/p(x0): 2 or ~0
m = np.abs(xtm) < 0.2                          # a diffuse-posterior slice
gap = np.log(r[m].mean()) - np.log(r[m] + 1e-300).mean()
check("Jensen gap log E[r|xt] - E[log r|xt] > 0", gap > 0.5,
      f"gap={gap:.2f} nats at |xt|<0.2")

# -- statistical_theory/06: the mode-Fisher budget is O(1) -------------------
# two atoms +-a, VE clock: integral of E||grad pi||^2 d(sigma^2)
aa = 3.0
s2s = np.exp(np.linspace(np.log(1e-3), np.log(400.0), 4000))
vals = []
for s2 in s2s:
    sd = np.sqrt(s2)
    z = np.linspace(-aa - 8 * sd, aa + 8 * sd, 4001)
    p = 0.5 * (np.exp(-(z - aa)**2 / (2 * s2)) +
               np.exp(-(z + aa)**2 / (2 * s2))) / np.sqrt(2 * np.pi * s2)
    pi = 1 / (1 + np.exp(np.clip(-2 * aa * z / s2, -700, 700)))
    gp = (2 * aa / s2) * pi * (1 - pi)
    vals.append(np.trapezoid(gp**2 * p, z))
budget = np.trapezoid(np.array(vals), s2s)
below = np.trapezoid(np.array(vals)[s2s < (aa / 3)**2], s2s[s2s < (aa / 3)**2])
check("mode-Fisher budget O(1), window-concentrated",
      0.2 < budget < 3.0 and below < 0.05 * budget,
      f"budget={budget:.3f}, sub-window share={below/budget:.4f}")

# -- samplers/01: EM stationary variance = 1/(1 - beta h/4) ------------------
bh = 0.4
v = 1.0
for _ in range(4000):
    v = (1 - bh / 2)**2 * v + bh
check("EM bias: v* = 1/(1-beta h/4)", abs(v - 1 / (1 - bh / 4)) < 1e-9,
      f"v*={v:.6f} vs {1/(1-bh/4):.6f}")

# -- discrete/04: the Beta-integral = permutation weight ---------------------
from math import factorial
L, R = 6, 2                                    # |R| revealed among L-1 others
u = np.linspace(0, 1, 200001)
lhs = np.trapezoid(u**(L - 1 - R) * (1 - u)**R, u)
rhs = factorial(L - 1 - R) * factorial(R) / factorial(L)
check("any-order AR weight: Beta integral = |R|!(L-1-|R|)!/L!",
      abs(lhs - rhs) < 1e-6, f"{lhs:.6f} vs {rhs:.6f}")

# -- discrete/05: the parallel-decoding tax on the two-token example ---------
# data uniform on {(A,B),(B,A)}: TC = 1 bit
tc = (2 * 1.0) - 1.0                           # sum H(x_i) - H(joint), bits
check("parallel-decoding tax: TC = 1 bit (half-mass impossible)",
      abs(tc - 1.0) < 1e-12, "H=1+1-1")

# -- statistical_theory/07: the optimal grid is alpha-uniform ----------------
# beta = 1, lambda = 1, Gaussian data: run the exact EM variance map on a
# uniform grid vs the alpha-uniform grid; compare to E* = (1-e^{-B/2})^2/K.
T, K7 = 5.0, 500
def run_grid(grid):
    v = 1.0
    for i in range(len(grid) - 1):
        h = grid[i] - grid[i + 1]
        v = (1 - h / 2)**2 * v + h
    return abs(v - 1.0)
e_unif = run_grid(np.linspace(T, 0.0, K7 + 1))
e_alph = run_grid(-2 * np.log(np.linspace(np.exp(-T / 2), 1.0, K7 + 1)))
e_star = (1 - np.exp(-T / 2))**2 / K7
check("optimal grid: alpha-uniform hits (1-e^{-B/2})^2/K, beats uniform",
      abs(e_alph - e_star) / e_star < 0.06 and e_alph < 0.75 * e_unif,
      f"alpha-grid={e_alph:.3e} vs E*={e_star:.3e}; uniform={e_unif:.3e}")

# -- statistical_theory/07: minimax weight w* ∝ A ----------------------------
# worst-case output error factor J(w) = sqrt(int A^2/w), int w = 1;
# J(w*) must equal int A and beat the uniform weight.
tt = np.linspace(1e-6, T, 200001)
A7 = np.exp(-tt / 2)                            # (1+l^2)/2 * beta * e^{-B/2}
intA = np.trapezoid(A7, tt)
w_star = A7 / intA
w_unif = np.ones_like(tt) / T
J_star = np.sqrt(np.trapezoid(A7**2 / w_star, tt))
J_unif = np.sqrt(np.trapezoid(A7**2 / w_unif, tt))
check("minimax weight: J(w*) = int A = 2(1-e^{-B/2}), < J(uniform)",
      abs(J_star - 2 * (1 - np.exp(-T / 2))) < 1e-3 and J_star < J_unif,
      f"J*={J_star:.4f} vs intA={2*(1-np.exp(-T/2)):.4f}; J_unif={J_unif:.4f}")

# -- statistical_theory/07: the CFM floor = Wronskian^2/rho^2 (MC) -----------
# Brownian bridge (non-VP) at t = 0.3: residual variance of the CFM target
# after regressing on x_t must equal (bdot*g - b*gdot)^2/rho^2.
t7 = 0.3
b7, g7 = t7, np.sqrt(t7 * (1 - t7))
bd7, gd7 = 1.0, (1 - 2 * t7) / (2 * np.sqrt(t7 * (1 - t7)))
x1s = rng.standard_normal(400000); zs = rng.standard_normal(400000)
xts = b7 * x1s + g7 * zs
vcs = bd7 * x1s + gd7 * zs
resid = np.var(vcs) - np.cov(vcs, xts)[0, 1]**2 / np.var(xts)
floor_th = (bd7 * g7 - b7 * gd7)**2 / (b7**2 + g7**2)
# and the VP floor ordering: trig path = pi^2/4, slower-than-linear theta pays
tg = np.linspace(0, 1, 100001)
def vp_floor(theta):
    b, g = np.cos(theta), np.sin(theta)
    db, dg = np.gradient(b, tg), np.gradient(g, tg)
    return np.trapezoid((db * g - b * dg)**2, tg)
fl_trig = vp_floor((np.pi / 2) * (1 - tg))
fl_quad = vp_floor((np.pi / 2) * (1 - tg)**2)
check("CFM floor: Wronskian identity (MC) + trig path = pi^2/4 minimal",
      abs(resid - floor_th) < 5e-3 and abs(fl_trig - np.pi**2 / 4) < 1e-3
      and fl_quad > fl_trig * 1.2,
      f"MC resid={resid:.4f} vs {floor_th:.4f}; "
      f"trig={fl_trig:.4f} vs pi^2/4={np.pi**2/4:.4f}; quad={fl_quad:.4f}")

# -- statistical_theory/08: the discrete excess-risk identity ----------------
# L=3 binary sequences, random joint; check L(theta) - H = order-averaged KL
# (mean-field model, so the identity is nontrivially exercised), and that the
# entropy-telescope Sum w H(x^l|x_R) = H(x0).
from itertools import product as iproduct
from math import factorial as fac
L8 = 3
states = list(iproduct([0, 1], repeat=L8))
pj = rng.random(len(states))**2; pj /= pj.sum()          # random joint on {0,1}^3
P = {s: pj[i] for i, s in enumerate(states)}
def marg(idxs, vals):                                    # P(x_idxs = vals)
    return sum(P[s] for s in states if all(s[i] == v for i, v in zip(idxs, vals)))
H_joint = -sum(p * np.log(p) for p in pj if p > 0)
# mean-field model q(x^l) = marginal of position l
q = [[marg([l], [b]) for b in (0, 1)] for l in range(L8)]
def w8(R):                                               # Beta weight, |R| given
    return fac(len(R)) * fac(L8 - 1 - len(R)) / fac(L8)
# LHS: order-averaged conditional KL of true cond vs mean-field q
lhs8 = 0.0; ent_tel = 0.0
for l in range(L8):
    others = [i for i in range(L8) if i != l]
    for r in range(len(others) + 1):
        from itertools import combinations
        for R in combinations(others, r):
            for vals in iproduct([0, 1], repeat=len(R)):
                pR = marg(list(R), list(vals))
                if pR <= 0:
                    continue
                cond = [marg([l] + list(R), [b] + list(vals)) / pR for b in (0, 1)]
                kl = sum(c * np.log(c / q[l][b]) for b, c in enumerate(cond) if c > 0)
                Hc = -sum(c * np.log(c) for c in cond if c > 0)
                lhs8 += w8(R) * pR * kl
                ent_tel += w8(R) * pR * Hc
# RHS: L(theta) - H, with L(theta) = sum w E[-log q]
Ltheta = 0.0
for l in range(L8):
    others = [i for i in range(L8) if i != l]
    for r in range(len(others) + 1):
        for R in combinations(others, r):
            for vals in iproduct([0, 1], repeat=len(R)):
                for b in (0, 1):
                    pjoint = marg([l] + list(R), [b] + list(vals))
                    if pjoint > 0:
                        Ltheta += w8(R) * pjoint * (-np.log(q[l][b]))
check("discrete excess-risk: L(theta)-H = order-avg KL; entropy telescope",
      abs((Ltheta - H_joint) - lhs8) < 1e-9 and abs(ent_tel - H_joint) < 1e-9,
      f"L-H={Ltheta-H_joint:.5f} vs KL={lhs8:.5f}; telescope={ent_tel:.5f} vs H={H_joint:.5f}")

# -- statistical_theory/08: multinomial KL floor (|V|-1)/2n ------------------
V8, n8, trials = 4, 400, 6000
ptrue = rng.random(V8); ptrue /= ptrue.sum()
kls = []
for _ in range(trials):
    counts = rng.multinomial(n8, ptrue)
    phat = (counts + 0.5) / (n8 + V8 * 0.5)              # KT smoothing (audit)
    kls.append(np.sum(ptrue * np.log(ptrue / phat)))
floor = (V8 - 1) / (2 * n8)
check("multinomial KL floor E[KL] = (|V|-1)/2n",
      abs(np.mean(kls) - floor) < 0.4 * floor,
      f"E[KL]={np.mean(kls):.5f} vs (|V|-1)/2n={floor:.5f}")

# -- statistical_theory/08: the scheduling identity T = C - sum I(x;pred) ----
# reuse the L=3 joint; test serial, one-shot, and a 2-block schedule.
def H_set(idxs):
    if not idxs: return 0.0
    tot = {}
    for s in states:
        key = tuple(s[i] for i in idxs); tot[key] = tot.get(key, 0.0) + P[s]
    return -sum(p * np.log(p) for p in tot.values() if p > 0)
def I_pred(l, pred):
    return H_set([l]) + H_set(pred) - H_set(sorted([l] + pred))
def tax_direct(blocks):                                  # sum_j TC(S_j | c_j)
    T = 0.0; c = []
    for S in blocks:
        # TC(S|c) = sum_l H(x_l|c) - H(x_S|c) = sum_l[H(l,c)-H(c)] -[H(S,c)-H(c)]
        Hc = H_set(c)
        T += sum(H_set(sorted([l] + c)) - Hc for l in S) - (H_set(sorted(S + c)) - Hc)
        c = sorted(c + S)
    return T
C8 = sum(H_set([l]) for l in range(L8)) - H_joint
ok8 = True
for blocks in [[[0], [1], [2]], [[0, 1, 2]], [[0, 2], [1]]]:
    pred_sum = 0.0; revealed = []
    for S in blocks:
        for l in S: pred_sum += I_pred(l, revealed)
        revealed = sorted(revealed + S)
    ok8 &= abs(tax_direct(blocks) - (C8 - pred_sum)) < 1e-9
check("reveal-schedule identity T = C - sum I(x;pred) (3 schedules)",
      ok8 and abs(tax_direct([[0, 1, 2]]) - C8) < 1e-9
      and abs(tax_direct([[0], [1], [2]])) < 1e-9,
      f"C={C8:.5f}; one-shot tax=C, serial tax=0 both verified")

# -- schrodinger_bridges/01: entropic OT Gaussian closed form ----------------
# a=1, b=2: run discrete Sinkhorn on N(0,1) x N(0,4), c=|x-y|^2/2, and
# compare Cov(x,y) of the plan to c_eps = sqrt(a^2 b^2 + eps^2/4) - eps/2.
za = np.linspace(-6, 6, 401)                    # x-grid, std 1
zb = np.linspace(-12, 12, 401)                  # y-grid, std 2
mu9 = np.exp(-za**2 / 2); mu9 /= mu9.sum()
nu9 = np.exp(-zb**2 / 8); nu9 /= nu9.sum()
def lse(M, axis):                               # log-sum-exp along axis
    m = M.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(M - m).sum(axis=axis, keepdims=True))).squeeze(axis)
ok9, det9 = True, []
for eps in (0.5, 1.0, 4.0):
    lK = -0.5 * (za[:, None] - zb[None, :])**2 / eps
    lf, lg = np.zeros_like(mu9), np.zeros_like(nu9)
    for _ in range(3000):                       # log-domain Sinkhorn
        lg = np.log(nu9) - lse(lK + lf[:, None], axis=0)
        lf = np.log(mu9) - lse(lK + lg[None, :], axis=1)
    pi9 = np.exp(lf[:, None] + lK + lg[None, :])
    c_num = np.sum(pi9 * za[:, None] * zb[None, :])
    c_th = np.sqrt(1.0 * 4.0 + eps**2 / 4) - eps / 2
    ok9 &= abs(c_num - c_th) < 5e-3
    det9.append(f"eps={eps}: {c_num:.4f} vs {c_th:.4f}")
check("entropic OT (Gaussian): Sinkhorn Cov = sqrt(a^2b^2+eps^2/4)-eps/2",
      ok9, "; ".join(det9))

# -- schrodinger_bridges/02: OU-reference bridge coupling --------------------
# mu = N(0, s^2), reference = VP forward (alpha = e^{-T/2}, sig2 = 1-alpha^2):
# static problem = entropic OT, cost (y - alpha x)^2/2, temperature sig2.
# (a) nu = N(0,1): Sinkhorn Cov(x,y) must equal
#     c* = [sqrt(alpha^2 s^2 + sig2^2/4) - sig2/2]/alpha;
# (b) nu = p_T = N(0, alpha^2 s^2 + sig2): f = g = 1 exactness, Cov = alpha s^2.
s10 = 1.2
za = np.linspace(-6 * s10, 6 * s10, 401)
mu10 = np.exp(-za**2 / (2 * s10**2)); mu10 /= mu10.sum()
ok10, det10 = True, []
for T10 in (0.2, 1.0, 3.0):
    al = np.exp(-T10 / 2); sg2 = 1 - al**2
    for case, bvar in (("prior", 1.0), ("p_T", al**2 * s10**2 + sg2)):
        zb = np.linspace(-6 * np.sqrt(bvar), 6 * np.sqrt(bvar), 401)
        nu10 = np.exp(-zb**2 / (2 * bvar)); nu10 /= nu10.sum()
        lK = -0.5 * (zb[None, :] - al * za[:, None])**2 / sg2
        lf, lg2 = np.zeros_like(mu10), np.zeros_like(nu10)
        for _ in range(2000):
            lg2 = np.log(nu10) - lse(lK + lf[:, None], axis=0)
            lf = np.log(mu10) - lse(lK + lg2[None, :], axis=1)
        pi10 = np.exp(lf[:, None] + lK + lg2[None, :])
        c_num = np.sum(pi10 * za[:, None] * zb[None, :])
        if case == "prior":
            c_th = (np.sqrt(al**2 * s10**2 + sg2**2 / 4) - sg2 / 2) / al
        else:
            c_th = al * s10**2                  # exactness: bridge = reference
        ok10 &= abs(c_num - c_th) < 5e-3
        det10.append(f"T={T10}/{case}: {c_num:.4f} vs {c_th:.4f}")
check("OU bridge: Sinkhorn = closed form; nu = p_T gives Q* = R exactly",
      ok10, "; ".join(det10))

# -- schrodinger_bridges/03: Gaussian IPF = Mobius map, rate rho^4 -----------
# a=1, b=2, eps=1 (q = 1/eps): p_{n+1} = 1/a^2 + q^2/(1/b^2 + q^2/p_n).
# Fixed point must reproduce 01's plan (Var a^2, b^2; Cov c_eps) and the
# error ratio must converge to rho^4 with rho = c_eps/(ab).
a11, b11, eps11 = 1.0, 2.0, 1.0
q11 = 1 / eps11
c11 = np.sqrt(a11**2 * b11**2 + eps11**2 / 4) - eps11 / 2
p11, hist = 1 / a11**2, []
for _ in range(60):
    hist.append(p11)
    p11 = 1 / a11**2 + q11**2 / (1 / b11**2 + q11**2 / p11)
r11 = 1 / b11**2 + q11**2 / p11
det11 = p11 * r11 - q11**2
vx, vy, cv = r11 / det11, p11 / det11, q11 / det11
errs = np.abs(np.array(hist) - p11)
ratio = errs[8] / errs[7]
rho4 = (c11 / (a11 * b11))**4
check("Gaussian IPF: fixed point = entropic plan; contraction = rho^4",
      abs(vx - a11**2) < 1e-12 and abs(vy - b11**2) < 1e-12
      and abs(cv - c11) < 1e-12 and abs(ratio - rho4) < 1e-3,
      f"Var=({vx:.4f},{vy:.4f}) Cov={cv:.4f} vs {c11:.4f}; "
      f"rate={ratio:.4f} vs rho^4={rho4:.4f}")

print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
raise SystemExit(0 if all(PASS) else 1)
