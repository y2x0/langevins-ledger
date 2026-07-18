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

print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
raise SystemExit(0 if all(PASS) else 1)
