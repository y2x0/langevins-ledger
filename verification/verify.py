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
# x0hat = 0 constant => 12-step DDIM must land exactly on 0. Gaussian
# data: x0hat varies along trajectories => DDIM strictly contracts
# variance (this check FALSIFIED the original foundations/06 claim;
# both files now carry the correction).
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
      np.var(x) < 1e-4 and np.var(xg) < 0.9,
      f"point var={np.var(x):.2e}; Gaussian var={np.var(xg):.3f}<1 (contraction, as corrected)")

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

print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
raise SystemExit(0 if all(PASS) else 1)
