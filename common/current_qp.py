"""The current augmented QP without actuation bounds, shared by Applications 1-a and 1-b.

    min  1/2 |u - u_ref|_R^2 + 1/(2 rho) |nu - nu_nom|_S^2 + 1/(2 rho^2) |eta - eta_lo|_T^2
    s.t. A_u u + A_nu nu + D_psi eta + b >= eps 1_p,
         u in R^m,   nu_lo(theta) <= nu <= nu_bar,   eta >= eta_lo 1_p,

with diagonal R, S, T.  Every barrier has relative degree r_i = 3, so it carries two lower-stage
gains: kappa_{i,1} (chain order d = 2, nu_{i,1} = kappa_{i,1}'') and kappa_{i,2} (d = 1,
nu_{i,2} = kappa_{i,2}'), and a terminal gain eta_i.

Lower rate bound = an HOCBF keeping every lower-stage gain above KMIN, Hurwitz polynomial pi_d(s) = prod_j (s + c_j):
nu >= nu_lo  <=>  pi_d(D)(kappa - KMIN) >= 0,
    d = 2:  nu = kappa'' >= -(C1+C2) kappa' + C1 C2 (KMIN - kappa),        d = 1:  nu = kappa' >= C1 (KMIN - kappa).
The upper bound is the constant cap nu_bar = NU_BAR = 10 of Table I; it is never active in the reported tests.
Nominal rate = the input that makes (D + LAMBDA)^d (kappa - kappa_nom) = 0, clipped to the rate box,
    d = 2:  LAMBDA^2 (kappa_nom - kappa) - 2 LAMBDA kappa',        d = 1:  LAMBDA (kappa_nom - kappa).
The terminal gains have no upper bound and are not held at eta_lo at the first sample.
"""
from __future__ import annotations

import numpy as np
import quadprog

EPS, ETA_MIN = 0.02, 0.35                   # terminal reserve, lower bound (and cost centre) of eta
KMIN = 0.30                                 # lower bound of every lower-stage gain
NU_BAR = 10.0                               # constant cap on every differential input (Table I)
C1, C2 = 4.0, 5.0                           # roots of the rate-bound polynomials (d = 2: C1, C2; d = 1: C1)
LAMBDA = 1.0                                # nominal regulation rate of the gains
K_NOM = np.array([2.3, 2.0])                # kappa_nom = the fixed-gain values (kappa_1, kappa_2)
RHO = 1e-2
R_DIAG, S_DIAG, T_DIAG = 1.0, 12.0, 8.0     # R = I, S = 12 I, T = 8 I -> weights 1, 1.2e3, 8e4

PARAMETERS = dict(eps=EPS, eta_lo=ETA_MIN, kappa_min=KMIN, nu_bar=NU_BAR, c=(C1, C2), lambda_kappa=LAMBDA,
                  kappa_nom=K_NOM.tolist(), rho=RHO, R=R_DIAG, S=S_DIAG, T=T_DIAG,
                  weights=(R_DIAG, S_DIAG / RHO, T_DIAG / RHO ** 2), input_bounds="none (u in R^m)")


def rate_box(gains):
    """Rate bounds and the clipped nominal rate for gains (p, 3) = (kappa_1, kappa_1', kappa_2)."""
    k1, k1d, k2 = gains[:, 0], gains[:, 1], gains[:, 2]
    lo = np.c_[-(C1 + C2) * k1d + C1 * C2 * (KMIN - k1), C1 * (KMIN - k2)].ravel()
    hi = np.full_like(lo, NU_BAR)
    raw = np.c_[LAMBDA ** 2 * (K_NOM[0] - k1) - 2 * LAMBDA * k1d, LAMBDA * (K_NOM[1] - k2)].ravel()
    return lo, hi, np.clip(raw, lo, hi)


MODES = ("both", "eta_only", "nu_only")   # ablation: which gain channels are decision variables
ETA_CAP = np.inf                           # optional upper bound on eta (only for the gain-limit study; none in the QP)


def solve(au, anu, psi_term, drift, gains, u_ref, mode="both"):
    """Solve the QP.  au (p, m) physical rows, anu (p, 2) = (h_i, psi_i1), psi_term (p,) = psi_i2,
    drift (p,) = b_i, gains (p, 3), u_ref (m,).  Returns (z, ok, A, info) with z = (u, nu (2p), eta (p)).
    mode = "both" (the proposed QP), "eta_only" (nu fixed at nu_nom: the lower-stage gains are not adapted)
    or "nu_only" (eta fixed at eta_lo: no private terminal direction).  If the QP is infeasible, z is the
    fallback (u_ref, nu_nom, eta_lo), the same fallback as the fixed-gain baseline."""
    au = np.atleast_2d(au); p, m = au.shape
    A = np.zeros((p, m + 3 * p))
    A[:, :m] = au
    for i in range(p):
        A[i, m + 2 * i:m + 2 * i + 2] = anu[i]
    A[:, m + 2 * p:] = np.diag(psi_term)
    nu_lo, nu_hi, nu_nom = rate_box(np.asarray(gains, float))
    ref = np.r_[u_ref, nu_nom, ETA_MIN * np.ones(p)]
    wts = np.r_[R_DIAG * np.ones(m), S_DIAG / RHO * np.ones(2 * p), T_DIAG / RHO ** 2 * np.ones(p)]
    lo = np.r_[-np.inf * np.ones(m), nu_lo, ETA_MIN * np.ones(p)]
    hi = np.r_[np.inf * np.ones(m), nu_hi, ETA_CAP * np.ones(p)]
    free = np.ones(m + 3 * p, bool)
    if mode == "eta_only":
        free[m:m + 2 * p] = False
    elif mode == "nu_only":
        free[m + 2 * p:] = False
    elif mode != "both":
        raise ValueError(mode)
    z = ref.copy()                                                  # fixed variables sit at their nominal values
    Af, rhs_f = A[:, free], EPS - np.asarray(drift, float) - A[:, ~free] @ ref[~free]
    wf, rf, lf, hf = wts[free], ref[free], lo[free], hi[free]
    G = np.diag(wf) + 1e-10 * np.eye(len(wf))
    blocks, rhs = [Af], [rhs_f]
    for k, v in enumerate(lf):
        if np.isfinite(v):
            e = np.zeros(len(wf)); e[k] = 1.0; blocks.append(e[None]); rhs.append(np.array([v]))
    for k, v in enumerate(hf):
        if np.isfinite(v):
            e = np.zeros(len(wf)); e[k] = -1.0; blocks.append(e[None]); rhs.append(np.array([-v]))
    try:
        z[free], ok = quadprog.solve_qp(G, np.diag(wf) @ rf, np.vstack(blocks).T, np.concatenate(rhs))[0], True
    except ValueError:
        ok = False
    return z, ok, A, dict(nu_lo=nu_lo, nu_hi=nu_hi, nu_nom=nu_nom)


def gain_step(gains, nu, dt):
    """Exact zero-order-hold update of the gain chains (kappa_1, kappa_1') <- nu_1 and kappa_2 <- nu_2."""
    g = np.array(gains, float)
    n1, n2 = nu[0::2], nu[1::2]
    g[:, 0] += dt * gains[:, 1] + 0.5 * dt ** 2 * n1
    g[:, 1] += dt * n1
    g[:, 2] += dt * n2
    return g
