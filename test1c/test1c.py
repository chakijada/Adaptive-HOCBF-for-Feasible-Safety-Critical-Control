#!/usr/bin/env python3
"""Test 1-c: harder variants of Test 1-b (Section IV-C of the paper).

Every variant keeps the Test 1-b plant, initial state, sampling (1020 samples of 0.01 s, zero-order hold),
weights and controllers, and changes one or more of: the reference speed, the poles of the nominal tracking
law, the obstacle geometry and the number of obstacles.  The three controllers are run on each variant:
the fixed-gain HOCBF-QP, LCS, and the proposed adaptive HOCBF-QP.

    python test1c/test1c.py              six variants of Fig. 3 of the paper
    python test1c/test1c.py --all        all sixteen variants (incl. those of the response to the reviewers)
    python test1c/test1c.py --ablation   also the eta-only and nu-only versions of the proposed QP
    python test1c/test1c.py --sweep      reference-speed sweep 1.0 ... 2.5 (proposed, eta-only, nu-only)

Results: results/test1c/<variant>_<controller>.npz and results/test1c/summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "test1b"))
import current_qp as cq                       # noqa: E402  the proposed QP
import test1b as B                            # noqa: E402  Test 1-b plant, reference and baseline controllers

RES = ROOT / "results" / "test1c"
RES.mkdir(parents=True, exist_ok=True)

CENTERS0 = B.CENTERS.copy()
R0 = float(B.R)
REFERENCE0 = B.REFERENCE.copy()
_NOMINAL0 = B.nominal_jerk


def restore():
    """Return test1b.py to the Test 1-b configuration."""
    B.CENTERS = CENTERS0.copy(); B.NOBS = len(CENTERS0); B.R = R0; B.REFERENCE = REFERENCE0.copy()
    B.nominal_jerk = _NOMINAL0
K_TRACK0 = np.array([12.0, 18.0, 8.0])                  # j_ref = j_r + 12 e_p + 18 e_v + 8 e_a  (poles 2, 3 -/+ sqrt 3)
G_NOM = np.array([cq.K_NOM[0], 0.0, cq.K_NOM[1]])       # (kappa_1, kappa_1', kappa_2) = (2.3, 0, 2.0)


@dataclass
class Scenario:
    key: str
    label: str
    centers: np.ndarray = field(default_factory=lambda: CENTERS0.copy())
    radius: float = R0
    speed: float = 1.0          # time compression of the reference
    track: float = 1.0          # pole scaling of the nominal tracking law


def shift(i, dxy):
    c = CENTERS0.copy(); c[i] += np.asarray(dxy, float); return c


def lower_passage(dist):
    """Move c_1 by `dist` metres toward c_4 (narrows the passage the proposed controller uses)."""
    u = CENTERS0[3] - CENTERS0[0]; u = u / np.linalg.norm(u)
    return shift(0, dist * u)


C5_EXIT = np.array([1.0, -1.45])                         # fifth obstacle at the exit of the lower passage
C5_LEFT = np.array([-1.0, -0.5])                         # fifth obstacle left of the start

SCENARIOS = {sc.key: sc for sc in [
    Scenario("S1", "Test 1-b"),
    Scenario("S2", "reference 1.25x faster", speed=1.25),
    Scenario("S3", "reference 1.5x faster", speed=1.5),
    Scenario("S4", "reference 1.75x faster", speed=1.75),
    Scenario("S5", "reference 2x faster", speed=2.0),
    Scenario("S6", "tracking poles 1.5x faster", track=1.5),
    Scenario("S7", "c4 moved 0.15 m onto the reference", centers=shift(3, (0.0, 0.15))),
    Scenario("S8", "passage between c1 and c4 narrowed by 0.25 m", centers=lower_passage(0.25)),
    Scenario("S9", "passage between c1 and c4 narrowed by 0.45 m", centers=lower_passage(0.45)),
    Scenario("S10", "all radii 0.68 m", radius=0.68),
    Scenario("S11", "fifth obstacle c5 at (1, -1.45)", centers=np.vstack([CENTERS0, C5_EXIT])),
    Scenario("S12", "S5 + S9 + S11 combined", centers=np.vstack([lower_passage(0.45), C5_EXIT]), speed=2.0),
    Scenario("S13", "poles 1.5x faster, fifth obstacle c5 at (-1, -0.5)", centers=np.vstack([CENTERS0, C5_LEFT]),
             track=1.5),
    Scenario("S14", "fifth obstacle c5 at (-1, -0.25)", centers=np.vstack([CENTERS0, [-1.0, -0.25]])),
    Scenario("S15", "reference and poles 1.5x faster, c5 at (-1, -0.5)", centers=np.vstack([CENTERS0, C5_LEFT]),
             speed=1.5, track=1.5),
    Scenario("S16", "passage narrowed by 0.1 m, radii 0.63 m", centers=lower_passage(0.1), radius=0.63),
]}
PAPER = ("S5", "S6", "S13", "S9", "S10", "S11")          # Fig. 3 (a)-(f) of the paper


# ------------------------------------------------------------------------------------------- scenario set-up
def time_scaled_reference(speed):
    """r_s(t) = r(T0 + speed (t - T0)), derivatives scaled by speed^k; the goal is unchanged."""
    out = []
    for t in B.SAMPLE_TIMES:
        r = B._reference_profile(float(B.T0 + speed * (t - B.T0)))
        out.append(np.r_[r[:2], speed * r[2:4], speed ** 2 * r[4:6], speed ** 3 * r[6:8]])
    return np.array(out)


def nominal_jerk(x, command, track):
    kp, kv, ka = K_TRACK0 * np.array([track ** 3, track ** 2, track])
    p, v, a = x[:2], x[2:4], x[4:6]
    return command[6:8] + kp * (command[:2] - p) + kv * (command[2:4] - v) + ka * (command[4:6] - a)


def configure(sc: Scenario):
    B.CENTERS = np.asarray(sc.centers, float); B.NOBS = len(B.CENTERS); B.R = float(sc.radius)
    B.REFERENCE = REFERENCE0.copy() if sc.speed == 1.0 else time_scaled_reference(sc.speed)
    # the baselines of test1b.py call B.nominal_jerk(x, t, waypoint)
    B.nominal_jerk = lambda x, t, waypoint=None: nominal_jerk(x, B.reference(t) if waypoint is None else waypoint,
                                                              sc.track)


def initial_gains(x, centers):
    """kappa(0) = nominal gains.  Where the nominal chain is not admissible at x(0) (psi_1 < 0 or psi_2 <= 0),
    kappa_1(0) and kappa_1'(0) are raised so that psi_1(0) = 0.01 and psi_2(0) = 1 (theta(0) in Theta, as allowed by
    Proposition 1).  The fixed-gain and LCS controllers keep the nominal gains."""
    G = np.tile(G_NOM, (len(centers), 1)); adjusted = []
    for i, c in enumerate(centers):
        r = B.obstacle_row(x, c, G[i], True)
        if r.h <= 0:
            raise ValueError(f"x(0) is inside obstacle {i + 1}")
        changed = False
        if r.psi1 < 0.0:
            G[i, 0] = (0.01 - r.hd) / r.h; changed = True
        r = B.obstacle_row(x, c, G[i], True)
        if r.psi2 <= 0.0:
            G[i, 1] = (1.0 - r.hdd - G[i, 0] * r.hd - G[i, 2] * r.psi1) / r.h; changed = True
        if changed:
            adjusted.append(i + 1)
    return G, adjusted


# ------------------------------------------------------------------------------------------- closed loops
def simulate_proposed(sc: Scenario, mode="both"):
    """Proposed QP (mode "both"), or its ablations "eta_only" (nu = nu_nom) and "nu_only" (eta = eta_lo)."""
    configure(sc)
    centers = B.CENTERS; p = len(centers); x = B.X0.copy()
    G, adjusted = initial_gains(x, centers)
    keys = ("t", "x", "ref", "u", "u_nom", "ok", "h", "psi1", "psi2", "psi3", "nu", "eta", "kappa", "kappa_d",
            "nu_lo", "nu_nom", "delta", "ms")
    log = {k: [] for k in keys}
    for t in B.SAMPLE_TIMES:
        command = B.reference(t)
        u_nom = nominal_jerk(x, command, sc.track)
        rows = [B.obstacle_row(x, c, G[i], True) for i, c in enumerate(centers)]
        au = np.array([r.au for r in rows]); anu = np.array([r.anu for r in rows])
        psi2 = np.array([r.psi2 for r in rows]); drift = np.array([r.drift for r in rows])
        t0 = time.perf_counter()
        z, ok, A, info = cq.solve(au, anu, psi2, drift, G, u_nom, mode)
        ms = 1e3 * (time.perf_counter() - t0)
        u, nu, eta = z[:2], z[2:2 + 2 * p], z[2 + 2 * p:]
        h = np.array([r.h for r in rows])
        # nominal reserve deficit d_i = eps - psi_dot_{i,2}(u_ref, nu_nom); delta_i = [d_i]_+ / h_i (condition (a))
        psidot_nom = au @ u_nom + np.array([anu[i] @ info["nu_nom"][2 * i:2 * i + 2] for i in range(p)]) + drift
        delta = np.maximum(cq.EPS - psidot_nom, 0.0) / h
        for k, v in (("t", float(t - B.TSH)), ("x", x.copy()), ("ref", command.copy()), ("u", u.copy()),
                     ("u_nom", u_nom.copy()), ("ok", bool(ok)), ("h", h), ("psi1", np.array([r.psi1 for r in rows])),
                     ("psi2", psi2), ("psi3", A @ z + drift), ("nu", nu.copy()), ("eta", eta.copy()),
                     ("kappa", G[:, [0, 2]].copy()), ("kappa_d", G[:, 1].copy()), ("nu_lo", info["nu_lo"].copy()),
                     ("nu_nom", info["nu_nom"].copy()), ("delta", delta), ("ms", ms)):
            log[k].append(v)
        if t < B.TF:
            x = B.plant_step(x, u); G = cq.gain_step(G, nu, B.DT)
    out = {k: np.array(v) for k, v in log.items()}
    out["adjusted"] = np.array(adjusted, int)
    return out


def simulate_baseline(sc: Scenario, kind):
    """Fixed-gain HOCBF-QP ("fixed") or LCS ("lcs"), the controllers of test1b.py, with the gains of Table I."""
    configure(sc)
    centers = B.CENTERS; p = len(centers); x = B.X0.copy()
    mask = np.ones(p, bool); lnu = np.ones(p)
    log = {k: [] for k in ("t", "x", "ref", "u", "u_nom", "ok", "h", "psi1", "psi2", "psi3", "mask", "ms")}
    for t in B.SAMPLE_TIMES:
        command = B.reference(t)
        t0 = time.perf_counter()
        if kind == "fixed":
            u, ok, unom, rows, A, beta, res = B.standard_control(x, t, centers, command); m = np.ones(p, bool)
        else:
            u, ok, unom, rows, A, beta, res, mask, lnu, _, _ = B.lcs_control(x, t, mask, lnu, centers, command)
            m = mask.copy()
        ms = 1e3 * (time.perf_counter() - t0)
        for k, v in (("t", float(t - B.TSH)), ("x", x.copy()), ("ref", command.copy()), ("u", u.copy()),
                     ("u_nom", unom.copy()), ("ok", bool(ok)), ("h", np.array([r.h for r in rows])),
                     ("psi1", np.array([r.psi1 for r in rows])), ("psi2", np.array([r.psi2 for r in rows])),
                     ("psi3", res + cq.EPS), ("mask", m), ("ms", ms)):
            log[k].append(v)
        if t < B.TF:
            x = B.plant_step(x, u)
    return {k: np.array(v) for k, v in log.items()}


# ------------------------------------------------------------------------------------------- metrics
def metrics(d):
    ok = d["ok"]; hmin = d["h"].min(1); unsafe = hmin < 0
    x, ref, u, un = d["x"], d["ref"], d["u"], d["u_nom"]
    m = dict(infeasible=int((~ok).sum()), unsafe=int(unsafe.sum()), unsafe_qp_feasible=int((unsafe & ok).sum()),
             h_min=float(d["h"].min()), collided=[int(i + 1) for i in range(d["h"].shape[1]) if d["h"][:, i].min() < 0],
             dropped=int((~d["mask"]).any(1).sum()) if "mask" in d else 0,
             rms_tracking=float(np.sqrt(np.mean(np.sum((x[:, :2] - ref[:, :2]) ** 2, 1)))),
             rms_jerk=float(np.sqrt(np.mean(np.sum(u ** 2, 1)))),
             rms_u_dev=float(np.sqrt(np.mean(np.sum((u - un) ** 2, 1)))),
             goal_error=float(np.linalg.norm(x[-1, :2] - B.GOAL[:2])),
             ms_mean=float(d["ms"].mean()), ms_max=float(d["ms"].max()))
    if "eta" in d:
        eta, nu, kap = d["eta"], d["nu"], d["kappa"]; p = eta.shape[1]
        c_i = d["delta"].max(0); c = float(c_i.max())
        room = float(np.min(cq.NU_BAR - (d["nu_nom"][:, 0::2] + c_i[None, :])))
        eta_hat = float(cq.ETA_MIN + c * np.sqrt(cq.RHO * p * cq.S_DIAG / cq.T_DIAG))
        m.update(eta_max=float(eta.max()), kappa_dev_max=float(np.abs(kap - cq.K_NOM[None, None, :]).max()),
                 nu_abs_max=float(np.abs(nu).max()), nu_lo_max=float(d["nu_lo"].max()),
                 nu_cap_active=int((nu >= cq.NU_BAR - 1e-7).any(1).sum()),
                 psi1_min=float(d["psi1"].min()), psi2_min=float(d["psi2"].min()),
                 psi3_min=float(d["psi3"][ok].min()) if ok.any() else None,
                 # condition (a) of Proposition 1 with the smallest constants c_i, the room left in condition (b),
                 # and the resulting terminal-gain bound eta_hat
                 condition_a_c=c, condition_b_room=room, condition_b_holds=bool(room >= 0), eta_hat=eta_hat,
                 adjusted_initial_gains=d["adjusted"].tolist())
    return m


def run(keys, ablation=False):
    summary = {}
    for key in keys:
        sc = SCENARIOS[key]; summary[key] = dict(label=sc.label, p=int(len(sc.centers)), radius=sc.radius,
                                                  speed=sc.speed, track=sc.track, centers=np.asarray(sc.centers).tolist())
        runs = [("proposed", lambda: simulate_proposed(sc, "both")),
                ("fixed", lambda: simulate_baseline(sc, "fixed")), ("lcs", lambda: simulate_baseline(sc, "lcs"))]
        if ablation:
            runs += [("eta_only", lambda: simulate_proposed(sc, "eta_only")),
                     ("nu_only", lambda: simulate_proposed(sc, "nu_only"))]
        for name, f in runs:
            d = f(); np.savez_compressed(RES / f"{key}_{name}.npz", **d); summary[key][name] = metrics(d)
        s = summary[key]
        print(f"{key:4s} {sc.label:52s} fixed: inf {s['fixed']['infeasible']:3d} unsafe {s['fixed']['unsafe']:3d} | "
              f"LCS: dropped {s['lcs']['dropped']:3d} unsafe {s['lcs']['unsafe']:3d} | proposed: inf "
              f"{s['proposed']['infeasible']} min h {s['proposed']['h_min']:.3f} max eta {s['proposed']['eta_max']:.3f}",
              flush=True)
    return summary


def speed_sweep():
    rows = []
    for s in np.round(np.arange(1.0, 2.51, 0.1), 2):
        sc = Scenario(f"v{s:.1f}", f"reference {s:.1f}x faster", speed=float(s))
        rows.append(dict(speed=float(s), **{mode: metrics(simulate_proposed(sc, mode))
                                            for mode in ("both", "eta_only", "nu_only")}))
        print(f"speed {s:.1f}: max eta {rows[-1]['both']['eta_max']:.4f} (eta only {rows[-1]['eta_only']['eta_max']:.4f})",
              flush=True)
    (RES / "speed_sweep.json").write_text(json.dumps(rows, indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="all sixteen variants")
    ap.add_argument("--ablation", action="store_true", help="also eta-only and nu-only versions of the proposed QP")
    ap.add_argument("--sweep", action="store_true", help="reference-speed sweep")
    ap.add_argument("--nu-bar", type=float, default=cq.NU_BAR, help="cap on the differential inputs (Table I: 10)")
    a = ap.parse_args(argv)
    nu_bar0 = cq.NU_BAR; cq.NU_BAR = a.nu_bar
    keys = list(SCENARIOS) if a.all else list(PAPER)
    try:
        summary = run(keys, a.ablation)
    finally:
        restore()
    summary["_meta"] = dict(nu_bar=cq.NU_BAR, paper_variants=list(PAPER), samples=len(B.SAMPLE_TIMES), dt=B.DT)
    out = RES / "summary.json"
    if out.exists():                                    # keep variants written by earlier runs
        old = json.loads(out.read_text()); old.update(summary); summary = old
    out.write_text(json.dumps(summary, indent=1) + "\n")
    if a.sweep:
        try:
            speed_sweep()
        finally:
            restore()
    cq.NU_BAR = nu_bar0


if __name__ == "__main__":
    main()
