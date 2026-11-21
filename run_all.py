#!/usr/bin/env python3
"""Reproduce every number, table and figure of Section IV of the paper.

    python run_all.py              Tests 1-a, 1-b, 1-c and Figs. 1-3   (about 2 min)
    python run_all.py --videos     also the three simulation videos     (about 5 min more, needs ffmpeg)

Outputs (results/, figures/, videos/) are listed in README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
for sub in ("common", "test1a", "test1b", "test1c", "figures", "videos"):
    sys.path.insert(0, str(ROOT / sub))
import current_qp as cq                                               # noqa: E402
import test1a as A                                                    # noqa: E402
import test1b as B                                                    # noqa: E402

OUT = ROOT / "results"
K_NOM = cq.K_NOM


def timed(module, name, store):
    """Wrap module.name so that each call's wall time is appended to store (QP time per sample)."""
    f = getattr(module, name)

    def g(*a, **k):
        t0 = time.perf_counter(); r = f(*a, **k); store.append(time.perf_counter() - t0); return r
    setattr(module, name, g)
    return f


# ------------------------------------------------------------------------------------------- Test 1-a
def test_1a():
    data = A.simulate()
    summary = A.export_results(data)                                  # results/test1a/v2b2_summary.json, signals csv
    eta = np.array([q["eta"] for q in data]); kap = np.array([q["kappa"] for q in data])
    ok = np.array([q["proposed_ok"] for q in data])
    stats = dict(samples=len(data), fixed_gain_infeasible=int(sum(not q["standard_ok"] for q in data)),
                 lcs_dropped=int(sum((~np.asarray(q["lcs_mask"], bool)).any() for q in data)),
                 proposed_infeasible=int((~ok).sum()), eta_max=float(eta.max()),
                 kappa_dev_max=float(np.abs(kap - K_NOM[None, None, :]).max()),
                 nu_abs_max=float(np.abs(np.array([q["nu"] for q in data])).max()),
                 h_min_by_obstacle=np.array([q["h"] for q in data]).min(0).tolist())
    (OUT / "test1a" / "metrics.json").write_text(json.dumps(stats, indent=1) + "\n")
    return stats


# ------------------------------------------------------------------------------------------- Test 1-b
def table_ii(data, times):
    """Table II of the paper: costs and computation time of the three controllers on Test 1-b."""
    ref = np.array([q["ref"][:2] for q in data]); goal = np.array(B.GOAL[:2]); out = {}
    for c in ("standard", "lcs", "proposed"):
        x = np.array([q[f"{c}_x"] for q in data]); u = np.array([q[f"{c}_u"] for q in data])
        un = np.array([q[f"{c}_nominal"] for q in data]); h = np.array([q[f"{c}_h"] for q in data])
        ok = np.array([q[f"{c}_ok"] for q in data]); tt = np.array(times[c]) * 1e3
        inside4 = h[:, 3] < 0
        out[c] = dict(infeasible=int((~ok).sum()),
                      dropped=int(sum((~np.asarray(q["lcs_mask"], bool)).any() for q in data)) if c == "lcs" else 0,
                      inside_c4=int(inside4.sum()), inside_c4_qp_feasible=int((inside4 & ok).sum()),
                      h_min=float(h.min()), rms_tracking=float(np.sqrt(np.mean(np.sum((x[:, :2] - ref) ** 2, 1)))),
                      rms_jerk=float(np.sqrt(np.mean(np.sum(u ** 2, 1)))),
                      rms_u_dev=float(np.sqrt(np.mean(np.sum((u - un) ** 2, 1)))),
                      goal_error=float(np.linalg.norm(x[-1, :2] - goal)),
                      ms_mean=float(tt.mean()), ms_max=float(tt.max()))
    eta = np.array([q["eta"] for q in data]); kap = np.array([q["kappa"] for q in data])
    out["proposed"].update(eta_max=float(eta.max()), kappa_dev_max=float(np.abs(kap - K_NOM[None, None, :]).max()),
                           psi1_min=float(np.array([q["proposed_phi1"] for q in data]).min()),
                           psi2_min=float(np.array([q["proposed_phi2"] for q in data]).min()))
    return out


def intersample(data, n_sub=100):
    """h_i and the proposed chain on a 100-point grid inside every hold interval (exact cubic motion under the
    zero-order hold; the gains follow the held nu)."""
    dt = B.DT; s = np.linspace(0.0, dt, n_sub + 1)
    G = np.tile(np.array([K_NOM[0], 0.0, K_NOM[1]]), (len(B.CENTERS), 1))
    res = {c: dict(h_min_samples=np.inf, h_min_grid=np.inf) for c in ("standard", "lcs", "proposed")}
    chain = dict(psi1_min_samples=np.inf, psi1_min_grid=np.inf, psi2_min_samples=np.inf, psi2_min_grid=np.inf)
    for k in range(len(data) - 1):
        q = data[k]
        for c in ("standard", "lcs", "proposed"):
            x, j = q[f"{c}_x"], q[f"{c}_u"]
            p0, v0, a0 = x[:2], x[2:4], x[4:6]
            P = p0[None] + v0[None] * s[:, None] + 0.5 * a0[None] * s[:, None] ** 2 + j[None] * s[:, None] ** 3 / 6
            V = v0[None] + a0[None] * s[:, None] + 0.5 * j[None] * s[:, None] ** 2
            Acc = a0[None] + j[None] * s[:, None]
            for i, cen in enumerate(B.CENTERS):
                d = P - cen; h = np.sum(d * d, 1) - B.R ** 2
                res[c]["h_min_samples"] = min(res[c]["h_min_samples"], float(h[0]))
                res[c]["h_min_grid"] = min(res[c]["h_min_grid"], float(h.min()))
                if c == "proposed":
                    hd = 2 * np.sum(d * V, 1); hdd = 2 * (np.sum(V * V, 1) + np.sum(d * Acc, 1))
                    nu1, nu2 = q["nu"][2 * i], q["nu"][2 * i + 1]
                    k1 = G[i, 0] + G[i, 1] * s + 0.5 * nu1 * s ** 2; k1d = G[i, 1] + nu1 * s; k2 = G[i, 2] + nu2 * s
                    p1 = hd + k1 * h; p2 = hdd + k1d * h + k1 * hd + k2 * p1
                    chain["psi1_min_samples"] = min(chain["psi1_min_samples"], float(p1[0]))
                    chain["psi1_min_grid"] = min(chain["psi1_min_grid"], float(p1.min()))
                    chain["psi2_min_samples"] = min(chain["psi2_min_samples"], float(p2[0]))
                    chain["psi2_min_grid"] = min(chain["psi2_min_grid"], float(p2.min()))
        G = cq.gain_step(G, q["nu"], dt)
    res["proposed_chain"] = chain; res["grid_points_per_interval"] = n_sub
    return res


def test_1b():
    times = {"standard": [], "lcs": [], "proposed": []}
    orig = {n: timed(B, f"{n}_control", times[n]) for n in times}
    try:
        data = B.simulate()
    finally:
        for n, f in orig.items():
            setattr(B, f"{n}_control", f)
    B.flatten(data)                                                   # results/test1b/closed_loop_signals.csv
    (B.RES / "closed_loop_summary.json").write_text(json.dumps(B.summary(data), indent=2) + "\n")
    B.verification(data)                                              # results/test1b/verification.json
    t2 = table_ii(data, times)
    (OUT / "test1b" / "table_ii.json").write_text(json.dumps(t2, indent=1) + "\n")
    (OUT / "test1b" / "intersample.json").write_text(json.dumps(intersample(data), indent=1) + "\n")
    return t2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--videos", action="store_true"); a = ap.parse_args()
    t0 = time.time()
    s1a = test_1a(); print("Test 1-a:", json.dumps(s1a), flush=True)
    s1b = test_1b(); print("Test 1-b (Table II):", json.dumps(s1b, indent=1), flush=True)
    import test1c
    test1c.main([])                                                   # the six variants of Fig. 3
    import make_figures
    for f in make_figures.main():
        print(f)
    if a.videos:
        import make_videos
        make_videos.main()
    (OUT / "run_all.json").write_text(json.dumps(dict(qp_parameters=cq.PARAMETERS, seconds=round(time.time() - t0, 1)),
                                                 indent=1, default=str) + "\n")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
