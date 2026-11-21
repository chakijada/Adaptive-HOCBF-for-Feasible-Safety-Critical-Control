#!/usr/bin/env python3
"""v2_b2: three-controller pointwise geometry test from the v3_e2 state.

All controllers evaluate the same prescribed state at every timestamp.  The
state begins at the exact v3_e2 initial condition and reaches the original
v2_b event-B state through one C2-smooth segment.  Its first crossing
of the C1-C2 centerline supplies event A; event B remains the C1-C3 midpoint.
C4 is not present.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import quadprog
from scipy.optimize import linprog
import os, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import current_qp as cq                      # the repaired (current) QP, shared with Application 1-b
MODE = os.environ.get("ACC_MODE", "both")      # ablation: both | eta_only | nu_only


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "test1a"
PDF_OUT = PNG_OUT = ROOT / "figures"           # only used by the legacy figure helpers
RES.mkdir(parents=True, exist_ok=True)

DT, EPS, ETA_MIN, KCAP = 0.01, 0.02, 0.35, 8.0
C1, C2, RADIUS = 4.0, 5.0, 0.58
TA, TB, TF = 0.23, 2.27, 2.77
K1_FIXED, K2_FIXED = 2.3, 2.0
CENTERS = np.array([[0.0, -1.7], [0.0, 1.7], [4.0, 1.7]])
X0 = np.array([
    -0.19430812336413172, 0.2761703758910268,
     0.9424633499916794, -1.8775789743224622,
    -0.5658242954692699, 3.4880128080005077,
])
PB = np.array([2.0, 0.0]); VB = np.array([1.38362536, -1.03078988]); AB = np.array([-3.9852242, 3.97578914])

COL = 3.50
RC = {
    "font.size": 6.8, "axes.labelsize": 6.8, "legend.fontsize": 5.9,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
    "axes.linewidth": 0.55, "xtick.major.width": 0.55,
    "ytick.major.width": 0.55, "xtick.major.size": 1.8,
    "ytick.major.size": 1.8, "axes.labelpad": 1.5,
    "legend.handlelength": 1.3, "legend.handletextpad": 0.45,
    "legend.columnspacing": 0.7, "legend.borderpad": 0.25,
    "legend.borderaxespad": 0.25,
}
STD, LCS, PROP = "#ef3b2c", "#e08214", "#2171b5"
OBS = ["#1f78b4", "#33a02c", "#6a3d9a"]
CONST = "#111111"


@dataclass
class BarrierRow:
    h: float
    psi1: float
    psi2: float
    au: np.ndarray
    anu: np.ndarray
    drift: float


def quintic_coeff(xa, xb, duration):
    """Coefficients for position/velocity/acceleration boundary data."""
    matrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, duration, duration**2, duration**3, duration**4, duration**5],
        [0, 1, 2*duration, 3*duration**2, 4*duration**3, 5*duration**4],
        [0, 0, 2, 6*duration, 12*duration**2, 20*duration**3],
    ], dtype=float)
    boundary = np.vstack((xa[:2], xa[2:4], xa[4:6], xb[:2], xb[2:4], xb[4:6]))
    return np.linalg.solve(matrix, boundary)


TRAJECTORY_COEF = np.vstack((
    quintic_coeff(X0, np.r_[PB, VB, AB], TB), np.zeros((1, 2))))
# t^3(t-TB)^3 and its first two derivatives vanish at both endpoints.  A tiny
# multiple therefore places the first x=0 crossing exactly on the dt=0.01 grid
# at t=TA without changing either endpoint position, velocity, or acceleration.
_endpoint_null = np.zeros(7)
_endpoint_null[3:] = [-TB**3, 3*TB**2, -3*TB, 1.0]
_at_event_a = np.array([TA**i for i in range(7)])
TRAJECTORY_COEF[:, 0] += (
    -(_at_event_a@TRAJECTORY_COEF[:, 0]) /
    (_at_event_a@_endpoint_null))*_endpoint_null


def evaluate_quintic(coef, t):
    degree = len(coef)-1
    p = np.array([t**i for i in range(degree+1)]) @ coef
    v = np.array([0.0]+[i*t**(i-1) for i in range(1, degree+1)]) @ coef
    a = np.array([0.0, 0.0]+[i*(i-1)*t**(i-2)
                              for i in range(2, degree+1)]) @ coef
    return np.r_[p, v, a]


def prescribed_state(t):
    if t <= TB:
        return evaluate_quintic(TRAJECTORY_COEF, t)
    q = t-TB
    return np.r_[PB + VB*q + 0.5*AB*q*q, VB + AB*q, AB]


def obstacle_row(x, center, gains, adaptive):
    p, v, a = x[:2], x[2:4], x[4:6]
    d = p-center
    h = float(d@d-RADIUS**2)
    hd = float(2*d@v)
    hdd = float(2*(v@v+d@a))
    k1, k1dot, k2 = gains
    psi1 = hd+k1*h
    psi2 = hdd+k1dot*h+k1*hd+k2*psi1
    au = 2*d
    anu = np.array([h, psi1]) if adaptive else np.empty(0)
    drift = float(6*v@a + 2*k1dot*hd + k1*hdd +
                  k2*(hdd+k1dot*h+k1*hd))
    return BarrierRow(h, psi1, psi2, au, anu, drift)


def solve_qp(weights, reference, rows, rhs, lower, upper):
    hessian = np.diag(weights)+1e-10*np.eye(len(weights))
    linear = np.diag(weights)@reference
    blocks, bounds = [rows], [rhs]
    for i, value in enumerate(lower):
        if np.isfinite(value):
            e = np.zeros(len(weights)); e[i] = 1.0
            blocks.append(e[None]); bounds.append(np.array([value]))
    for i, value in enumerate(upper):
        if np.isfinite(value):
            e = np.zeros(len(weights)); e[i] = -1.0
            blocks.append(e[None]); bounds.append(np.array([-value]))
    try:
        return quadprog.solve_qp(hessian, linear, np.vstack(blocks).T,
                                 np.concatenate(bounds))[0], True
    except ValueError:
        return reference.copy(), False


def fixed_geometry(x):
    gains = np.array([K1_FIXED, 0.0, K2_FIXED])
    rows = [obstacle_row(x, center, gains, False) for center in CENTERS]
    au = np.array([row.au for row in rows])
    beta = np.array([row.drift+ETA_MIN*row.psi2-EPS for row in rows])
    return rows, au, beta


def fixed_control(x):
    rows, au, beta = fixed_geometry(x)
    u, ok = solve_qp(np.ones(2), np.zeros(2), au, -beta,
                     np.full(2, -np.inf), np.full(2, np.inf))
    return u, ok, rows, au, beta, au@u+beta


def nullspace_basis(matrix, tolerance=1e-10):
    _, singular, vh = np.linalg.svd(np.asarray(matrix, float), full_matrices=True)
    rank = int(np.sum(singular > tolerance*(singular[0] if len(singular) else 1.0)))
    return vh[rank:].T.copy()


def configuration_feasible(au, beta, mask):
    mask = np.asarray(mask, bool)
    paper_a = -np.asarray(au, float).T
    null = nullspace_basis(paper_a)
    if null.shape[1] == 0:
        return True, np.zeros(len(beta))
    equality = null.T@beta
    cost = mask.astype(float)-0.5
    variable_bounds = [(0.0, None) if keep else (None, None) for keep in mask]
    solution = linprog(cost, A_eq=null.T, b_eq=equality,
                       bounds=variable_bounds, method="highs")
    if solution.success:
        return True, solution.x
    phase = linprog(np.zeros(len(beta)), A_eq=null.T, b_eq=equality,
                    bounds=variable_bounds, method="highs")
    return (True, phase.x) if phase.success else (False, None)


def lcs_step(au, beta, previous_mask, previous_nu):
    mask = previous_mask.copy()
    ok, nu = configuration_feasible(au, beta, mask)
    was_infeasible = not ok
    if ok:
        score = nu if nu is not None else previous_nu
        for index in sorted(np.flatnonzero(~mask), key=lambda i: score[i], reverse=True):
            trial = mask.copy(); trial[index] = True
            trial_ok, trial_nu = configuration_feasible(au, beta, trial)
            if trial_ok:
                mask, nu = trial, trial_nu
    else:
        for index in sorted(np.flatnonzero(mask), key=lambda i: previous_nu[i]):
            trial = mask.copy(); trial[index] = False
            trial_ok, trial_nu = configuration_feasible(au, beta, trial)
            if trial_ok:
                mask, nu, ok = trial, trial_nu, True
                break
    if was_infeasible and ok:
        for index in sorted(np.flatnonzero(~mask), key=lambda i: nu[i], reverse=True):
            trial = mask.copy(); trial[index] = True
            trial_ok, trial_nu = configuration_feasible(au, beta, trial)
            if trial_ok:
                mask, nu = trial, trial_nu
    ok, nu = configuration_feasible(au, beta, mask)
    return mask, nu, ok


def lcs_control(x, previous_mask, previous_nu):
    rows, au, beta = fixed_geometry(x)
    mask, nu, selection_ok = lcs_step(au, beta, previous_mask, previous_nu)
    if np.any(mask):
        u, qp_ok = solve_qp(np.ones(2), np.zeros(2), au[mask], -beta[mask],
                            np.full(2, -np.inf), np.full(2, np.inf))
    else:
        u, qp_ok = np.zeros(2), True
    return u, bool(selection_ok and qp_ok), rows, au, beta, au@u+beta, mask, nu


def proposed_control(x, gains, lock_initial_eta=False):
    """The current QP without actuation bounds (common/current_qp.py); lock_initial_eta is ignored."""
    rows = [obstacle_row(x, center, gains[i], True)
            for i, center in enumerate(CENTERS)]
    drift = np.array([row.drift for row in rows])
    t0 = time.perf_counter()
    z, ok, matrix, _ = cq.solve(np.array([row.au for row in rows]), np.array([row.anu for row in rows]),
                                np.array([row.psi2 for row in rows]), drift, gains, np.zeros(2), MODE)
    proposed_control.last_time = time.perf_counter() - t0
    return z, ok, rows, matrix, drift, matrix@z+drift


def simulate():
    times = np.round(np.arange(0.0, TF+0.5*DT, DT), 10)
    gains = np.tile(np.array([K1_FIXED, 0.0, K2_FIXED]), (3, 1))
    mask = np.ones(3, dtype=bool)
    lcs_nu = np.ones(3)
    output = []
    for k, t in enumerate(times):
        x = prescribed_state(float(t))
        std_u, std_ok, std_rows, std_a, std_b, std_res = fixed_control(x)
        lcs_u, lcs_ok, _, _, _, lcs_res, mask, lcs_nu = lcs_control(x, mask, lcs_nu)
        z, prop_ok, prop_rows, prop_a, prop_drift, prop_terminal = proposed_control(
            x, gains, lock_initial_eta=(k == 0))
        nu, eta = z[2:8], z[8:]
        prop_offset = np.array([
            prop_rows[i].anu@nu[2*i:2*i+2] + prop_rows[i].psi2*eta[i] +
            prop_rows[i].drift-EPS for i in range(3)])
        output.append({
            "t": float(t), "x": x.copy(),
            "standard_u": std_u.copy(), "standard_ok": bool(std_ok),
            "standard_res": std_res.copy(), "std_A": std_a.copy(),
            "std_beta": std_b.copy(),
            "lcs_u": lcs_u.copy(), "lcs_ok": bool(lcs_ok),
            "lcs_res": lcs_res.copy(), "lcs_mask": mask.copy(),
            "lcs_nu": lcs_nu.copy(),
            "proposed_u": z[:2].copy(), "proposed_ok": bool(prop_ok),
            "prop_Au": np.array([row.au for row in prop_rows]),
            "prop_offset": prop_offset, "nu": nu.copy(), "eta": eta.copy(),
            "kappa": gains[:, [0, 2]].copy(),
            "h": np.array([row.h for row in prop_rows]),
            "psi1": np.array([row.psi1 for row in prop_rows]),
            "psi2": np.array([row.psi2 for row in prop_rows]),
            "psi3": prop_terminal.copy(),
            "standard_h": np.array([row.h for row in std_rows]),
            "standard_psi1": np.array([row.psi1 for row in std_rows]),
            "standard_psi2": np.array([row.psi2 for row in std_rows]),
            "standard_psi3": std_res+EPS,
        })
        if k < len(times)-1:
            gains = cq.gain_step(gains, nu, DT)                  # exact zero-order-hold update of the gain chains
    return output


def index_runs(mask):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    return list(np.split(ids, np.flatnonzero(np.diff(ids) > 1)+1))


def farkas_certificate(row, pair):
    active = np.array(pair, dtype=int)
    pair_rows = row["std_A"][active]
    pair_beta = row["std_beta"][active]
    certificate = linprog(
        pair_beta,
        A_eq=np.vstack((pair_rows.T, np.ones((1, len(active))))),
        b_eq=np.r_[np.zeros(2), 1.0], bounds=[(0.0, None)]*len(active),
        method="highs")
    if not certificate.success:
        raise RuntimeError(f"No normalized pair certificate for {pair}")
    lam = np.zeros(3); lam[active] = certificate.x
    return {
        "t": row["t"], "pair": [i+1 for i in pair],
        "state": row["x"].tolist(), "lambda": lam.tolist(),
        "A_u": row["std_A"].tolist(), "beta": row["std_beta"].tolist(),
        "A_u_transpose_lambda": (row["std_A"].T@lam).tolist(),
        "beta_transpose_lambda": float(row["std_beta"]@lam),
        "standard_feasible": bool(row["standard_ok"]),
        "lcs_mask": row["lcs_mask"].astype(int).tolist(),
        "proposed_feasible": bool(row["proposed_ok"]),
    }


def export_results(data):
    times = np.array([row["t"] for row in data])
    event_a = int(np.argmin(np.abs(times-TA)))
    event_b = int(np.argmin(np.abs(times-TB)))
    bad = np.array([not row["standard_ok"] for row in data])
    masks = np.array([row["lcs_mask"] for row in data])
    initial_chains = {}
    for name in ("standard", "lcs", "proposed"):
        if name == "proposed":
            values = [data[0]["h"], data[0]["psi1"], data[0]["psi2"], data[0]["psi3"]]
        else:
            values = [data[0]["standard_h"], data[0]["standard_psi1"],
                      data[0]["standard_psi2"], data[0]["standard_psi3"]]
        initial_chains[name] = {
            "psi_0_h": values[0].tolist(), "psi_1": values[1].tolist(),
            "psi_2": values[2].tolist(), "psi_3": values[3].tolist(),
            "minimum": float(min(np.min(v) for v in values)),
            "all_nonnegative": bool(all(np.all(v >= -1e-10) for v in values)),
        }
    summary = {
        "version": "v2_b2 current QP (acc_27_revision)", "qp_mode": MODE, "qp_parameters": cq.PARAMETERS, "dt": DT, "samples": len(data), "obstacles": 3,
        "experiment_type": "pointwise common-state evaluation; controller outputs do not alter the prescribed motion",
        "same_pointwise_state_all_controllers": True,
        "c4_removed": True,
        "common_initial_state": X0.tolist(),
        "common_initial_gains": {"kappa_1": K1_FIXED, "kappa_1_dot": 0.0,
                                 "kappa_2": K2_FIXED, "eta": ETA_MIN},
        "initial_chain_by_controller": initial_chains,
        "all_controllers_complete_chain_nonnegative_at_t0": bool(
            all(item["all_nonnegative"] for item in initial_chains.values())),
        "minimum_physical_margin_over_prescribed_path": {
            f"C{i+1}": float(min(row["h"][i] for row in data))
            for i in range(3)
        },
        "standard_null_samples": int(bad.sum()),
        "standard_null_intervals": [[float(times[r[0]]), float(times[r[-1]]), int(len(r))]
                                    for r in index_runs(bad)],
        "lcs_null_samples": int(sum(not row["lcs_ok"] for row in data)),
        "proposed_null_samples": int(sum(not row["proposed_ok"] for row in data)),
        "lcs_drop_intervals": {f"C{i+1}": [[float(times[r[0]]), float(times[r[-1]]), int(len(r))]
                                             for r in index_runs(~masks[:, i])]
                               for i in range(3)},
        "event_A": farkas_certificate(data[event_a], (0, 1)),
        "event_B": farkas_certificate(data[event_b], (0, 2)),
        "proposed_eta_at_t0": data[0]["eta"].tolist(),
        "proposed_kappa_at_t0": data[0]["kappa"].tolist(),
    }
    (RES / "v2b2_summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    header = ["t", "x", "y", "vx", "vy", "ax", "ay", "standard_ok",
              "lcs_ok", "proposed_ok", "standard_jx", "standard_jy",
              "lcs_jx", "lcs_jy", "proposed_jx", "proposed_jy"]
    for stem, count in (("lcs_keep", 3), ("standard_res", 3), ("lcs_res", 3),
                        ("nu", 6), ("eta", 3), ("kappa", 6),
                        ("h", 3), ("psi1", 3), ("psi2", 3), ("psi3", 3)):
        header.extend([f"{stem}_{i+1}" for i in range(count)])
    with open(RES / "v2b2_signals.csv", "w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(header)
        for row in data:
            writer.writerow([
                row["t"], *row["x"], row["standard_ok"], row["lcs_ok"],
                row["proposed_ok"], *row["standard_u"], *row["lcs_u"],
                *row["proposed_u"], *row["lcs_mask"].astype(int),
                *row["standard_res"], *row["lcs_res"], *row["nu"],
                *row["eta"], *row["kappa"].ravel(), *row["h"],
                *row["psi1"], *row["psi2"], *row["psi3"],
            ])
    return summary


def save_figure(fig, name):
    pdf = PDF_OUT / f"{name}.pdf"
    png = PNG_OUT / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.006)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.006)
    plt.close(fig)
    return pdf, png


def compact_status(ax, data):
    ok_arrays = [np.array([row["standard_ok"] for row in data]),
                 np.array([row["lcs_ok"] for row in data]),
                 np.array([row["proposed_ok"] for row in data])]
    for y, (ok, colour) in enumerate(zip(reversed(ok_arrays),
                                          reversed((STD, LCS, PROP)))):
        ax.broken_barh([(T[s], T[e]-T[s]+DT) for s, e in
                        [(r[0], r[-1]) for r in index_runs(ok)]],
                       (y+0.15, 0.70), facecolors=colour, alpha=0.90, lw=0)
        ax.broken_barh([(T[s], T[e]-T[s]+DT) for s, e in
                        [(r[0], r[-1]) for r in index_runs(~ok)]],
                       (y+0.15, 0.70), facecolors="white", edgecolor=colour,
                       lw=0.65, hatch="////")
    ax.set_ylim(0, 3); ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["prop.", "LCS", "fixed"], fontsize=5.8)
    ax.tick_params(axis="y", length=0)


def panel_tag(axis, tag, x=0.032, y=0.982, ha="left", va="top", size=6.2):
    """Small (a), (b), ... identifier drawn inside an axes."""
    axis.text(x, y, tag, transform=axis.transAxes, ha=ha, va=va,
              fontsize=size, weight="bold", color="#111111", zorder=12,
              bbox=dict(fc="white", ec="none", alpha=0.72, pad=0.9))


def figure1(data, summary):
    global T
    T = np.array([row["t"] for row in data])
    states = np.array([row["x"] for row in data])
    masks = np.array([row["lcs_mask"] for row in data])
    hvals = np.array([row["h"] for row in data])
    events = ((TA, "#8073ac", "A"), (TB, "#2ca25f", "B"))
    markers = ("o", "s", "^")

    # Explicit panel boxes: the left column is packed from the top so that the
    # equal-aspect workspace leaves no vertical hole above or below it, and the
    # two columns end level at the bottom.
    height = 2.69
    x_left, w_left = 0.108, 0.42137
    x_right, w_right = 0.671, 0.32414
    top = 0.970
    def box(offset_in, height_in):
        return [x_left, top-(offset_in+height_in)/height,
                w_left, height_in/height]

    with plt.rc_context(dict(RC, **{"font.size": 7.2, "axes.labelsize": 7.2,
                                   "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
                                   "legend.fontsize": 6.2})):
        fig = plt.figure(figsize=(COL, height))
        workspace = fig.add_axes(box(0.0, 1.1798))
        margin_ax = fig.add_axes(box(1.4798, 0.3996))
        status = fig.add_axes(box(1.9544, 0.3996))
        grid = fig.add_gridspec(4, 1, left=x_right, right=x_right+w_right,
                                top=top, bottom=0.100, hspace=0.55,
                                height_ratios=[1.18, 1.0, 1.0, 1.0])
        right = [fig.add_subplot(grid[r, 0]) for r in range(4)]

        for i, center in enumerate(CENTERS):
            workspace.add_patch(Circle(center, RADIUS, fc="#dcdcdc",
                                       ec="#5a5a5a", lw=0.6))
            workspace.annotate(f"$c_{i+1}$", center, ha="center", va="center",
                               fontsize=6.4)
        workspace.plot(states[:, 0], states[:, 1], color="#4a4a4a", lw=1.0)
        bad = np.array([not row["standard_ok"] for row in data])
        workspace.plot(states[bad, 0], states[bad, 1], ls="none", marker="o",
                       ms=2.2, color=STD, mec="none", zorder=5)
        for time, colour, label in events:
            index = int(np.argmin(np.abs(T-time)))
            workspace.plot(states[index, 0], states[index, 1], marker="*", ms=8,
                           color=colour, mec="#333", mew=0.35, zorder=6)
            workspace.annotate(label, states[index, :2], textcoords="offset points",
                               xytext=(2.5, -8), fontsize=7.0, color=colour,
                               weight="bold")
        workspace.set_aspect("equal")
        workspace.set_xlim(-0.95, 4.85); workspace.set_ylim(-2.32, 2.32)
        workspace.set_xlabel("$x$ [m]"); workspace.set_ylabel("$y$ [m]")
        workspace.set_xticks([0, 2, 4]); workspace.set_yticks([-2, 0, 2])

        # The two magnifications occupy the empty top and bottom-right corners
        # of the workspace; a boxed leader ties each one to its event.
        bad_runs = index_runs(bad)
        boxes = ([0.315, 0.700, 0.255, 0.285], [0.640, 0.045, 0.265, 0.300])
        leaders = ((0.0, 0.0), (0.0, 1.0))
        for number, (run, place, event, leader) in enumerate(
                zip(bad_runs[:2], boxes, events, leaders), 1):
            colour, label = event[1], event[2]
            inset = workspace.inset_axes(place)
            lo, hi = max(int(run[0])-10, 0), min(int(run[-1])+10, len(data)-1)
            inset.plot(states[lo:hi+1, 0], states[lo:hi+1, 1],
                       color="#4a4a4a", lw=0.8)
            inset.plot(states[run, 0], states[run, 1], ls="none", marker="o",
                       ms=2.1, color=STD, mec="none")
            xbad, ybad = states[run, 0], states[run, 1]
            margin = 0.55*max(xbad.max()-xbad.min(), ybad.max()-ybad.min(), 1e-3)+0.02
            xlim = (xbad.mean()-margin, xbad.mean()+margin)
            ylim = (ybad.mean()-margin, ybad.mean()+margin)
            inset.set_xlim(*xlim); inset.set_ylim(*ylim)
            inset.set_xticks([]); inset.set_yticks([])
            inset.set_facecolor("white")
            for spine in inset.spines.values():
                spine.set(color=colour, lw=0.7)
            inset.text(0.06, 0.93, label, transform=inset.transAxes,
                       ha="left", va="top", fontsize=6.2, color=colour,
                       weight="bold")
            inset.text(0.94, 0.93, rf"$\mathcal{{I}}_{number}$",
                       transform=inset.transAxes, ha="right", va="top",
                       fontsize=6.2, color=STD)
            workspace.add_patch(Rectangle(
                (xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                fill=False, ec=colour, lw=0.6, zorder=7))
            workspace.annotate(
                "", xy=(place[0]+leader[0]*place[2],
                        place[1]+leader[1]*place[3]), xycoords=workspace.transAxes,
                xytext=(0.5*(xlim[0]+xlim[1]), 0.5*(ylim[0]+ylim[1])),
                textcoords="data", zorder=4,
                arrowprops=dict(arrowstyle="-", color=colour, lw=0.5,
                                shrinkA=3.5, shrinkB=0, alpha=0.85))

        for i in range(3):
            margin_ax.semilogy(T, hvals[:, i], color=OBS[i], lw=1.0,
                               marker=markers[i], ms=1.9, mec="none",
                               markevery=(8+15*i, len(T)//6 or 1))
        margin_ax.set_ylabel("$h_i$")
        margin_ax.set_ylim(0.85, 45.0)
        margin_ax.set_yticks([1, 10]); margin_ax.set_yticklabels(["1", "10"])
        margin_ax.tick_params(labelbottom=False)

        compact_status(status, data)
        status.set_ylim(0, 3.62)
        status.set_xlabel("time [s]")
        for axis in (margin_ax, status):
            axis.set_xlim(T.min(), T.max()); axis.set_xticks([0, 1, 2])
            for time, colour, _ in events:
                axis.axvline(time, color=colour, lw=0.7, ls=(0, (1.8, 1.4)))

        right[0].step(T, np.full(len(T), 3), where="post", color=STD,
                      lw=1.1, ls=(0, (2.4, 1.5)))
        right[0].step(T, np.full(len(T), 3), where="post", color=PROP, lw=1.1)
        right[0].step(T, masks.sum(1), where="post", color=LCS, lw=1.1)
        right[0].set_ylim(1.6, 3.42); right[0].set_yticks([2, 3])
        right[0].set_ylabel("constraints", fontsize=5.9)
        kappa = np.array([row["kappa"] for row in data])
        nu = np.array([row["nu"] for row in data])
        eta = np.array([row["eta"] for row in data])
        every = len(T)//6 or 1
        for i in range(3):
            mark = dict(marker=markers[i], ms=1.9, markevery=(8+15*i, every),
                        mec="none")
            right[1].plot(T, kappa[:, i, 0], color=OBS[i], lw=1.0, **mark)
            right[1].plot(T, kappa[:, i, 1], color=OBS[i], lw=1.0,
                          ls=(0, (1.8, 1.2)), **mark)
            right[2].plot(T, np.abs(nu[:, 2*i]), color=OBS[i], lw=1.0)
            right[2].plot(T, np.abs(nu[:, 2*i+1]), color=OBS[i], lw=1.0,
                          ls=(0, (1.8, 1.2)))
            right[3].plot(T, eta[:, i], color=OBS[i], lw=1.0)
        for level in (K1_FIXED, K2_FIXED):
            right[1].plot(T, np.full_like(T, level), color=CONST, lw=0.9,
                          ls=(0, (3, 1.6)), zorder=1.5)
        right[2].plot(T, np.zeros_like(T), color=CONST, lw=0.9,
                      ls=(0, (3, 1.6)), zorder=1.5)
        right[3].plot(T, np.full_like(T, ETA_MIN), color=CONST, lw=0.9,
                      ls=(0, (3, 1.6)), zorder=1.5)
        right[1].set_ylabel(r"$\kappa_{i\ell}$")
        right[1].set_ylim(1.90, max(2.80, float(kappa.max())+0.18))
        right[1].set_yticks([2.0, 2.5])
        right[2].set_ylabel(r"$|\nu_{i\ell}|$")
        nu_top = max(0.11, float(np.abs(nu).max())*1.90)
        right[2].set_ylim(-0.16, nu_top); right[2].set_yticks([0, 1, 2, 3])
        right[3].set_ylabel(r"$\eta_i$")
        eta_top = max(0.390, float(eta.max())+0.06*(float(eta.max())-ETA_MIN)+0.002)   # headroom above the peak
        step = np.floor((eta_top-ETA_MIN)/2*100)/100
        right[3].set_ylim(0.3485, eta_top); right[3].set_yticks(np.round([ETA_MIN, ETA_MIN+step, ETA_MIN+2*step], 2))
        right[3].set_xlabel("time [s]")
        right[2].legend(
            handles=[plt.Line2D([], [], color=OBS[i], lw=1.0, marker=markers[i],
                                ms=1.9, mec="none", label=f"$c_{i+1}$")
                     for i in range(3)] +
                    [plt.Line2D([], [], color="#555", lw=1.0, label=r"$\ell{=}1$"),
                     plt.Line2D([], [], color="#555", lw=1.0,
                                ls=(0, (1.8, 1.2)), label=r"$\ell{=}2$"),
                     plt.Line2D([], [], color=CONST, lw=0.9,
                                ls=(0, (3, 1.6)), label="fix/LCS")],
            frameon=False, ncol=3, loc="upper center", fontsize=5.7,
            bbox_to_anchor=(0.56, 1.00), handlelength=1.1,
            columnspacing=0.50, labelspacing=0.16)
        for axis in right:
            axis.set_xlim(T.min(), T.max()); axis.set_xticks([0, 1, 2])
            for time, colour, _ in events:
                axis.axvline(time, color=colour, lw=0.7, ls=(0, (1.8, 1.4)))
            if axis is not right[3]:
                axis.tick_params(labelbottom=False)
        panel_tag(workspace, "(a)")
        panel_tag(margin_ax, "(b)")
        panel_tag(status, "(c)")
        for axis, tag in zip(right, ("(d)", "(e)", "(f)")):
            panel_tag(axis, tag)
        panel_tag(right[3], "(g)", x=0.42)                       # clear of the eta peak at event A
        fig.align_ylabels(right)
        return save_figure(fig, "fig1_app1a")


def control_geometry(row, controller):
    if controller == "standard":
        return row["std_A"], row["std_beta"], row["standard_u"], row["standard_ok"], np.ones(3, bool)
    if controller == "lcs":
        return row["std_A"], row["std_beta"], row["lcs_u"], row["lcs_ok"], row["lcs_mask"]
    return row["prop_Au"], row["prop_offset"], row["proposed_u"], row["proposed_ok"], np.ones(3, bool)


def halfspace_arrow(axis, a, b, gx, gy, colour):
    a = np.asarray(a, float); norm = np.linalg.norm(a)
    if norm < 1e-12:
        return
    normal, tangent = a/norm, np.array([-a[1], a[0]])/norm
    center = np.array([0.5*(gx[0]+gx[-1]), 0.5*(gy[0]+gy[-1])])
    foot = center + ((-b-a@center)/norm**2)*a
    span = max(gx[-1]-gx[0], gy[-1]-gy[0])
    candidates = foot+np.linspace(-span, span, 501)[:, None]*tangent
    inside = ((candidates[:, 0] >= gx[0]) & (candidates[:, 0] <= gx[-1]) &
              (candidates[:, 1] >= gy[0]) & (candidates[:, 1] <= gy[-1]))
    if not inside.any():
        return
    anchor = candidates[inside].mean(axis=0)
    axis.annotate("", xy=anchor+0.13*span*normal, xytext=anchor,
                  arrowprops=dict(arrowstyle="-|>", color=colour, lw=0.85,
                                  shrinkA=0, shrinkB=0), zorder=7)


# Four snapshots inside the two infeasible clusters, t in [0.23,0.40] and
# [2.22,2.27].  The first pair carries the certificates quoted in the text and
# is the one shipped as fig2_certificates; the others are alternates.
FIG2_EVENTS = (
    ((0.23, "event A"), (2.27, "event B")),
    ((0.28, "event A"), (2.22, "event B")),
    ((0.33, "event A"), (2.24, "event B")),
    ((0.40, "event A"), (2.26, "event B")),
)


def figure2(data, events=FIG2_EVENTS[0], name="fig2_certificates"):
    times = np.array([row["t"] for row in data])
    tags = (("(a)", "(b)"), ("(c)", "(d)"), ("(e)", "(f)"))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(3, 2, figsize=(COL, 2.56),
                                 gridspec_kw=dict(hspace=0.10, wspace=0.055,
                                                  left=0.105, right=0.995,
                                                  top=0.925, bottom=0.105))
        for column, (time, title) in enumerate(events):
            index = int(np.argmin(np.abs(times-time)))
            row = data[index]
            points = np.vstack((row["standard_u"], row["lcs_u"], row["proposed_u"]))
            center = points.mean(axis=0)
            span = max(1.9*np.abs(points-center).max(), 9.0)
            gx = np.linspace(center[0]-span, center[0]+span, 380)
            gy = np.linspace(center[1]-0.70*span, center[1]+0.70*span, 380)
            X, Y = np.meshgrid(gx, gy)
            for r, (controller, colour, label) in enumerate((
                    ("standard", STD, "fixed-gain"),
                    ("lcs", LCS, "LCS"),
                    ("proposed", PROP, "proposed"))):
                axis = axes[r, column]
                a, beta, control, feasible, mask = control_geometry(row, controller)
                values = np.stack([a[i, 0]*X+a[i, 1]*Y+beta[i] for i in range(3)])
                selected = np.flatnonzero(mask)
                admissible = np.all(values[selected] >= 0, axis=0)
                axis.contourf(X, Y, admissible.astype(float), levels=[0.5, 1.5],
                              colors=[colour], alpha=0.26)
                for i in range(3):
                    kept = bool(mask[i])
                    axis.contour(X, Y, values[i], levels=[0.0],
                                 colors=[colour if kept else "#9a9a9a"],
                                 linewidths=1.2 if kept else 0.85,
                                 linestyles=["-"] if kept else [(0, (2.6, 1.7))])
                    if kept:
                        halfspace_arrow(axis, a[i], beta[i], gx, gy, colour)
                if controller == "standard" and not feasible:
                    axis.text(0.175, 0.955, "empty", transform=axis.transAxes,
                              ha="left", va="top", fontsize=6.4,
                              color=STD, weight="bold", zorder=10, bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
                elif controller == "lcs":
                    dropped = [i+1 for i in range(3) if not mask[i]]
                    axis.plot(*control, marker="s", ms=3.8, color=LCS,
                              mec="#333", mew=0.4, zorder=8)
                    axis.text(0.175, 0.955,
                              rf"drops $c_{{{dropped[0]}}}$" if dropped else "all rows kept",
                              transform=axis.transAxes, ha="left", va="top",
                              fontsize=6.0, color=LCS, zorder=10, bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
                else:
                    axis.plot(*control, marker="*", ms=7, color=PROP,
                              mec="#333", mew=0.4, zorder=8)
                    axis.text(0.175, 0.955, "all rows kept",
                              transform=axis.transAxes, ha="left", va="top",
                              fontsize=6.0, color=PROP, zorder=10, bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
                axis.set_xlim(gx[0], gx[-1]); axis.set_ylim(gy[0], gy[-1])
                if column == 0:
                    axis.set_ylabel(r"$j_y$")
                else:
                    axis.tick_params(labelleft=False)
                if r == 2:
                    axis.set_xlabel(r"$j_x$ [m/s$^3$]")
                else:
                    axis.tick_params(labelbottom=False)
                if column == 1:
                    axis.text(0.965, 0.955, label, transform=axis.transAxes,
                              ha="right", va="top", fontsize=6.4,
                              weight="bold", color=colour, zorder=10, bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
                panel_tag(axis, tags[r][column], x=0.028, y=0.955)
            axes[0, column].set_title(f"{title}, $t={times[index]:.2f}$ s",
                                      fontsize=6.6, pad=2)
        return save_figure(fig, name)


def main():
    """Simulate Test 1-a and write results/test1a/ (the paper figure is made by figures/make_figures.py)."""
    data = simulate()
    summary = export_results(data)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
