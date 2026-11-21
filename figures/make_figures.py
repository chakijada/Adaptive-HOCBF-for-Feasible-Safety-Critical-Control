#!/usr/bin/env python3
"""Figures 1-3 of the paper.

    python figures/make_figures.py     (after test1a, test1b and test1c have written results/)

Fig. 1  fig1_test1a.pdf   Test 1-a: (a)-(g) path, margins, QP status, constraints kept, kappa, |nu|, eta;
                          (h)-(j) input space of the three controllers at event B.
Fig. 2  fig2_test1b.pdf   Test 1-b: (a)-(d) HOCBF chain, (e)-(g) kappa, |nu|, eta; (h) paths, (i) QP status,
                          (j) constraints kept by LCS.
Fig. 3  fig3_test1c.pdf   Test 1-c: (a)-(f) paths in six harder variants, (g)-(i) eta, |nu|, |kappa - kappa_nom|.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42      # TrueType text in PDFs (no Type 3 fonts; required by PaperPlaza)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("common", "test1a", "figures"):
    sys.path.insert(0, str(ROOT / sub))
import current_qp as cq                  # noqa: E402
import test1a as A                       # noqa: E402  Test 1-a (simulated here, pointwise)
import test1b_data as F                  # noqa: E402  Test 1-b signals (results/test1b/closed_loop_signals.csv)

OUT = HERE
STD, LCS, PROP = A.STD, A.LCS, A.PROP
RC = dict(A.RC, **{"font.size": 6.8, "axes.labelsize": 6.8, "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
                   "legend.fontsize": 5.8})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / f"{name}.png", dpi=400, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return OUT / f"{name}.pdf"


def ax_in(fig, W, H, x, y, w, h):
    """Axes from inch coordinates (x, y = lower-left corner)."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def tag(ax, s, x=0.035, y=0.965, ha="left"):
    A.panel_tag(ax, s, x=x, y=y, ha=ha, size=6.0)


# ------------------------------------------------------------------------------------------- Test 1-a pieces
DATA_A = A.simulate()
T_A = np.array([r["t"] for r in DATA_A]); A.T = T_A
EVENTS = ((A.TA, "#8073ac", "A"), (A.TB, "#2ca25f", "B"))
MARK = ("o", "s", "^")


def workspace_1a(ax):
    states = np.array([r["x"] for r in DATA_A])
    for i, c in enumerate(A.CENTERS):
        ax.add_patch(Circle(c, A.RADIUS, fc="#dcdcdc", ec="#5a5a5a", lw=0.55))
        ax.annotate(f"$c_{i+1}$", c, ha="center", va="center", fontsize=6.0)
    ax.plot(states[:, 0], states[:, 1], color="#4a4a4a", lw=0.95)
    bad = np.array([not r["standard_ok"] for r in DATA_A])
    ax.plot(states[bad, 0], states[bad, 1], ls="none", marker="o", ms=2.0, color=STD, mec="none", zorder=5)
    for time, colour, label in EVENTS:
        k = int(np.argmin(np.abs(T_A - time)))
        ax.plot(*states[k, :2], marker="*", ms=7, color=colour, mec="#333", mew=0.35, zorder=6)
        ax.annotate(label, states[k, :2], textcoords="offset points", xytext=(2.5, -8), fontsize=6.6,
                    color=colour, weight="bold")
    ax.set_aspect("equal"); ax.set_xlim(-0.95, 4.85); ax.set_ylim(-2.32, 2.32)
    ax.set_xlabel("$x$ [m]", labelpad=0.5); ax.set_ylabel("$y$ [m]")
    ax.set_xticks([0, 2, 4]); ax.set_yticks([-2, 0, 2])
    # the two magnified infeasible clusters, as in Fig. 1
    runs = A.index_runs(bad)
    boxes = ([0.315, 0.700, 0.255, 0.285], [0.640, 0.045, 0.265, 0.300]); leaders = ((0.0, 0.0), (0.0, 1.0))
    for n, (run, place, ev, ld) in enumerate(zip(runs[:2], boxes, EVENTS, leaders), 1):
        colour, label = ev[1], ev[2]
        ins = ax.inset_axes(place)
        lo, hi = max(int(run[0]) - 10, 0), min(int(run[-1]) + 10, len(DATA_A) - 1)
        ins.plot(states[lo:hi + 1, 0], states[lo:hi + 1, 1], color="#4a4a4a", lw=0.75)
        ins.plot(states[run, 0], states[run, 1], ls="none", marker="o", ms=1.9, color=STD, mec="none")
        xb, yb = states[run, 0], states[run, 1]
        m = 0.55 * max(xb.max() - xb.min(), yb.max() - yb.min(), 1e-3) + 0.02
        xl, yl = (xb.mean() - m, xb.mean() + m), (yb.mean() - m, yb.mean() + m)
        ins.set_xlim(*xl); ins.set_ylim(*yl); ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set(color=colour, lw=0.65)
        ins.text(0.06, 0.93, label, transform=ins.transAxes, ha="left", va="top", fontsize=5.8, color=colour,
                 weight="bold")
        ins.text(0.94, 0.93, rf"$\mathcal{{I}}_{n}$", transform=ins.transAxes, ha="right", va="top", fontsize=5.8,
                 color=STD)
        ax.add_patch(Rectangle((xl[0], yl[0]), xl[1] - xl[0], yl[1] - yl[0], fill=False, ec=colour, lw=0.55,
                               zorder=7))
        ax.annotate("", xy=(place[0] + ld[0] * place[2], place[1] + ld[1] * place[3]), xycoords=ax.transAxes,
                    xytext=(np.mean(xl), np.mean(yl)), textcoords="data", zorder=4,
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.5, shrinkA=3.5, shrinkB=0, alpha=0.85))


def vlines(ax):
    for time, colour, _ in EVENTS:
        ax.axvline(time, color=colour, lw=0.65, ls=(0, (1.8, 1.4)))


def margin_1a(ax):
    h = np.array([r["h"] for r in DATA_A])
    for i in range(3):
        ax.semilogy(T_A, h[:, i], color=A.OBS[i], lw=0.95, marker=MARK[i], ms=1.8, mec="none",
                    markevery=(8 + 15 * i, len(T_A) // 6 or 1))
    ax.set_ylabel("$h_i$"); ax.set_ylim(0.85, 45.0); ax.set_yticks([1, 10]); ax.set_yticklabels(["1", "10"])
    ax.set_xlim(T_A.min(), T_A.max()); ax.set_xticks([0, 1, 2]); vlines(ax)


def status_1a(ax):
    A.compact_status(ax, DATA_A)
    ax.set_ylim(0, 3.62); ax.set_xlim(T_A.min(), T_A.max()); ax.set_xticks([0, 1, 2]); vlines(ax)


def right_1a(axes, legend=True):
    masks = np.array([r["lcs_mask"] for r in DATA_A])
    kap = np.array([r["kappa"] for r in DATA_A]); nu = np.array([r["nu"] for r in DATA_A])
    eta = np.array([r["eta"] for r in DATA_A])
    a0, a1, a2, a3 = axes
    a0.step(T_A, np.full(len(T_A), 3), where="post", color=STD, lw=1.0, ls=(0, (2.4, 1.5)))
    a0.step(T_A, np.full(len(T_A), 3), where="post", color=PROP, lw=1.0)
    a0.step(T_A, masks.sum(1), where="post", color=LCS, lw=1.0)
    a0.set_ylim(1.6, 3.42); a0.set_yticks([2, 3]); a0.set_ylabel("constr.", fontsize=6.0)
    ev = len(T_A) // 6 or 1
    for i in range(3):
        mk = dict(marker=MARK[i], ms=1.8, markevery=(8 + 15 * i, ev), mec="none")
        a1.plot(T_A, kap[:, i, 0], color=A.OBS[i], lw=0.95, **mk)
        a1.plot(T_A, kap[:, i, 1], color=A.OBS[i], lw=0.95, ls=(0, (1.8, 1.2)), **mk)
        a2.plot(T_A, np.abs(nu[:, 2 * i]), color=A.OBS[i], lw=0.95)
        a2.plot(T_A, np.abs(nu[:, 2 * i + 1]), color=A.OBS[i], lw=0.95, ls=(0, (1.8, 1.2)))
        a3.plot(T_A, eta[:, i], color=A.OBS[i], lw=0.95)
    for lv in (A.K1_FIXED, A.K2_FIXED):
        a1.axhline(lv, color=A.CONST, lw=0.8, ls=(0, (3, 1.6)), zorder=1.5)
    a2.axhline(0, color=A.CONST, lw=0.8, ls=(0, (3, 1.6)), zorder=1.5)
    a3.axhline(A.ETA_MIN, color=A.CONST, lw=0.8, ls=(0, (3, 1.6)), zorder=1.5)
    a1.set_ylabel(r"$\kappa_{i\ell}$"); a1.set_ylim(1.90, max(2.80, float(kap.max()) + 0.18)); a1.set_yticks([2.0, 2.5])
    a2.set_ylabel(r"$|\nu_{i\ell}|$"); a2.set_ylim(-0.16, max(0.11, float(np.abs(nu).max()) * 1.9))
    a2.set_yticks([0, 1, 2, 3])
    a3.set_ylabel(r"$\eta_i$"); top = max(0.39, float(eta.max()) + 0.06 * (float(eta.max()) - A.ETA_MIN) + 0.002)
    st = np.floor((top - A.ETA_MIN) / 2 * 100) / 100
    a3.set_ylim(0.3485, top); a3.set_yticks(np.round([A.ETA_MIN, A.ETA_MIN + st, A.ETA_MIN + 2 * st], 2))
    for ax in axes:
        ax.set_xlim(T_A.min(), T_A.max()); ax.set_xticks([0, 1, 2]); vlines(ax)
        if ax is not a3:
            ax.tick_params(labelbottom=False)
    a3.set_xlabel("time [s]", labelpad=0.5)
    if legend:
        a2.legend(handles=[plt.Line2D([], [], color=A.OBS[i], lw=0.95, marker=MARK[i], ms=1.8, mec="none",
                                      label=f"$c_{i+1}$") for i in range(3)] +
                  [plt.Line2D([], [], color="#555", lw=0.95, label=r"$\ell{=}1$"),
                   plt.Line2D([], [], color="#555", lw=0.95, ls=(0, (1.8, 1.2)), label=r"$\ell{=}2$"),
                   plt.Line2D([], [], color=A.CONST, lw=0.8, ls=(0, (3, 1.6)), label="fix/LCS")],
                  frameon=False, ncol=3, loc="upper center", fontsize=5.3, bbox_to_anchor=(0.56, 1.02),
                  handlelength=1.0, columnspacing=0.4, labelspacing=0.12)


def event_panel(ax, controller, time=2.27, show_label=True, compact=False):
    """One cell of the event-B column of Fig. 2: the jerk plane at t = 2.27 s."""
    k = int(np.argmin(np.abs(T_A - time))); row = DATA_A[k]
    pts = np.vstack((row["standard_u"], row["lcs_u"], row["proposed_u"])); c = pts.mean(0)
    span = max(1.9 * np.abs(pts - c).max(), 9.0)
    gx = np.linspace(c[0] - span, c[0] + span, 300); gy = np.linspace(c[1] - 0.70 * span, c[1] + 0.70 * span, 300)
    X, Y = np.meshgrid(gx, gy)
    colour, label = {"standard": (STD, "fixed-gain"), "lcs": (LCS, "LCS"), "proposed": (PROP, "proposed")}[controller]
    a, beta, u, ok, mask = A.control_geometry(row, controller)
    vals = np.stack([a[i, 0] * X + a[i, 1] * Y + beta[i] for i in range(3)])
    adm = np.all(vals[np.flatnonzero(mask)] >= 0, axis=0)
    ax.contourf(X, Y, adm.astype(float), levels=[0.5, 1.5], colors=[colour], alpha=0.26)
    for i in range(3):
        kept = bool(mask[i])
        ax.contour(X, Y, vals[i], levels=[0.0], colors=[colour if kept else "#9a9a9a"],
                   linewidths=1.1 if kept else 0.8, linestyles=["-"] if kept else [(0, (2.6, 1.7))])
        if kept:
            A.halfspace_arrow(ax, a[i], beta[i], gx, gy, colour)
    fs = 5.6
    if controller == "standard" and not ok:
        note = "empty"
    elif controller == "lcs":
        ax.plot(*u, marker="s", ms=3.2, color=LCS, mec="#333", mew=0.4, zorder=8)
        dropped = [i + 1 for i in range(3) if not mask[i]]
        note = rf"drops $c_{{{dropped[0]}}}$" if dropped else "all kept"
    else:
        ax.plot(*u, marker="*", ms=6, color=PROP, mec="#333", mew=0.4, zorder=8)
        note = "all kept"
    ax.text(0.30 if compact else 0.04, 0.95, note, transform=ax.transAxes, ha="left", va="top", fontsize=fs,
            color=colour, weight="bold" if note == "empty" else "normal", zorder=10,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
    if show_label and not compact:
        ax.text(0.96, 0.95, label, transform=ax.transAxes, ha="right", va="top", fontsize=fs, weight="bold",
                color=colour, zorder=10, bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
    if compact:
        ax.yaxis.set_label_position("right")
        ax.set_ylabel({"standard": "fixed", "lcs": "LCS", "proposed": "prop."}[controller], color=colour,
                      fontsize=6.0, weight="bold", labelpad=2)
    ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[0], gy[-1])
    return T_A[k]


def fig1():
    """Fig. 1 + event-B column on the right (single column, 3.5 in)."""
    W, H = 3.5, 2.50
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W, H))
        x1, w1 = 0.30, 1.28
        st = ax_in(fig, W, H, x1, 0.25, w1, 0.36)
        mg = ax_in(fig, W, H, x1, 0.67, w1, 0.36)
        ws = ax_in(fig, W, H, x1, 1.32, w1, w1 * 4.64 / 5.80)
        x2, w2 = 1.98, 0.62
        hs = [0.40, 0.45, 0.45, 0.45]; ys = [1.96, 1.43, 0.90, 0.25]
        R = [ax_in(fig, W, H, x2, y, w2, h) for y, h in zip(ys, hs)]
        x3, w3 = 2.66, 0.66
        E = [ax_in(fig, W, H, x3, y, w3, 0.58) for y in (1.78, 1.02, 0.25)]
        workspace_1a(ws); margin_1a(mg); status_1a(st)
        mg.tick_params(labelbottom=False); st.set_xlabel("time [s]", labelpad=0.5)
        right_1a(R, legend=False)
        for ax, c in zip(E, ("standard", "lcs", "proposed")):
            tB = event_panel(ax, c, compact=True)
            ax.tick_params(labelleft=False, labelsize=5.4)
            if c != "proposed":
                ax.tick_params(labelbottom=False)
        E[2].set_xlabel(r"$j_x$ [m/s$^3$]", labelpad=0.5)
        E[0].set_title(f"event B, $t={tB:.2f}$ s", fontsize=6.0, pad=2, color="#2ca25f")
        for ax, s_ in zip([ws, mg, st] + R + E, "abcdefghij"):
            tag(ax, f"({s_})", x=0.035 if ax not in E else 0.03, y=0.965)
        R[2].legend(handles=[plt.Line2D([], [], color=A.OBS[i], lw=0.95, label=f"$c_{i+1}$") for i in range(3)] +
                    [plt.Line2D([], [], color="#555", lw=0.95, ls=(0, (1.8, 1.2)), label=r"$\ell{=}2$")],
                    frameon=False, ncol=2, loc="upper right", fontsize=4.9, handlelength=0.9, columnspacing=0.3,
                    labelspacing=0.05, borderaxespad=0.05)
        return save(fig, "fig1_test1a")


# ------------------------------------------------------------------------------------------- Test 1-b pieces
DB = F.DATA; TB_ = F.T
SER = F.SERIES                    # fixed-gain, proposed, LCS (no RT-CBF, as in the paper)


def chain_1b(axes):
    levels = ((0, r"$\psi_{i,0}$", [-0.2, 0, 1], (-0.45, 9.0)), (1, r"$\psi_{i,1}$", [-1, 0, 1], (-1.3, 22.0)),
              (2, r"$\psi_{i,2}$", [-1, 0, 1], (-5.0, 45.0)), (3, r"$\psi_{i,3}$", [-100, 0, 1], (-260.0, 30.0)))
    for (lv, lab, ticks, span), ax in zip(levels, axes):
        for prefix, colour, width, style, _ in SER:
            ax.plot(TB_, F.chain(prefix, lv), color=colour, lw=width * 0.9, ls=style)
        ax.axhline(0.0, color=F.CONST, lw=0.6, zorder=1.5)
        ax.set_yscale("symlog", linthresh=0.05); ax.set_yticks(ticks); ax.set_ylabel(lab, labelpad=0.5)
        ax.set_yticklabels([f"{v:g}".replace("-", "\u2212") for v in ticks])
        ax.set_xlim(TB_.min(), TB_.max()); ax.set_xticks([0, 5, 10]); ax.set_ylim(*span)
        if lv != 3:
            ax.tick_params(labelbottom=False)
    axes[3].axhline(F.EPS, color=F.CONST, ls=(0, (1, 1)), lw=0.7, zorder=1.5)
    axes[3].set_xlabel("time [s]", labelpad=0.5)


def gains_1b(axes, legends=True):
    ka, na, ea = axes
    for i in range(4):
        mk = dict(marker=F.MARKERS[i], ms=1.7, mec="none", markevery=(40 + 62 * i, 250))
        ka.plot(TB_, DB[f"kappa_{2*i+1}"], color=F.OBS[i], lw=0.9, **mk)
        ka.plot(TB_, DB[f"kappa_{2*i+2}"], color=F.OBS[i], lw=0.9, ls=(0, (1.8, 1.2)), **mk)
        na.plot(TB_, DB[f"nu_{2*i+1}"].abs(), color=F.OBS[i], lw=0.9)
        na.plot(TB_, DB[f"nu_{2*i+2}"].abs(), color=F.OBS[i], lw=0.9, ls=(0, (1.8, 1.2)))
        ea.plot(TB_, DB[f"eta_{i+1}"], color=F.OBS[i], lw=0.9)
    for fx in (F.K1_FIXED, F.K2_FIXED):
        ka.axhline(fx, color=F.CONST, lw=0.8, ls=(0, (3, 1.6)), zorder=1.5)
    na.axhline(0, color=F.CONST, lw=0.8, ls=(0, (3, 1.6)), zorder=1.5)
    ea.axhline(F.ETA_FIXED, color=F.CONST, lw=0.8, ls=(0, (3, 1.6)), zorder=1.5)
    ka.set_ylabel(r"$\kappa_{i\ell}$", labelpad=0.5); ka.set_ylim(1.97, 2.54); ka.set_yticks([2.0, 2.2, 2.4])
    na.set_ylabel(r"$|\nu_{i\ell}|$", labelpad=0.5); na.set_ylim(-0.01, 0.215); na.set_yticks([0, 0.1, 0.2])
    ea.set_ylabel(r"$\eta_i$", labelpad=0.5); ea.set_ylim(0.3492, 0.3560); ea.set_yticks([0.350, 0.353, 0.356])
    for ax in axes:
        ax.set_xlim(TB_.min(), TB_.max()); ax.set_xticks([0, 5, 10])
    ka.tick_params(labelbottom=False); na.tick_params(labelbottom=False); ea.set_xlabel("time [s]", labelpad=0.5)
    if legends:
        ka.legend(handles=[plt.Line2D([], [], color=F.OBS[i], lw=0.9, marker=F.MARKERS[i], ms=1.7, mec="none",
                                      label=f"$c_{i+1}$") for i in range(4)], frameon=False, ncol=2,
                  loc="upper right", fontsize=5.0, handlelength=1.0, columnspacing=0.4, labelspacing=0.1)
        na.legend(handles=[plt.Line2D([], [], color="#555", lw=0.9, label=r"$\ell{=}1$"),
                           plt.Line2D([], [], color="#555", lw=0.9, ls=(0, (1.8, 1.2)), label=r"$\ell{=}2$"),
                           plt.Line2D([], [], color=F.CONST, lw=0.8, ls=(0, (3, 1.6)), label="fix/LCS")],
                  frameon=False, loc="upper right", fontsize=5.0, handlelength=1.2, labelspacing=0.1)


def paths_1b(ax, inset=True):
    for i, c in enumerate(F.CENTERS):
        ax.add_patch(Circle(c, F.RADIUS, fc="#dcdcdc", ec="#5a5a5a", lw=0.55))
        ax.annotate(f"$c_{i+1}$", c, ha="center", va="center", fontsize=5.4)
    ax.plot(DB.ref_x, DB.ref_y, color=F.REF, lw=0.7, ls=(0, (1.0, 1.4)))
    for prefix, colour, width, style, _ in SER:
        ax.plot(DB[f"{prefix}_x"], DB[f"{prefix}_y"], color=colour, lw=width * 0.9, ls=style)
    col = F.chain("standard", 0) < 0
    ax.plot(DB.standard_x[col], DB.standard_y[col], ls="none", marker="o", ms=1.3, color="#7f1d1d", mec="none",
            zorder=6)
    ax.set_aspect("equal"); ax.set_xlim(-1.05, 4.72); ax.set_ylim(-2.34, 2.34)
    ax.set_xticks([0, 2, 4]); ax.set_yticks([-2, 0, 2]); ax.set_xlabel("$x$ [m]", labelpad=0.5)
    if inset:
        ins = ax.inset_axes([0.58, 0.03, 0.40, 0.40])
        ins.add_patch(Circle(F.CENTERS[3], F.RADIUS, fc="#dcdcdc", ec="#5a5a5a", lw=0.5))
        for prefix, colour, width, style, _ in SER:
            ins.plot(DB[f"{prefix}_x"], DB[f"{prefix}_y"], color=colour, lw=width, ls=style)
        ins.plot(DB.standard_x[col], DB.standard_y[col], ls="none", marker="o", ms=1.5, color="#7f1d1d", mec="none")
        cx, cy = DB.standard_x[col], DB.standard_y[col]
        sp = 0.62 * max(cx.max() - cx.min(), cy.max() - cy.min()) + 0.10
        ins.set_xlim(cx.mean() - sp, cx.mean() + sp); ins.set_ylim(cy.mean() - sp, cy.mean() + sp)
        ins.set_xticks([]); ins.set_yticks([])
        for s in ins.spines.values():
            s.set(color="#7f1d1d", lw=0.6)
        ins.text(0.05, 0.93, r"$h_4<0$", transform=ins.transAxes, ha="left", va="top", fontsize=5.0, color="#7f1d1d")


def status_1b(ax):
    rows = [("fixed", DB.standard_ok, STD), ("LCS", DB.lcs_ok, LCS), ("prop.", DB.proposed_ok, PROP)]
    for y, (name, okc, colour) in enumerate(reversed(rows)):
        ok = okc.astype(bool).to_numpy()
        ax.broken_barh([(TB_[s], TB_[e] - TB_[s] + 0.01) for s, e in F.runs(ok)], (y + 0.15, 0.70),
                       facecolors=colour, alpha=0.90, lw=0)
        ax.broken_barh([(TB_[s], TB_[e] - TB_[s] + 0.01) for s, e in F.runs(~ok)], (y + 0.15, 0.70),
                       facecolors="white", edgecolor=colour, lw=0.6, hatch="////")
    ax.set_ylim(0, 3.62); ax.set_yticks([0.5, 1.5, 2.5]); ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=5.4)
    ax.tick_params(axis="y", length=0); ax.set_xlim(TB_.min(), TB_.max()); ax.set_xticks([0, 5, 10])


def margin_1b(ax):
    for prefix, colour, width, style, _ in SER:
        ax.plot(TB_, F.chain(prefix, 0), color=colour, lw=width * 0.9, ls=style)
    ax.axhline(0.0, color=F.CONST, lw=0.6, zorder=1.5)
    ax.set_yscale("symlog", linthresh=0.03); ax.set_ylim(-0.35, 3.2); ax.set_yticks([-0.1, 0, 0.1, 1])
    ax.set_yticklabels(["\u22120.1", "0", "0.1", "1"])
    ax.set_ylabel(r"$\min_i h_i$", labelpad=0.5); ax.set_xlim(TB_.min(), TB_.max()); ax.set_xticks([0, 5, 10])


def keep_1b(ax):
    """Fig. 3(d): the constraints LCS retains (hatched = dropped)."""
    for i in range(4):
        dropped = ~F.KEEP[:, i]
        ax.broken_barh([(TB_[s], TB_[e] - TB_[s] + 0.01) for s, e in F.runs(~dropped)], (3 - i + 0.18, 0.64),
                       facecolors=LCS, alpha=0.72, lw=0)
        ax.broken_barh([(TB_[s], TB_[e] - TB_[s] + 0.01) for s, e in F.runs(dropped)], (3 - i + 0.18, 0.64),
                       facecolors="white", edgecolor=LCS, lw=0.6, hatch="////")
    ax.set_ylim(0, 4.62); ax.set_yticks([3.5, 2.5, 1.5, 0.5])
    ax.set_yticklabels([f"$c_{i}$" for i in (1, 2, 3, 4)], fontsize=5.6)
    ax.tick_params(axis="y", length=0); ax.set_ylabel("kept", fontsize=6.0, labelpad=1.5)
    ax.set_xlim(TB_.min(), TB_.max()); ax.set_xticks([0, 5, 10])


def legend_1b(fig, y):
    h = [plt.Line2D([], [], color=c, lw=w, ls=s, label=n) for _, c, w, s, n in F.LEGEND_SERIES]
    h += [plt.Line2D([], [], color=F.REF, lw=0.7, ls=(0, (1.0, 1.4)), label="reference"),
          plt.Line2D([], [], color="#7f1d1d", ls="none", marker="o", ms=2.2, label="$h_i<0$")]
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, fontsize=5.8, bbox_to_anchor=(0.53, y),
               handlelength=1.6, columnspacing=0.8)


def fig2():
    """Fig. 4 + Fig. 3 (a)-(c) as a third column on the right (single column, 3.5 in)."""
    W, H = 3.5, 3.05
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W, H))
        x1, w1 = 0.33, 0.78
        ch = [ax_in(fig, W, H, x1, 0.25 + (3 - r) * 0.66, w1, 0.58) for r in range(4)]
        x2, w2 = 1.50, 0.66
        gn = [ax_in(fig, W, H, x2, 0.25 + (2 - r) * 0.89, w2, 0.79) for r in range(3)]
        x3, w3 = 2.52, 0.96
        pa = ax_in(fig, W, H, x3, 2.05, w3, w3 * 4.68 / 5.77)
        st = ax_in(fig, W, H, x3, 1.17, w3, 0.55)
        mg = ax_in(fig, W, H, x3, 0.25, w3, 0.78)
        chain_1b(ch); gains_1b(gn); paths_1b(pa, inset=False); status_1b(st); keep_1b(mg)
        pa.tick_params(labelleft=False); pa.set_ylabel("")
        st.tick_params(labelbottom=False); mg.set_xlabel("time [s]", labelpad=0.5)
        # legend entries inside the panels: one controller per chain panel, markers in the paths panel
        kw = dict(frameon=False, fontsize=5.5, loc="lower right", handlelength=1.3, borderaxespad=0.1,
                  handletextpad=0.4)
        for ax, (_, c, w, st_, n) in zip(ch[:3], F.LEGEND_SERIES):
            ax.legend(handles=[plt.Line2D([], [], color=c, lw=w * 0.9, ls=st_, label=n)], **kw)
        pa.legend(handles=[plt.Line2D([], [], color=F.REF, lw=0.7, ls=(0, (1.0, 1.4)), label="reference"),
                           plt.Line2D([], [], color="#7f1d1d", ls="none", marker="o", ms=2.2, label="$h_i<0$")],
                  frameon=False, fontsize=5.4, loc="lower right", handlelength=1.4, borderaxespad=0.15,
                  handletextpad=0.3, labelspacing=0.15)
        for ax, s_ in zip(ch + gn + [pa, st, mg], "abcdefghij"):
            tag(ax, f"({s_})")
        return save(fig, "fig2_test1b")


# ------------------------------------------------------------------------------------------- Test 1-c pieces
RES_C = ROOT / "results" / "test1c"
STD_C, REF, CONST = "#ef3b2c", "#9a9a9a", "#111111"
ETA_LO, K_NOM = cq.ETA_MIN, cq.K_NOM


def load(key, mode):
    return dict(np.load(RES_C / f"{key}_{'proposed' if mode == 'both' else mode}.npz"))


class _Summary(dict):
    """Variant data (centers, radius) from results/test1c/summary.json, read on first use."""
    def __missing__(self, key):
        self.update(json.loads((RES_C / "summary.json").read_text()))
        return dict.__getitem__(self, key)


SC = _Summary()


PAPER_RC = {
    "font.size": 6.8, "axes.labelsize": 6.8, "legend.fontsize": 5.9, "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0, "axes.titlesize": 6.4, "axes.linewidth": 0.55, "xtick.major.width": 0.55,
    "ytick.major.width": 0.55, "xtick.major.size": 1.8, "ytick.major.size": 1.8, "axes.labelpad": 1.4,
    "legend.handlelength": 1.6, "legend.handletextpad": 0.4, "legend.columnspacing": 0.8,
    "legend.borderpad": 0.2, "legend.borderaxespad": 0.2,
}
SIX = ("S5", "S6", "S13", "S9", "S10", "S11")
SIX_TITLE = {"S5": r"reference $2\times$", "S6": r"poles $1.5\times$", "S7": r"$c_4$ moved 0.15 m",
             "S9": r"passage $-0.45$ m", "S10": r"radii 0.68 m", "S11": r"fifth obstacle $c_5$",
             "S13": r"poles $1.5\times$, $c_5$ left"}
SIX_COL = dict(zip(SIX, ("#1b9e77", "#d95f02", "#e7298a", "#66a61e", "#a6761d", "#4d4d4d")))
LCS6 = "#7b3294"
PROP_C = "#2171b5"


def _six_path(ax, key, letter, eta_colour=None, norm=None, title_colour=True):
    d = load(key, "both"); f = load(key, "fixed"); l = load(key, "lcs"); sc = SC[key]
    XL, YL = (-1.65, 4.75), (-2.42, 2.42)
    for i, c in enumerate(np.array(sc["centers"])):
        ax.add_patch(Circle(c, sc["radius"], fc="#e3e3e3", ec="#6a6a6a", lw=0.45, zorder=1))
        ax.annotate(f"$c_{i+1}$", c, ha="center", va="center", fontsize=5.2, zorder=2)
    ax.plot(d["ref"][:, 0], d["ref"][:, 1], color=REF, lw=0.6, ls=(0, (1.0, 1.2)), zorder=2)
    ax.plot(f["x"][:, 0], f["x"][:, 1], color=STD_C, lw=0.9, ls=(0, (2.2, 1.3)), zorder=3)
    ax.plot(l["x"][:, 0], l["x"][:, 1], color=LCS6, lw=0.9, ls=(0, (3.2, 1.0, 1.0, 1.0)), zorder=3.2)
    dr = (~l["mask"]).any(1)
    ax.plot(l["x"][dr, 0], l["x"][dr, 1], ls="none", marker="x", ms=1.9, mew=0.5, color=LCS6, zorder=3.6)
    for g in (f, l):
        bad = g["h"].min(1) < 0
        ax.plot(g["x"][bad, 0], g["x"][bad, 1], ls="none", marker="o", ms=1.0, color="#7f1d1d", mec="none",
                zorder=3.7)
    if eta_colour is None:
        ax.plot(d["x"][:, 0], d["x"][:, 1], color=PROP_C, lw=1.1, zorder=4)
    else:
        pts = d["x"][:, :2]; seg = np.stack([pts[:-1], pts[1:]], 1)
        lc = LineCollection(seg, cmap=eta_colour, norm=norm, lw=1.3, zorder=4)
        lc.set_array(d["eta"].max(1)[:-1]); ax.add_collection(lc)
    ax.plot(*d["x"][0, :2], "o", ms=1.9, color="k", zorder=5)
    ax.plot(2.0, 0.0, "*", ms=3.8, color="#b2182b", mec="none", zorder=5)
    xmin = min(f["x"][:, 0].min(), l["x"][:, 0].min())
    if xmin < XL[0]:
        ax.annotate("", xy=(XL[0] + 0.02, -0.3), xytext=(XL[0] + 0.7, -0.3),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=STD_C, mutation_scale=5), zorder=6)
    ax.set_title(letter + " " + SIX_TITLE[key], pad=1.2, color=SIX_COL[key] if title_colour else "k",
                 fontsize=6.4 if title_colour else 6.0)
    ax.set_aspect("equal"); ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.set_xticks([0, 2, 4]); ax.set_yticks([-2, 0, 2])



def _legend(fig, y, extra_eta=False):
    h = [plt.Line2D([], [], color=STD_C, lw=0.9, ls=(0, (2.2, 1.3)), label="fixed-gain"),
         plt.Line2D([], [], color=LCS6, lw=0.9, ls=(0, (3.2, 1.0, 1.0, 1.0)), label="LCS")]
    if not extra_eta:
        h.append(plt.Line2D([], [], color=PROP_C, lw=1.1, label="proposed"))
    h += [plt.Line2D([], [], color="#7f1d1d", ls="none", marker="o", ms=2.0, label="$h_i<0$"),
          plt.Line2D([], [], color=LCS6, ls="none", marker="x", ms=2.6, mew=0.6, label="dropped"),
          plt.Line2D([], [], color=REF, lw=0.6, ls=(0, (1.0, 1.2)), label="reference")]
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, fontsize=5.6, bbox_to_anchor=(0.53, y),
               handlelength=1.4, columnspacing=0.6, handletextpad=0.3)


def _channels(axs, lt="abc"):
    for key in SIX:
        d = load(key, "both"); t = d["t"]; kw = dict(color=SIX_COL[key], lw=0.85)
        axs[0].plot(t, d["eta"].max(1), **kw)
        axs[1].plot(t, np.abs(d["nu"]).max(1), **kw)
        axs[2].plot(t, np.abs(d["kappa"] - K_NOM[None, None, :]).max((1, 2)), **kw)
    axs[0].axhline(ETA_LO, color=CONST, lw=0.6, ls=(0, (3, 1.6)), zorder=1)
    axs[0].set_ylim(0.34, 0.63); axs[0].set_yticks([0.35, 0.45, 0.55])
    axs[1].set_ylim(-0.08, 3.7); axs[1].set_yticks([0, 1, 2, 3])
    axs[2].set_ylim(-0.03, 0.9); axs[2].set_yticks([0, 0.4, 0.8])
    for ax, lab, s in zip(axs, (r"$\max_i\eta_i$", r"$\max|\nu_{i\ell}|$", r"$\max|\kappa_{i\ell}-\kappa_{{\rm nom}}|$"),
                          lt):
        ax.set_xlim(0, 5.0); ax.set_xticks([0, 2.5, 5]); ax.set_xticklabels(["0", "2.5", "5"])
        ax.set_xlabel("time [s]"); ax.set_ylabel(lab, labelpad=1.0)
        ax.text(0.95, 0.94, f"({s})", transform=ax.transAxes, ha="right", va="top", fontsize=6.0, weight="bold")
    axs[2].yaxis.label.set_size(6.2)


def fig3():
    """Option A: 2x3 paths (full workspace) + eta, |nu|, |kappa-kappa_nom| for the six cases."""
    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(3.45, 3.22))
        top = fig.add_gridspec(2, 3, left=0.075, right=0.995, top=0.915, bottom=0.355, wspace=0.06, hspace=0.20)
        bot = fig.add_gridspec(1, 3, left=0.085, right=0.995, top=0.255, bottom=0.085, wspace=0.42)
        P = [fig.add_subplot(top[r, c]) for r in range(2) for c in range(3)]
        for ax, key, s in zip(P, SIX, ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")):
            _six_path(ax, key, s)
        for k, ax in enumerate(P):
            if k % 3:
                ax.tick_params(labelleft=False)
            if k < 3:
                ax.tick_params(labelbottom=False)
        P[0].set_ylabel("$y$ [m]"); P[3].set_ylabel("$y$ [m]")
        for ax in P[3:]:
            ax.set_xlabel("$x$ [m]", labelpad=0.5)
        _legend(fig, 1.005)
        _channels([fig.add_subplot(bot[0, k]) for k in range(3)], lt="ghi")
        return save(fig, "fig3_test1c")


def main():
    return [fig1(), fig2(), fig3()]


if __name__ == "__main__":
    for f in main():
        print(f)
