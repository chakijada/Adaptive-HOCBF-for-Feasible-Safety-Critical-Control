#!/usr/bin/env python3
"""Simulation videos of Section IV (fixed-gain HOCBF-QP, LCS and the proposed adaptive HOCBF-QP).

    python videos/make_videos.py   (needs ffmpeg; about 5 min)

Writes videos/test1a.mp4, videos/test1b.mp4, videos/test1b_chain.mp4 and videos/test1c.mp4.

Each frame of the first two shows the three controllers side by side: the workspace with the obstacle
sets {h_i < 0}; the barrier values h_i, one row per constraint; the input space (j_x, j_y) of each
controller with its terminal constraints A_{u,i} j + beta_i >= 0, the admissible set shaded and the
applied input marked (constraints dropped by LCS dashed grey; for the proposed controller beta_i
contains the repaired nu and eta, so its constraints keep a common point); the number of enforced
constraints; and the adapted gains eta_i >= eta_lo, nu_{i,l} >= nu_lo_{i,l}(theta) and kappa_{i,l}. The third
shows the complete HOCBF chain h_i, psi_{i,1}, psi_{i,2}, psi_{i,3} of each controller in Test 1-b.

Every QP enforces only the terminal level psi_{i,3} >= eps of each chain; it keeps h_i >= 0 only while the
lower levels psi_{i,1}, psi_{i,2} are nonnegative. In Test 1-b the fixed-gain controller applies the nominal
input while its QP is empty (0.23-0.49 s), which drives psi_{4,2} and psi_{4,1} negative; its QP is then feasible
again but cannot keep h_4 >= 0. The labels therefore report QP feasibility and safety separately.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.animation import FFMpegWriter                       # noqa: E402
from matplotlib.colors import ListedColormap                        # noqa: E402
from matplotlib.lines import Line2D                                 # noqa: E402
from matplotlib.patches import Circle                               # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("common", "test1a", "test1b", "test1c"):
    sys.path.insert(0, str(ROOT / sub))
import current_qp as cq                                              # noqa: E402
import test1a as A                                                   # noqa: E402  Test 1-a
import test1b as B                                                   # noqa: E402  Test 1-b

OUT = HERE
CTRL = ("standard", "lcs", "proposed")
NAME = {"standard": "fixed-gain HOCBF-QP", "lcs": "LCS (constraint selection)", "rtcbf": "RT-CBF (rate-tunable CBF)",
        "proposed": "proposed adaptive HOCBF-QP"}
SHORT = {"standard": "fixed-gain", "lcs": "LCS", "rtcbf": "RT-CBF", "proposed": "proposed"}
CCOL = {"standard": "#ef3b2c", "lcs": "#e08214", "rtcbf": "#6a51a3", "proposed": "#2171b5"}
MARK = {"standard": "o", "lcs": "s", "rtcbf": "D", "proposed": "*"}
OCOL = ["#1b9e77", "#7570b3", "#e7298a", "#66a61e"]
NOTE = (r"each QP enforces only $\psi_{i,3}\geq\varepsilon$, which keeps $h_i\geq0$ only while $\psi_{i,1},\psi_{i,2}\geq0$: "
        r"a feasible QP with a broken chain is not safe")
FORMULATION = (r"proposed QP: $u\in\mathbb{R}^2$, $\underline{\nu}(\vartheta)\leq\nu\leq\bar\nu$, $\eta\geq\underline{\eta}$, "
               r"$A_u u+A_\nu\nu+D_\psi\eta+b\geq\varepsilon\mathbf{1}$")
RC = {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
      "xtick.labelsize": 8.5, "ytick.labelsize": 8.5}


# ------------------------------------------------------------------ simulation data in one layout
def test_1a():
    A.MODE = "both"
    data = A.simulate()
    x = np.array([q["x"][:2] for q in data]); h = np.array([q["h"] for q in data]); p = h.shape[1]
    return dict(
        title="Test 1-a: prescribed trajectory, pointwise compatibility", tag="test1a", common=True, stride=1, fps=16, ctrls=CTRL,
        t=np.array([q["t"] for q in data]), centers=A.CENTERS, radius=A.RADIUS, xlim=(-1.1, 4.9), ylim=(-2.5, 2.5),
        ref=x, path={c: x for c in CTRL}, h={c: h for c in CTRL},
        ok={c: np.array([q[f"{c}_ok"] for q in data]) for c in CTRL},
        u={c: np.array([q[f"{c}_u"] for q in data]) for c in CTRL},
        mask=np.array([q["lcs_mask"] for q in data], bool),
        geom=[{"standard": (q["std_A"], q["std_beta"], np.ones(p, bool)),
               "lcs": (q["std_A"], q["std_beta"], np.asarray(q["lcs_mask"], bool)),
               "proposed": (q["prop_Au"], q["prop_offset"], np.ones(p, bool))} for q in data],
        eta=np.array([q["eta"] for q in data]), kappa=np.array([q["kappa"] for q in data]),
        nu=np.array([q["nu"] for q in data]), phase=None)


def test_1b():
    B.MODE = "both"
    data = B.simulate()
    p = len(B.CENTERS)
    T = dict(
        title="Test 1-b: closed loop through the conflicted region", tag="test1b", common=False, stride=2, fps=24,
        ctrls=CTRL,
        t=np.array([q["t"] for q in data]), centers=B.CENTERS, radius=B.R, xlim=(-1.1, 4.9), ylim=(-2.5, 2.5),
        ref=np.array([q["ref"][:2] for q in data]),
        path={c: np.array([q[f"{c}_x"][:2] for q in data]) for c in CTRL},
        h={c: np.array([q[f"{c}_h"] for q in data]) for c in CTRL},
        ok={c: np.array([q[f"{c}_ok"] for q in data]) for c in CTRL},
        u={c: np.array([q[f"{c}_u"] for q in data]) for c in CTRL},
        mask=np.array([q["lcs_mask"] for q in data], bool),
        geom=[{"standard": (q["std_A"], q["std_beta"], np.ones(p, bool)),
               "lcs": (q["lcs_A"], q["lcs_beta"], np.asarray(q["lcs_mask"], bool)),
               "proposed": (q["prop_Au"], q["prop_offset"], np.ones(p, bool))} for q in data],
        eta=np.array([q["eta"] for q in data]), kappa=np.array([q["kappa"] for q in data]),
        nu=np.array([q["nu"] for q in data]), phase=[q["mission_phase"].lower() for q in data])
    T["chain"] = {c: [np.array([q[f"{c}_{k}"] for q in data]) for k in ("h", "phi1", "phi2", "phi3")] for c in CTRL}
    return T


def add_rate_bounds(T, dt):
    """Lower rate bound nu_lo(theta) of (10) along the proposed run: rebuild the gain states
    (kappa_1, kappa_1', kappa_2) from the logged nu exactly as the simulation does."""
    p = T["nu"].shape[1] // 2
    G = np.tile(np.array([cq.K_NOM[0], 0.0, cq.K_NOM[1]]), (p, 1))
    lo = np.empty_like(T["nu"])
    for k in range(len(lo)):
        assert np.allclose(G[:, [0, 2]], T["kappa"][k], atol=1e-9), "gain reconstruction"
        lo[k] = cq.rate_box(G)[0]
        G = cq.gain_step(G, T["nu"][k], dt)
    T["nu_lo"] = lo
    assert (T["nu"] - lo).min() >= -1e-7, "nu >= nu_lo violated"
    return T


def status(T, k, c):
    """(panel text, colour, short text): QP feasibility, and in closed loop also safety and chain positivity."""
    p = len(T["centers"]); ok = bool(T["ok"][c][k]); m = T["geom"][k][c][2]
    dropped = [i + 1 for i in range(p) if not m[i]]
    if not ok:
        text, colour, short = ["QP INFEASIBLE: empty input set,", "nominal input applied"], "#b00020", "QP INFEASIBLE (nominal applied)"
    elif dropped:
        cs = ", ".join(f"$c_{d}$" for d in dropped)
        text, colour, short = [f"QP feasible only by dropping {cs}:", f"{cs} not enforced"], "#b35806", "drops " + ", ".join(f"c{d}" for d in dropped)
    else:
        text, colour, short = ["QP feasible, all constraints kept"], "#1a7f37", "QP feasible, all kept"
    if not T["common"]:
        h = T["h"][c][k]; ch = T["chain"][c]
        inside = [i for i in range(p) if h[i] < 0]
        broken = [i for i in range(p) if min(h[i], ch[1][k, i], ch[2][k, i]) < 0]
        if inside:
            text += ["UNSAFE: inside " + ", ".join(rf"$c_{i + 1}$ ($h_{i + 1}={h[i]:.2f}$)" for i in inside)]
            short += ", UNSAFE (inside " + ", ".join(f"c{i + 1}" for i in inside) + ")"
            colour = "#b00020"
        if broken:
            text += ["chain negative for " + ", ".join(f"$c_{i + 1}$" for i in broken) + ": no safety guarantee"]
            if ok:
                text += ["although the QP is feasible"]
            if not inside:
                short += ", chain broken"
                colour = "#b00020" if colour == "#b00020" else "#b35806"
        elif ok and not dropped:
            text += [r"safe: every $\psi_{i,k}>0$"]
    return "\n".join(text), colour, short


# ------------------------------------------------------------------ input-space panels
def window_sizes(T):
    """Half-width of the input-space window, smoothed in time so the zoom changes slowly: for a feasible QP it
    shows the binding constraints and the next nearest one; for an empty fixed-gain QP it shows every
    constraint violated by the applied (nominal) input, i.e. the conflicting ones."""
    n = len(T["t"]); raw = np.empty(n)
    for k in range(n):
        w = []
        for c in T["ctrls"]:
            Am, b, m = T["geom"][k][c]; u = T["u"][c][k]
            g = (Am @ u + b) / np.maximum(np.linalg.norm(Am, axis=1), 1e-9)
            if c == "standard" and not T["ok"][c][k]:
                w.append(np.max(np.abs(g[g < 0]), initial=0.0))
            else:
                d = np.sort(np.abs(g[m])) if m.any() else np.zeros(1)
                w.append(d[min(1, len(d) - 1)])
        raw[k] = np.clip(1.35 * max(w), 8.0, 120.0)
    sm = raw.copy()
    for k in range(1, n):
        sm[k] = max(raw[k], 0.93 * sm[k - 1])
    for k in range(n - 2, -1, -1):
        sm[k] = max(sm[k], 0.93 * sm[k + 1])
    return sm


def draw_input_space(ax, T, k, c, W, ratio=1.0):
    """ratio = height/width of the panel, so the window fills it at equal axis scales."""
    ax.clear()
    Am, b, m = T["geom"][k][c]; u = T["u"][c][k]; ok = T["ok"][c][k]
    gx = np.linspace(u[0] - W, u[0] + W, 240); gy = np.linspace(u[1] - ratio * W, u[1] + ratio * W, int(240 * ratio))
    X, Y = np.meshgrid(gx, gy)
    if m.any():
        adm = np.all(np.stack([Am[i, 0] * X + Am[i, 1] * Y + b[i] for i in np.flatnonzero(m)]) >= 0, axis=0)
        ax.imshow(np.where(adm, 1.0, np.nan), extent=(gx[0], gx[-1], gy[0], gy[-1]), origin="lower",
                  cmap=ListedColormap([CCOL[c]]), vmin=0, vmax=1, alpha=0.20, interpolation="nearest", zorder=1)
    for i in range(len(b)):
        a = Am[i]; nrm = np.linalg.norm(a)
        if nrm < 1e-9:
            continue
        foot = u - (a @ u + b[i]) / nrm ** 2 * a
        tangent = np.array([-a[1], a[0]]) / nrm
        seg = foot[None] + np.array([-4, 4])[:, None] * max(1.0, ratio) * W * tangent[None]
        kept = bool(m[i])
        ax.plot(seg[:, 0], seg[:, 1], color=OCOL[i] if kept else "#9a9a9a", lw=2.2 if kept else 1.4,
                ls="-" if kept else (0, (4, 3)), zorder=3)
        if abs(foot[0] - u[0]) < 0.92 * W and abs(foot[1] - u[1]) < 0.92 * ratio * W:
            n_in = a / nrm
            if kept:
                ax.annotate("", xy=foot + 0.15 * W * n_in, xytext=foot, zorder=4,
                            arrowprops=dict(arrowstyle="-|>", color=OCOL[i], lw=1.8, mutation_scale=14))
            ax.text(*(foot + 0.23 * W * n_in), f"$c_{i + 1}$" + ("" if kept else " dropped"), zorder=5,
                    color=OCOL[i] if kept else "#777", fontsize=10, weight="bold", ha="center", va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))
    if c == "standard" and not ok:
        ax.plot(*u, "X", ms=14, color="#b00020", mec="white", mew=1.2, zorder=6)
    else:
        ax.plot(*u, MARK[c], ms=16 if c == "proposed" else 10, color=CCOL[c], mec="#222", mew=0.8, zorder=6)
    text, colour, _ = status(T, k, c)
    ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=9.5, weight="bold", color=colour,
            zorder=7, bbox=dict(fc="white", ec="#ccc", alpha=0.92, boxstyle="round,pad=0.3"))
    ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[0], gy[-1]); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$j_x$ [m/s$^3$]"); ax.set_ylabel(r"$j_y$ [m/s$^3$]")
    ax.set_title(NAME[c], color=CCOL[c], weight="bold")
    ax.grid(alpha=0.15)


def draw_barrier_bars(ax, T, k, hlim):
    ax.clear()
    p = len(T["centers"]); yy = np.arange(p)
    if T["common"]:
        hk = T["h"]["proposed"][k]
        ax.barh(yy, hk, height=0.55, color=[OCOL[i] for i in range(p)])
        for i in range(p):
            ax.text(hk[i], yy[i], f" {hk[i]:.2f}", va="center", ha="left", fontsize=9)
        ax.set_title(r"barrier values $h_i$ (common state)")
    else:
        n = len(T["ctrls"]); w = 0.8 / n
        for j, c in enumerate(T["ctrls"]):
            hk = T["h"][c][k]
            ax.barh(yy + (j - (n - 1) / 2) * w, hk, height=w, color=[CCOL[c] if v >= 0 else "#7a0015" for v in hk], label=SHORT[c])
        ax.set_title("barrier values $h_i$: " + ", ".join(SHORT[c] for c in T["ctrls"]) + "\n(top to bottom; dark red = inside the obstacle)", fontsize=10)
    ax.axvline(0, color="k", lw=1)
    ax.set_xscale("symlog", linthresh=0.05); ax.set_xlim(*hlim)
    ticks = [v for v in (-1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0) if hlim[0] <= v <= hlim[1]]
    ax.set_xticks(ticks, [f"{v:g}" for v in ticks]); ax.minorticks_off()
    ax.set_yticks(yy, [f"$c_{i + 1}$" for i in range(p)]); ax.set_ylim(p - 0.45, -0.55)
    ax.grid(axis="x", alpha=0.18)


# ------------------------------------------------------------------ main video (both tests)
def overview_video(T):
    t = T["t"]; p = len(T["centers"]); W = window_sizes(T)
    C = T["ctrls"]; n = len(C)
    hall = np.concatenate([T["h"][c].ravel() for c in C])
    hlim = (min(-0.6, 1.4 * hall.min()), 1.6 * hall.max())
    enforced = {c: (T["mask"].sum(1) if c == "lcs" else np.where(T["ok"][c], p, 0)) for c in C}
    path = OUT / f"{T['tag']}.mp4"
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(19.2, 10.8), facecolor="white")
        gs = fig.add_gridspec(3, n + 1, width_ratios=(1.08,) + (1,) * n, height_ratios=(1, 1, 0.78),
                              left=0.04, right=0.99, top=0.875, bottom=0.06, wspace=0.24, hspace=0.36)
        scene = fig.add_subplot(gs[0, 0]); bars = fig.add_subplot(gs[1, 0])
        inputs = {c: fig.add_subplot(gs[0:2, j + 1]) for j, c in enumerate(C)}
        ax_rows = fig.add_subplot(gs[2, 0]); ax_eta = fig.add_subplot(gs[2, 1])
        ax_nu = fig.add_subplot(gs[2, 2]); ax_kap = fig.add_subplot(gs[2, 3:])

        for i, cen in enumerate(T["centers"]):
            scene.add_patch(Circle(cen, T["radius"], fc=OCOL[i], ec=OCOL[i], alpha=0.30, lw=1.5))
            scene.text(*cen, f"$c_{i + 1}$", ha="center", va="center", weight="bold", fontsize=11)
        scene.plot(T["ref"][:, 0], T["ref"][:, 1], "--", color="#999", lw=1.2,
                   label="prescribed path" if T["common"] else "reference")
        alert = scene.text(0.02, 0.98, "", transform=scene.transAxes, ha="left", va="top", fontsize=10, weight="bold",
                           color="#b00020", zorder=8)
        trails, dots = {}, {}
        for c in (("proposed",) if T["common"] else C):
            col = "#222" if T["common"] else CCOL[c]
            trails[c], = scene.plot([], [], color=col, lw=2.0, label="common state" if T["common"] else SHORT[c])
            dots[c], = scene.plot([], [], MARK[c] if not T["common"] else "o", ms=9, color=col, mec="#222")
        scene.set(xlim=T["xlim"], ylim=T["ylim"], xlabel="$x$ [m]", ylabel="$y$ [m]", title="workspace")
        scene.set_aspect("equal", adjustable="box"); scene.grid(alpha=0.15); scene.legend(loc="lower right", ncol=2)

        lines_rows = {c: ax_rows.step([], [], where="post", color=CCOL[c], lw=1.8, label=SHORT[c])[0] for c in C}
        ax_rows.set(xlim=(t[0], t[-1]), ylim=(-0.3, p + 0.4), yticks=range(p + 1), xlabel="time [s]",
                    title="constraints enforced (0 = QP empty)")
        ax_rows.legend(loc="lower left", ncol=2 if n > 3 else 3); ax_rows.grid(alpha=0.15)
        lines_eta = [ax_eta.plot([], [], color=OCOL[i], lw=1.8, label=rf"$\eta_{i + 1}$")[0] for i in range(p)]
        ax_eta.axhline(cq.ETA_MIN, color="#555", ls=":", lw=1.1, label=r"$\underline{\eta}$")
        ax_eta.set(xlim=(t[0], t[-1]), ylim=(0.9 * cq.ETA_MIN, max(1.08 * T["eta"].max(), cq.ETA_MIN + 0.05)),
                   xlabel="time [s]", title=r"proposed: terminal gains $\eta_i\geq\underline{\eta}$")
        ax_eta.legend(loc="upper right", ncol=3); ax_eta.grid(alpha=0.15)
        lines_nu = [[ax_nu.plot([], [], color=OCOL[i], lw=1.5, ls=("-" if l == 0 else (0, (3, 1.6))))[0]
                     for l in range(2)] for i in range(p)]
        lines_nulo = [[ax_nu.plot([], [], color=OCOL[i], lw=1.0, ls=":")[0] for l in range(2)] for i in range(p)]
        ylo, yhi = 2.0 * min(T["nu_lo"].min(), T["nu"].min()), 1.6 * max(T["nu"].max(), 1.0)
        ax_nu.set_yscale("symlog", linthresh=1.0); ax_nu.set_ylim(ylo, yhi)
        nt = [v for v in (-100.0, -10.0, -1.0, 0.0, 1.0, 10.0) if ylo <= v <= yhi]
        ax_nu.set_yticks(nt, [f"{v:g}" for v in nt]); ax_nu.minorticks_off()
        ax_nu.axhline(0, color="#999", lw=0.7)
        ax_nu.set(xlim=(t[0], t[-1]), xlabel="time [s]")
        ax_nu.legend(handles=[Line2D([], [], color="#444", label=r"$\nu_{i,1}$"),
                              Line2D([], [], color="#444", ls=(0, (3, 1.6)), label=r"$\nu_{i,2}$"),
                              Line2D([], [], color="#444", ls=":", label=r"$\underline{\nu}_{i,\ell}$")], loc="center right")
        ax_nu.grid(alpha=0.15)
        lines_kap = [[ax_kap.plot([], [], color=OCOL[i], lw=1.6, ls=("-" if l == 0 else (0, (3, 1.6))))[0]
                      for l in range(2)] for i in range(p)]
        for l, kn in enumerate(cq.K_NOM):
            ax_kap.axhline(kn, color="#aaa", lw=0.8)
        ax_kap.set(xlim=(t[0], t[-1]), ylim=(min(T["kappa"].min(), cq.K_NOM.min()) - 0.1, max(T["kappa"].max(), cq.K_NOM.max()) + 0.1),
                   xlabel="time [s]", title=r"proposed: lower-stage gains $\kappa_{i,\ell}$")
        ax_kap.legend(handles=[Line2D([], [], color="#444", label=r"$\ell=1$"),
                               Line2D([], [], color="#444", ls=(0, (3, 1.6)), label=r"$\ell=2$"),
                               Line2D([], [], color="#aaa", label=r"$\kappa_{\rm nom}$")], loc="upper right", ncol=3)
        ax_kap.grid(alpha=0.15)
        head = fig.suptitle("", fontsize=12.5 if n > 3 else 13, weight="bold", y=0.997, va="top")
        fig.text(0.5, 0.932, FORMULATION, ha="center", fontsize=11, color="#2171b5")
        fig.text(0.5, 0.906, NOTE if not T["common"] else "state prescribed (common to all controllers): the test is "
                 "pointwise feasibility of the three QPs, not closed-loop safety", ha="center", fontsize=11, color="#444")

        fig.canvas.draw()
        ratio = {c: inputs[c].get_position().height * 10.8 / (inputs[c].get_position().width * 19.2) for c in C}
        writer = FFMpegWriter(fps=T["fps"], bitrate=3500, metadata=dict(title=T["title"]))
        frames = list(range(0, len(t), T["stride"]))
        if frames[-1] != len(t) - 1:
            frames.append(len(t) - 1)
        with writer.saving(fig, str(path), dpi=100):
            for k in frames:
                for c in trails:
                    trails[c].set_data(T["path"][c][:k + 1, 0], T["path"][c][:k + 1, 1])
                    dots[c].set_data([T["path"][c][k, 0]], [T["path"][c][k, 1]])
                draw_barrier_bars(bars, T, k, hlim)
                for c in C:
                    draw_input_space(inputs[c], T, k, c, W[k], ratio[c])
                for j, c in enumerate(C):
                    lines_rows[c].set_data(t[:k + 1], enforced[c][:k + 1] + 0.06 - 0.12 * j / max(n - 1, 1))
                for i in range(p):
                    lines_eta[i].set_data(t[:k + 1], T["eta"][:k + 1, i])
                    for l in range(2):
                        lines_kap[i][l].set_data(t[:k + 1], T["kappa"][:k + 1, i, l])
                        lines_nu[i][l].set_data(t[:k + 1], T["nu"][:k + 1, 2 * i + l])
                        lines_nulo[i][l].set_data(t[:k + 1], T["nu_lo"][:k + 1, 2 * i + l])
                ax_nu.set_title(r"proposed: $\nu_{i,\ell}\geq\underline{\nu}_{i,\ell}(\vartheta)$ (dotted), "
                                rf"$\min(\nu-\underline{{\nu}})={np.min(T['nu'][:k + 1] - T['nu_lo'][:k + 1]):.2f}$")
                st = {c: status(T, k, c) for c in C}
                unsafe = [SHORT[c] + " inside " + ", ".join(f"c{i + 1}" for i in range(p) if T["h"][c][k, i] < 0)
                          for c in C if not T["common"] and T["h"][c][k].min() < 0]
                alert.set_text("\n".join(unsafe))
                phase = f" ({T['phase'][k]})" if T["phase"] else ""
                head.set_text(f"{T['title']}   |   t = {t[k]:.2f} s{phase}\n"
                              + "   |   ".join(f"{SHORT[c]}: {st[c][2]}" for c in C))
                head.set_color("#b00020" if any(st[c][1] != "#1a7f37" for c in C) else "#111")
                writer.grab_frame()
        plt.close(fig)
    return path


# ------------------------------------------------------------------ chain video (Test 1-b)
def chain_video(T):
    t = T["t"]; p = len(T["centers"])
    rows = (r"$h_i=\psi_{i,0}$", r"$\psi_{i,1}$", r"$\psi_{i,2}$", r"$\psi_{i,3}$ (terminal)")
    thresh = (0.05, 0.1, 0.25, 0.02)
    path = OUT / "test1b_chain.mp4"
    with plt.rc_context(RC):
        C = T["ctrls"]
        fig, axes = plt.subplots(4, len(C), figsize=(19.2, 10.8), sharex=True, facecolor="white")
        fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.06, hspace=0.28, wspace=0.16)
        lines, titles = {}, {}
        for j, c in enumerate(C):
            for r in range(4):
                ax = axes[r, j]; Y = T["chain"][c][r]
                lo, hi = min(Y.min(), -thresh[r]), max(Y.max(), thresh[r])
                ax.set_yscale("symlog", linthresh=thresh[r], linscale=1.2)
                ylim = (lo * (1.6 if lo < 0 else 0.5), hi * 1.6); ax.set_ylim(*ylim)
                ticks = [v for v in (-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0) if ylim[0] <= v <= ylim[1]]
                ax.set_yticks(ticks, [f"{v:g}" for v in ticks]); ax.minorticks_off()
                ax.set_xlim(t[0], t[-1]); ax.axhline(0, color="k", lw=0.9)
                if r == 3:
                    ax.axhline(cq.EPS, color="#555", ls=":", lw=1.1)
                    ax.set_xlabel("time [s]")
                if j == 0:
                    ax.set_ylabel(rows[r])
                ax.grid(alpha=0.15)
                lines[c, r] = [ax.plot([], [], color=OCOL[i], lw=1.6, label=f"$c_{i + 1}$")[0] for i in range(p)]
            axes[0, j].legend(loc="lower right", ncol=p)
            titles[c] = axes[0, j].set_title(NAME[c], color=CCOL[c], weight="bold")
        head = fig.suptitle("", fontsize=14, weight="bold", y=0.985)
        fig.text(0.5, 0.945, r"HOCBF chain $\psi_{i,k}=\dot\psi_{i,k-1}+\kappa_{i,k}\psi_{i,k-1}$: safety needs every "
                 r"level positive; dotted line: reserve $\varepsilon$", ha="center", fontsize=11)
        fig.text(0.5, 0.92, NOTE, ha="center", fontsize=11, color="#444")
        writer = FFMpegWriter(fps=T["fps"], bitrate=3000, metadata=dict(title="Test 1-b barrier chain"))
        frames = list(range(0, len(t), T["stride"]))
        if frames[-1] != len(t) - 1:
            frames.append(len(t) - 1)
        with writer.saving(fig, str(path), dpi=100):
            for k in frames:
                for c in C:
                    for r in range(4):
                        for i in range(p):
                            lines[c, r][i].set_data(t[:k + 1], T["chain"][c][r][:k + 1, i])
                st = {c: status(T, k, c) for c in C}
                for c in C:
                    titles[c].set_text(NAME[c] + "\n" + st[c][2])
                    titles[c].set_color(st[c][1] if st[c][1] != "#1a7f37" else CCOL[c])
                head.set_text(f"Test 1-b: complete HOCBF chain   |   t = {t[k]:.2f} s ({T['phase'][k]})")
                writer.grab_frame()
        plt.close(fig)
    return path


# ------------------------------------------------------------------ Test 1-c (six variants of Fig. 3)
def test1c_video(stride=4, fps=24):
    import test1c as TC
    keys = TC.PAPER; letters = "abcdef"
    D = {}
    for key in keys:
        sc = TC.SCENARIOS[key]
        D[key] = dict(sc=sc, fixed=TC.simulate_baseline(sc, "fixed"), lcs=TC.simulate_baseline(sc, "lcs"),
                      proposed=TC.simulate_proposed(sc, "both"))
    TC.restore()
    t = D[keys[0]]["proposed"]["t"]
    path = OUT / "test1c.mp4"
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(19.2, 10.8), facecolor="white")
        gs = fig.add_gridspec(3, 3, height_ratios=(1, 1, 0.55), left=0.04, right=0.99, top=0.855, bottom=0.06,
                              wspace=0.12, hspace=0.36)
        axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
        ax_eta = fig.add_subplot(gs[2, :])
        art = {}
        cols6 = ("#1b9e77", "#d95f02", "#e7298a", "#66a61e", "#a6761d", "#4d4d4d")
        for ax, key, L in zip(axes, keys, letters):
            sc = D[key]["sc"]
            for i, cen in enumerate(np.asarray(sc.centers)):
                ax.add_patch(Circle(cen, sc.radius, fc="#e3e3e3", ec="#666", lw=1.0))
                ax.text(*cen, f"$c_{i + 1}$", ha="center", va="center", fontsize=11)
            ref = D[key]["proposed"]["ref"]
            ax.plot(ref[:, 0], ref[:, 1], ":", color="#999", lw=1.2)
            a = {}
            for c, col, ls in (("fixed", CCOL["standard"], "--"), ("lcs", "#7b3294", "-."), ("proposed", CCOL["proposed"], "-")):
                a[c] = (ax.plot([], [], color=col, lw=2.0, ls=ls)[0], ax.plot([], [], "o", color=col, ms=7, mec="#222")[0])
            a["bad"] = ax.plot([], [], "o", color="#7f1d1d", ms=3.5, mec="none", zorder=6)[0]
            a["drop"] = ax.plot([], [], "x", color="#7b3294", ms=5, mew=1.2, zorder=6)[0]
            a["title"] = ax.set_title("", fontsize=11, loc="left")
            ax.set(xlim=(-1.7, 4.8), ylim=(-2.45, 2.45)); ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.12)
            art[key] = a
        eta_lines = {key: ax_eta.plot([], [], color=col, lw=1.8, label=f"({L}) {TC.SCENARIOS[key].label}")[0]
                     for key, L, col in zip(keys, letters, cols6)}
        ax_eta.axhline(cq.ETA_MIN, color="#555", ls=":", lw=1.1)
        ax_eta.set(xlim=(t[0], t[-1]), ylim=(0.34, 0.64), xlabel="time [s]",
                   title=r"proposed controller: largest terminal gain $\max_i\eta_i(t)$ (dotted: $\underline{\eta}$)")
        ax_eta.legend(loc="upper right", ncol=3, fontsize=9); ax_eta.grid(alpha=0.15)
        fig.legend(handles=[Line2D([], [], color=CCOL["standard"], ls="--", lw=2, label="fixed-gain HOCBF-QP"),
                            Line2D([], [], color="#7b3294", ls="-.", lw=2, label="LCS"),
                            Line2D([], [], color=CCOL["proposed"], lw=2, label="proposed"),
                            Line2D([], [], color="#7f1d1d", ls="none", marker="o", label="inside an obstacle"),
                            Line2D([], [], color="#7b3294", ls="none", marker="x", label="LCS constraint dropped"),
                            Line2D([], [], color="#999", ls=":", label="reference")],
                   loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.962), frameon=False, fontsize=11)
        head = fig.suptitle("", fontsize=14, weight="bold", y=0.995)
        writer = FFMpegWriter(fps=fps, bitrate=3500, metadata=dict(title="Test 1-c"))
        frames = list(range(0, len(t), stride))
        if frames[-1] != len(t) - 1:
            frames.append(len(t) - 1)
        with writer.saving(fig, str(path), dpi=100):
            for k in frames:
                for key, L in zip(keys, letters):
                    a = art[key]; status = []
                    for c in ("fixed", "lcs", "proposed"):
                        x = D[key][c]["x"]
                        a[c][0].set_data(x[:k + 1, 0], x[:k + 1, 1]); a[c][1].set_data([x[k, 0]], [x[k, 1]])
                    bx = np.vstack([D[key][c]["x"][:k + 1][D[key][c]["h"][:k + 1].min(1) < 0, :2] for c in ("fixed", "lcs")])
                    a["bad"].set_data(bx[:, 0], bx[:, 1])
                    l = D[key]["lcs"]; dr = (~l["mask"][:k + 1]).any(1)
                    a["drop"].set_data(l["x"][:k + 1][dr, 0], l["x"][:k + 1][dr, 1])
                    for c, n in (("fixed", "fixed"), ("lcs", "LCS")):
                        if D[key][c]["h"][k].min() < 0:
                            status.append(f"{n} inside an obstacle")
                    if not D[key]["fixed"]["ok"][k]:
                        status.append("fixed QP empty")
                    if (~l["mask"][k]).any():
                        status.append("LCS drops a constraint")
                    a["title"].set_text(f"({L}) {D[key]['sc'].label}\n" + ("; ".join(status) if status else " "))
                    a["title"].set_color("#b00020" if status else "#111")
                    e = D[key]["proposed"]["eta"]; eta_lines[key].set_data(t[:k + 1], e[:k + 1].max(1))
                head.set_text(f"Test 1-c: six harder variants of Test 1-b   |   t = {t[k]:.2f} s   |   "
                              "proposed: QP feasible at every sample, all constraints kept")
                writer.grab_frame()
        plt.close(fig)
    return path


def main(only=None):
    """only = None (all videos) or one of "test1a", "test1b", "test1c"."""
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    if only == "test1c":
        print(test1c_video(), round(time.time() - t0, 1), "s", flush=True)
        return
    if only == "test1a":
        print(overview_video(add_rate_bounds(test_1a(), A.DT)), flush=True)
        return
    if only == "test1b":
        T2 = add_rate_bounds(test_1b(), B.DT); print(overview_video(T2), chain_video(T2), flush=True)
        return
    T1 = add_rate_bounds(test_1a(), A.DT); print("Test 1-a simulated", flush=True)
    print(overview_video(T1), round(time.time() - t0, 1), "s", flush=True)
    T2 = add_rate_bounds(test_1b(), B.DT); print("Test 1-b simulated", flush=True)
    print(overview_video(T2), round(time.time() - t0, 1), "s", flush=True)
    print(chain_video(T2), round(time.time() - t0, 1), "s", flush=True)
    print(test1c_video(), round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
