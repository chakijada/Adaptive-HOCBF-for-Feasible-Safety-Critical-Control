"""Test 1-b signals and plotting constants shared by make_figures.py and make_videos.py
(reads results/test1b/closed_loop_signals.csv written by test1b/test1b.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = pd.read_csv(ROOT / "results" / "test1b" / "closed_loop_signals.csv")
T = DATA.t.to_numpy()
KEEP = DATA[[f"lcs_keep_{i}" for i in range(1, 5)]].to_numpy() > 0.5     # constraints kept by LCS

STD, LCS, PROP = "#ef3b2c", "#e08214", "#2171b5"
OBS = ["#1f78b4", "#33a02c", "#6a3d9a", "#17becf"]
CONST, REF = "#111111", "#9a9a9a"
EPS, RADIUS = 0.02, 0.58
K1_FIXED, K2_FIXED, ETA_FIXED = 2.3, 2.0, 0.35
CENTERS = np.array([[0.0, -1.7], [0.0, 1.7], [4.0, 1.7], [0.90, 0.30]])
MARKERS = ("o", "s", "^", "D")
LW = 1.15
# (prefix in the CSV, colour, width, style, label); LCS is drawn last with a long dash so that the proposed
# line stays visible underneath where the two coincide
SERIES = (("standard", STD, LW, (0, (2.2, 1.4)), "fixed-gain"),
          ("proposed", PROP, LW, "-", "proposed"),
          ("lcs", LCS, LW, (0, (3.6, 1.5)), "LCS"))
LEGEND_SERIES = (SERIES[0], SERIES[2], SERIES[1])


def runs(mask):
    idx = np.flatnonzero(np.asarray(mask, bool))
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[cuts + 1]], np.r_[idx[cuts], idx[-1]]))


def chain(prefix, level):
    """min_i psi_{i,level} of a controller (level 0 = h)."""
    cols = ([f"{prefix}_h_{i}" for i in range(1, 5)] if level == 0
            else [f"{prefix}_phi{level}_{i}" for i in range(1, 5)])
    return DATA[cols].min(axis=1).to_numpy()
