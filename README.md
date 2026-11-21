# Adaptive High-Order Control Barrier Functions for Feasible Safety-Critical Control — simulation code

S. Midda, C. Jada, A. Tiwari, R. Pasumarthy (IIT Madras).

This repository reproduces every simulation, number, table and figure of Section IV of the paper:
Test 1-a (pointwise compatibility along a prescribed path), Test 1-b (closed loop through a conflicted
region) and Test 1-c (six harder variants of Test 1-b). Each test compares three safety filters: the
fixed-gain HOCBF-QP, the Local Configuration Search (LCS) controller, and the proposed adaptive HOCBF-QP.

## Quick start

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python run_all.py            # Tests 1-a, 1-b, 1-c and Figs. 1-3, under a minute
venv/bin/python run_all.py --videos   # also the simulation videos (needs ffmpeg), about 5 min more
```

Each part can also be run on its own:

```bash
venv/bin/python test1a/test1a.py
venv/bin/python test1b/test1b.py
venv/bin/python test1c/test1c.py      # options: --all, --ablation, --sweep, --nu-bar
venv/bin/python figures/make_figures.py
venv/bin/python videos/make_videos.py # or: test1a | test1b | test1c
```

Python 3.10 or later. Tested with the package versions in `requirements.txt`; ffmpeg is needed only for the videos.

## Repository layout

| Path | Content |
|---|---|
| `common/current_qp.py` | the proposed augmented HOCBF-QP, its rate bounds, nominal rates and gain update (shared by all tests) |
| `test1a/test1a.py` | Test 1-a: prescribed state, the three controllers, the Farkas certificates at events A and B |
| `test1b/test1b.py` | Test 1-b: plant, reference, nominal law, the three controllers in closed loop |
| `test1c/test1c.py` | Test 1-c: the harder variants, built on the Test 1-b model |
| `run_all.py` | runs everything; also computes Table II and the inter-sample check of Test 1-b |
| `figures/make_figures.py` | Figs. 1–3 of the paper (`fig1_test1a.pdf`, `fig2_test1b.pdf`, `fig3_test1c.pdf`) |
| `figures/test1b_data.py` | reader for the Test 1-b signals and the plotting constants |
| `videos/make_videos.py` | the simulation videos (`test1a.mp4`, `test1b.mp4`, `test1b_chain.mp4`, `test1c.mp4`) |
| `results/` | written by the scripts: summaries (JSON), signals (CSV, NPZ) |

## Table I: parameters of all tests

| Item | Value |
|---|---|
| weights, reserve | R = I₂, ρ = 10⁻², S = 12 I, T = 8 I; ε = 0.02, η̲ᵢ = 0.35; no upper bound on η |
| gain positivity, rate bounds | κ_min = 0.3, (c_{i,1,1}, c_{i,1,2}) = (4, 5), c_{i,2,1} = 4; ν̄ = 10, never active (max ν = 2.09 in Test 1-a, 3.76 in all tests) |
| gains | κ_nom = (2.3, 2.0), λ_κ = 1; initial κ_{i,1} = 2.3, κ̇_{i,1} = 0, κ_{i,2} = 2.0 |
| baselines | fixed-gain, LCS: (κ₁, κ₂, η) = (2.3, 2.0, 0.35); LCS at depth 3; nominal input if a QP is infeasible |
| initial p, v, a | (−0.194, 0.276), (0.942, −1.878), (−0.566, 3.488) |
| reference, j_ref | 1-a: quintic path, j_ref = 0; 1-b: to (2, 0) m at 2.19 s, then held, j_ref = j_r + 12(p_r − p) + 18(v_r − v) + 8(a_r − a) |
| sampling, solvers | Δt = 0.01 s, zero-order hold of j and ν, exact integration; quadprog (10⁻¹⁰ I regularization), HiGHS (LPs) |

## Implementation

### Plant and safety constraints

The agent is a planar triple integrator, ṗ = v, v̇ = a, ȧ = j, with state x = (p, v, a) ∈ R⁶ and the jerk
j ∈ R² as the unbounded physical input. Obstacle i has center cᵢ and clearance R = 0.58 m, and its barrier is

    h_i(x) = ||p - c_i||^2 - R^2,

which has relative degree rᵢ = 3, with input row A_{u,i} = 2(p − cᵢ)ᵀ. With p obstacles and m = 2 inputs,
p > m in every test, so A_u has a null space and the constraints can be incompatible even though the input
is unbounded.

### HOCBF chain and the fixed-gain HOCBF-QP

The HOCBF chain of each obstacle uses linear class-K gains:

    psi_{i,0} = h_i,   psi_{i,1} = dh_i + k_{i,1} h_i,   psi_{i,2} = dpsi_{i,1} + k_{i,2} psi_{i,1},
    psi_{i,3} = dpsi_{i,2} + eta_i psi_{i,2} >= eps.

The fixed-gain HOCBF-QP keeps (κ₁, κ₂, η) = (2.3, 2.0, 0.35) and solves

    min_j  1/2 ||j - j_ref||^2   s.t.   A_u j + beta >= 0,

with one row per obstacle (`standard_control` in `test1b.py`, `fixed_control` in `test1a.py`). When the
QP is infeasible, the nominal input j_ref is applied.

### Proposed adaptive HOCBF-QP (`common/current_qp.py`)

The lower-stage gains become states driven by differential inputs ν through matched-order integrator chains,
κ̈_{i,1} = ν_{i,1} and κ̇_{i,2} = ν_{i,2}, and each obstacle gets a free terminal gain ηᵢ. The terminal row of
obstacle i is affine in all decision variables w = (j, ν, η):

    psi_{i,3} = A_{u,i} j + h_i nu_{i,1} + psi_{i,1} nu_{i,2} + b_i + eta_i psi_{i,2} >= eps.

At every sample the controller solves the QP

    min  1/2 ||j - j_ref||^2_R + 1/(2 rho) ||nu - nu_nom||^2_S + 1/(2 rho^2) ||eta - eta_lo||^2_T
    s.t. A_u j + A_nu nu + D_psi eta + b >= eps 1,
         nu_lo(theta) <= nu <= nu_bar,   eta >= eta_lo,

with diagonal weights R = I, S = 12 I, T = 8 I and ρ = 0.01. The effective weights are 1, 1.2·10³ and 8·10⁴,
so the QP corrects the physical input first, then the gain rates, and only then the terminal gains. The parts are:

- **Lower rate bound ν̲(θ)** (`rate_box`). This is an HOCBF on each gain chain that keeps every lower-stage gain
  above κ_min = 0.3, with the Hurwitz polynomials (s + 4)(s + 5) and s + 4:
  ν_{i,1} ≥ −9 κ̇_{i,1} + 20 (κ_min − κ_{i,1}) and ν_{i,2} ≥ 4 (κ_min − κ_{i,2}).
- **Cap ν̄ = 10** (`NU_BAR`). This is a constant upper bound that is never active in the reported tests.
- **Nominal rate ν_nom.** This input returns the gains to κ_nom = (2.3, 2.0) with rate λ_κ = 1:
  ν_nom,{i,1} = −2 κ̇_{i,1} − (κ_{i,1} − 2.3) and ν_nom,{i,2} = −(κ_{i,2} − 2.0), clipped to [ν̲, ν̄].
- **Solver.** quadprog (Goldfarb–Idnani) with a 10⁻¹⁰ I regularization. Because the terminal coefficient
  ψ_{i,2} > 0 gives each row a private direction ηᵢ, the QP is feasible whenever the chain is positive
  (Proposition 1 of the paper). The ablation modes `eta_only` (ν = ν_nom) and `nu_only` (η = η̲) are
  available through the `mode` argument of `solve`.

### LCS baseline

LCS is the Local Configuration Search of Rousseas and Panagou, with every obstacle constraint declared soft
(`lcs_control` and `selection_step` in `test1b.py`, `lcs_step` in `test1a.py`). At each sample it tests
the feasibility of the selected fixed-gain constraints with the null-space linear program of the Farkas
alternative (HiGHS). If the set is infeasible, it drops constraints, at most 3 per step (depth 3, with an
exact fallback). If the set is feasible, it tries to restore dropped constraints. The fixed-gain QP is then
solved on the retained constraints.

### Sampling, integration and the inter-sample check

All QPs are solved at t_k = kΔt with Δt = 0.01 s. The jerk j and the differential inputs ν are held constant
on [t_k, t_{k+1}) (zero-order hold), and η enters only the constraint at t_k. Under the hold, the triple
integrator and the gain chains are polynomial in s = t − t_k, so they are propagated with their exact
closed-form solutions (`plant_step` in `test1b.py`, `gain_step` in `current_qp.py`); no numerical ODE solver
is used.

    p(t_k+s) = p_k + v_k s + a_k s^2/2 + j_k s^3/6,   kappa_1(t_k+s) = kappa_1 + dkappa_1 s + nu_1 s^2/2,
    kappa_2(t_k+s) = kappa_2 + nu_2 s.

`run_all.py` also evaluates hᵢ and the proposed chain on a 100-point grid inside every hold interval of
Test 1-b (`results/test1b/intersample.json`). Every hᵢ stays within 10⁻⁵ of its sampled minimum, and the
proposed chain stays positive.

### Test 1-a: pointwise compatibility (`test1a/test1a.py`)

- **Setup.** Three obstacles at c₁ = (0, −1.7), c₂ = (0, 1.7) and c₃ = (4, 1.7) m.
- **State.** The state is prescribed, not integrated: one C² quintic segment from the common initial state
  to the event-B state at t_B = 2.27 s, shaped so that the path crosses the c₁–c₂ centerline at event A,
  t_A = 0.23 s. All three controllers are evaluated at the same states, 278 samples over [0, 2.77] s,
  with j_ref = 0. Only the gain chains of the proposed controller are propagated.
- **Outputs.** `results/test1a/` holds the summary, with the Farkas certificates λ_A and λ_B, and the
  per-sample signals.

### Test 1-b: closed loop (`test1b/test1b.py`)

- **Setup.** A fourth obstacle at c₄ = (0.90, 0.30) m sits on the reference.
- **Reference.** A quintic from (0, 0) to the goal (2, 0) m, reached at 2.19 s on the rebased time axis,
  then held.
- **Nominal law.** j_ref = j_r + 12(p_r − p) + 18(v_r − v) + 8(a_r − a).
- **Runs.** The three controllers run as independent closed loops from one initial state at which every
  chain is nonnegative, for 1020 samples (10.19 s).
- **Outputs.** `results/test1b/` holds the signals, a summary, `table_ii.json` (Table II) and
  `intersample.json`.

### Test 1-c: harder variants (`test1c/test1c.py`)

Each variant keeps the Test 1-b model, sampling and controllers, and changes one or more features.
The six variants of Fig. 3 (default run) are:

| Panel | Variant | Change |
|---|---|---|
| (a) | S5 | reference traversed 2 times faster |
| (b) | S6 | poles of the nominal tracking law 1.5 times faster |
| (c) | S13 | as (b), with a fifth obstacle c₅ at (−1, −0.5) m |
| (d) | S9 | passage between c₁ and c₄ narrowed by 0.45 m |
| (e) | S10 | all radii 0.68 m |
| (f) | S11 | fifth obstacle c₅ at (1, −1.45) m |

In S8–S10, S12 and S16 the nominal gains give ψ_{i,1}(0) < 0 for one or two obstacles. The proposed controller
then starts with a raised κ_{i,1}(0) (and κ̇_{i,1}(0) if needed), so that ψ_{i,1}(0) = 0.01 and ψ_{i,2}(0) > 0,
which Proposition 1 allows. The fixed-gain and LCS controllers keep the nominal gains.

Other options:

- `--all` runs all sixteen variants, including those of the response to the reviewers.
- `--ablation` adds the η-only and ν-only versions of the proposed QP.
- `--sweep` runs the reference-speed sweep from 1.0 to 2.5 times.
- `--nu-bar` sets the cap. The response to the reviewers used ν̄ = 600 to check condition (b) of
  Proposition 1. The cap is never active, so the trajectories are identical to those with ν̄ = 10.

`summary.json` also reports the constants of condition (a) of Proposition 1 along each run, the room left
in condition (b), and the resulting terminal-gain bound η̂.

## Outputs and the paper

| Paper | File |
|---|---|
| Fig. 1 (Test 1-a, with event-B input space) | `figures/fig1_test1a.pdf` |
| Fig. 2 (Test 1-b) | `figures/fig2_test1b.pdf` |
| Fig. 3 (Test 1-c) | `figures/fig3_test1c.pdf` |
| Table II | `results/test1b/table_ii.json` (solve times are machine dependent) |
| Test 1-a numbers (24 empty fixed-gain samples, LCS drops, η ≤ 0.413, min hᵢ) | `results/test1a/metrics.json`, `v2b2_summary.json` |
| Test 1-b numbers (52 empty samples, 293 inside c₄, min hᵢ = 0.580, η ≤ 0.356) | `results/test1b/closed_loop_summary.json`, `table_ii.json` |
| Inter-sample check | `results/test1b/intersample.json` |
| Test 1-c numbers | `results/test1c/summary.json` |

## Videos

| File | Content |
|---|---|
| `videos/test1a.mp4` | Test 1-a: workspace, barrier values, the input space of each controller with its constraints and applied input, constraints enforced, and the adapted gains η, ν, κ |
| `videos/test1b.mp4` | Test 1-b: the same layout in closed loop, with QP feasibility and safety reported separately |
| `videos/test1b_chain.mp4` | Test 1-b: the complete HOCBF chain h, ψ₁, ψ₂, ψ₃ of each controller |
| `videos/test1c.mp4` | Test 1-c: the six variants of Fig. 3, with the paths of the three controllers and the largest terminal gain of the proposed controller |

## Citation

If you use this code, please cite the paper (citation to be added on publication).
