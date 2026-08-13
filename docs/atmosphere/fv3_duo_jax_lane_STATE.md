# FV3 duo-grid JAX lane — campaign STATE / handoff

Companion to `fv3_duo_jax_lane_strategy.md`. That document is the contract and
does not change often; **this file is the live state and is rewritten at the end
of every session.** The pattern is taken from the FESOM2 port (arXiv:2606.11356
§4.3), which found that a port of this length cannot be carried in a model's
own memory: it needs a plan file, a lessons file, and a handoff file kept beside
the code, so a fresh short session can rebuild context from a few reads.

---

## Where the work lives

| | |
|---|---|
| worktree | `/burg-archive/glab/users/pg2328/legoESM_fv3jax` |
| branch | `feat/fv3-duo-jax-lane` (pushed to `cf`) |
| run/log dir | `/burg-archive/glab/users/pg2328/fv3_duo_gaps/` |
| **pinned oracle tree** | `/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean` |
| oracle reference runs (42) | `/burg-archive/glab/users/pg2328/Code/FV3/duogrid_zenodo/extracted/Code and simulations files` |
| oracle 1-step/N-step runs | `/burg-archive/glab/users/pg2328/fv3_oracle_pinned/run_{hydro,nh}_{zerostep,1step,96min,320min}*` |

**Do not cite `fv3_recon/duo_model/`.** Its `dyn_core.F90` is an instrumented
copy (3230 lines, md5 `0a5df09a…`) against the pinned tree's 3128 / `e5a5fab9…`.
`sw_core.F90` IS identical across copies. See the correction banner in
`fv3_duo_gap_register_2026-08-06.md`.

---

## The one-line summary of the gap

The faithful duo port is **certified but not runnable as a model**: ~15k lines
of loop-faithful NumPy fp64, verified against Fortran dumps down to 1.19e-9
(hydrostatic, one step), imported by nothing outside `tests/` and
`scripts/validate/`. This campaign gives it a JAX twin that is jittable,
differentiable, and wired in.

Exception worth knowing: the NumPy lane is **not** entirely leaf-only. Five
init-time imports are load-bearing — `cubed_sphere.py:476`,
`cubed_sphere_cdgrid.py:768/1081`, and `duogrid_bgrid_ring.py:95/149/250` pull
the halo tables, metrics, ext machinery and `build_six_face_duo_context` from
it. Those are setup, not hot path, and they stay NumPy.

---

## Status

### Done this session

- `fv3_duo_jax_lane_strategy.md` — the contract: three-hop authority chain,
  tolerance policy keyed to what a kernel does, six-tier validation ladder,
  differentiability gates, failure-mode catalogue, work breakdown, scope
  boundary.
- **Register correction**: two duo-barrier line numbers were from the
  instrumented tree. Pinned-tree values are `dyn_core.F90:872` (CGRID_NE,
  barrier 1) and `:984` (BGRID_NE, barrier 2). The CODE was already right.
- **Two self-caught strategy defects fixed**: the live-twin switch cannot call
  NumPy from inside `jit` (verify mode is therefore eager); and §2 did not state
  what hop B is blind to.
- **`nsplt` design resolved** — static, with the argument for why that costs
  nothing in the gradient (floor ⇒ derivative zero a.e.).
- Measurement job `scripts/cluster/fv3_native/jax_lane_measure.sbatch`.
- Codex adversarial review of the strategy submitted.

### In flight

Five kernel-port agents writing JAX mirrors (`update_dz_d`; `fv3_pgrad.py`;
`fv3_tp_core.py`; `fv3_mapz.py`; `fv3_duo_halos.py`), each under the rules in
the strategy doc, each leaving `TOL-PENDING` markers for the measurement job to
replace.

### Not started

SW core (`c_sw`, `d_sw1..6_duo`) — blocked on `fv3_tp_core.py` landing so it can
be written against the real API. Then the km=1 stepper, then Phases 2 and 3.

---

## Next task, precisely

1. `sbatch scripts/cluster/fv3_native/jax_lane_measure.sbatch`; read the
   `MEASURED`/assertion values; replace **every** `TOL-PENDING` marker with
   `measured X, bound = measured × N`. Nothing ships with a marker left —
   the job censuses them in step 1 for exactly this reason.
2. Codex adversarial review of the code (not just the strategy), then fix and
   re-review until clean.
3. SW core port, then the km=1 `full_acoustic_step_sixface` JAX twin.
4. `--backend {numpy,jax}` on `run_duo_stepper_w2.py` / `run_duo_stepper_case6.py`
   and score with the **existing** calibrated gates.

---

## The canonical reference runs, and their calibrated bounds

Recorded here so a later session does not re-derive them.

| gate | reference | calibrated bound |
|---|---|---|
| `w2_duo_oracle_gate.py` | `C48.sw.case2.alpha0.duo.hord6`, day 5.0 | envelope max\|v\| 0.0236, rms(v) 0.0096 m/s, scaled `(48/N)²`, `--tol-factor` 1.5 |
| `case6_duo_oracle_gate.py --case 6` | `C48.sw.case6.alpha0.duo.hord8` | `--max-gh 5e-3 --max-wind 6e-2` (worst measured 2.304e-3 / 2.850e-2) |
| `case6_duo_oracle_gate.py --case 2` | `C48.sw.case2.alpha0.duo.hord8` | `--max-gh 1.1e-2 --max-wind 2e-2` |
| `case6_duo_oracle_gate.py --case 2 --ref-alpha 45` | `C48.sw.case2.alpha45.duo.hord8` | `--max-gh 1.1e-2 --max-wind 2.5e-1` (wind bound is pole-artifact-limited) |
| `full_step_oracle_parity.py` | `run_hydro_1step_gfs` | IC control 1e-12 hard; step floor measured 1.1866e-9 |
| `full_step_oracle_parity.py --nh` | `run_nh_1step_gfs` | delz 6e-8, pt 1e-8, delp 5e-8, u/v 3-10e-6, w 3.4e-7 m/s |
| `jw_duo_oracle_compare.py` | `C48.nh.case-13.alpha0.duo.hord6` | `--max-rmse-hpa`; oracle grows 0.0295 → 2.380 hPa over days 1-9 |

`d_ext` **must** be 0: every duo deck resolves `D_EXT = 0.0`. A run with
`d_ext=0.02` is not comparable to any oracle number and has already produced one
retracted claim in this campaign.

---

## Lessons file (append, never prune)

Each entry cost something. New sessions read this before touching the port.

1. **Port the EXECUTION, not the call text.** A Fortran routine whose body is
   guarded on `present(...)` arguments the caller never passes is a silent
   no-op. `fill_corners_agrid_y(f0)` in the port vs the oracle's inert
   `fill_corners(f0, npx, npy, YDir)` was the entire 9.7882e-6 → 1.1866e-9
   hydrostatic boundary floor. One deletion, four orders of magnitude.
2. **Read what the deck RESOLVES.** The pinned deck runs `adiabatic = .true.`,
   `nwat = 0`, so `zvir` is identically zero and the whole moist / `q_con` /
   `moist_cv` path is structurally dead — not untested, never executed.
   Certifying it needs a new Fortran deck.
3. **Three copies of the oracle source exist and are not the same file.** Cite
   the pinned tree. This one bit the gap register itself.
4. **A tolerance inherited from another gate is a guess.** `1e-15` carried over
   to the kernel-4/5 gates when the measurements were 1.03e-15 and 1.02e-15 —
   a bound with no margin, which codex caught.
5. **Evidence in a comment does not fail when the code regresses.** Both hops
   (JAX-vs-authority and jit-vs-eager) must be ASSERTIONS. Also caught by codex,
   on this same file.
6. **`fv3_recon/duo_model` line numbers are off by roughly +60 in dyn_core.**
   If a citation looks close but not right, this is why.
7. **SLURM sets `TMPDIR=/local` and it gets reaped mid-job.** Force
   `pytest --basetemp` onto a burg path.
8. **Codex jobs serialise.** Two concurrent runs refreshing `CODEX_HOME` killed
   the token once. Use a node-local `CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID`.
9. **Login-node policy is hard.** Nothing over a few seconds or one core,
   including codex. Everything goes through `sbatch`.
