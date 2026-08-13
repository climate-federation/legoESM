"""#1492 item 1.1: multi-step trajectory replay against NEMO's own per-step
restarts (``RUN_TWIN_STEP1``, ``nn_stock=1``).

Single-step operator comparisons (the 18-row gate) verify a tendency in
isolation, at one instant, on one state. They are structurally BLIND to a
whole class of defect: state carried BETWEEN steps rather than recomputed
within one. The #1492 finding that motivated this lane was exactly that
class -- a correctly-computed surface increment was written outside the
leap-frog RHS, where the MLF combine (state_expl.T - state.T) and the
Asselin now-weight cancel most of it (retention (1-2*gamma)/(2*(1-gamma)) =
4/9 at gamma=0.1, i.e. 56% discarded) -- invisible to any test that checks
the tendency alone, because the tendency WAS correct; only its PLACEMENT in
the time-stepping was wrong. FESOM2-JAX (arXiv:2608.01546 section 2.4) calls
this state threading and tests it with multi-step replay against the
reference model's own restarts. This module is legoESM's equivalent.

Method
------
Bridge legoESM directly from NEMO's y20-continuation restart at step 230400
(``RUN_TWIN_STEP1``'s day-0 state -- the SAME state the #1492 climate result
twinned from, see ``docs/ocean/fidelity/dino_1226_state.md`` "#1455 A5"),
including the leap-frog BEFORE level (``--bridge-before`` equivalent) and the
TKE closure carry (``en``), so the twin's entry state is a genuine leap-frog
continuation, not a forward-Euler cold start. Integrate legoESM forward
step-by-step and, at EACH step k = 1..32 (RUN_TWIN_STEP1 covers exactly one
day, 32 steps at rn_Dt=2700s), compare against NEMO's OWN restart at the same
step. A join defect shows up as a difference that GROWS with k while the
day-0 gate and the single-step (#1226 gate) numbers sit at the bar; a clean
join shows a curve that is flat or decays (dominated by the well-documented
integrator-memory handshake noise, largest at k=1).

``RUN_TWIN_STEP1`` restarts are per-rank NetCDF tiles (16 ranks, no rebuilt
single file exists for this run) -- stitched via the SAME
``rebuild_nemo_restart.rebuild`` in-memory stitcher ``deep_box_heat_budget.py``
already uses for ``RUN_90D_TWIN``'s per-rank trend dumps; no new stitching
logic. The ``nn_hls=0`` axis reshape those tiles need
(``(z,y,x) -> (y,x,z)``, halo-free) is the SAME one-line no-op branch
``nemo_io._to_latlon_lev``/``_strip_halo_2d`` take at ``nn_hls=0`` -- inlined
here (not imported) only because those are private module symbols and the
no-private-cross-import rule forbids importing them; the transform itself is
not re-derived, it is the documented halo=0 identity case.

Paths covered (#1492 checklist):
  1. MLF leapfrog before/now/after level threading (T/S/u/v/eta, all levels)
  2. Asselin/ATF-filtered T_before carried into the next step
  3. Prognostic TKE (en) carry
  4. z* layer thickness (e3t) / ssh commit ordering relative to the tracer
     update (checked via ``compute_layer_thickness`` on the carried eta
     levels -- the same public function ``model.step`` itself calls, not a
     re-derivation)
  5. Solver warm-start state: CHECKED, does not exist on this path.
     ``nemo_dino_kamm_mlf`` sets ``barotropic_solver="explicit_substep"``
     (NEMO's split-explicit ``dynspg_ts``, matching the card's own barotropic
     scheme) -- confirmed by reading ``dino_config_for_recipe`` directly. The
     ONLY legoESM barotropic solver with elliptic/warm-start carry state
     (streamfunction psi/dpsi/dpsin_prev) is ``barotropic_solver="rigid_lid"``,
     seeded once via ``model.seed_scan_carry`` in ``run_dino.py`` (lines
     709-721) -- a DIFFERENT card, not this one. Nothing to replay here; this
     is a documented finding, not a gap.

Sensitivity demonstration
-------------------------
``DINOConfig.surface_tendency_placement`` provides a ready-made positive
control: "applied_now" (legacy, loses 56% of the retained surface tendency
via the Asselin cancellation above) vs "leapfrog_rhs" (faithful, full
retention). Run BOTH from the identical bridged IC and show the trajectories
separate -- proof the multi-step comparison is sensitive to a real threading
defect, not merely reporting noise.

Precision / fidelity conventions (mandatory, not optional)
------------------------------------------------------------
fp64 explicit (``PrecisionPolicy.fp64()`` set before any bridge call, printed
dtypes), ``LEGOESM_NEMO_E3T=both`` (NEMO's true 3-D e3t/gdept ladder, per
``ocean.fidelity.precision_gate.require_explicit_e3t_mode`` -- #1226's
12.9% analytic-vs-true-ladder trap), registry-checked time levels wherever a
dump-file convention applies. NEMO restart fields (tn/sn/.../tb/sb/.../en)
are NOT ambiguous dump files -- their now/before-level convention is the
restart file format itself (documented on ``NemoState``/``NemoBeforeState``),
so ``ocean.fidelity.time_levels`` (built for ambiguous ``.bin`` dumps whose
level requires reading NEMO source) does not apply to them; using it here
would be a category error, not a missed check.

CI status
---------
``.github/workflows/ci.yml`` has not run since 2026-05-27 and its recent runs
failed -- these tests are NOT CI-enforced. They are gated
``pytest.mark.skipif`` on the machine-local oracle tree
(``RUN_TRAJ``/``RUN_TWIN_STEP1``, overridable via ``$DINO_NEMO_RUN_TRAJ`` /
``$DINO_NEMO_RUN_TWIN_STEP1``, see path constants below) and run locally via:

    JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \\
        .venv/bin/python -m pytest tests/ocean/unit/test_dino_1492_multistep_replay.py -v

Runtime: ~2m50s measured for the default (16-step, half-day) suite on CPU
(bridge + 16 steps x 2 placement arms x per-step restart rebuild I/O). Set
``DINO_1492_REPLAY_STEPS`` (default 16, max 32 = the full RUN_TWIN_STEP1 day)
to widen or narrow the window.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np

RUN_TRAJ = os.environ.get(
    "DINO_NEMO_RUN_TRAJ",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ",
)
RUN_TWIN_STEP1 = os.environ.get(
    "DINO_NEMO_RUN_TWIN_STEP1",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TWIN_STEP1",
)
IC_STEP = 230400  # y20 continuation day-0 (DINO_00230400_restart_*.nc per-rank tiles)
STEPS_PER_DAY = 32  # RUN_TWIN_STEP1 covers exactly nit000+1 .. nit000+32 (one day)

RECIPE = "nemo_dino_kamm_mlf"


def have_step1_artifacts() -> bool:
    """True iff both the RUN_TRAJ mesh donor and RUN_TWIN_STEP1 per-rank
    restart tiles for the IC step are present on this machine."""
    import glob
    mesh_ok = os.path.isfile(f"{RUN_TRAJ}/mesh_mask.nc")
    step0_pattern = f"{RUN_TWIN_STEP1}/DINO_{IC_STEP:08d}_restart_*.nc"
    return mesh_ok and len(glob.glob(step0_pattern)) > 0


def _rebuild_step(kt: int, fields: list[str], *, run_twin_step1: str = RUN_TWIN_STEP1
                   ) -> dict[str, np.ndarray]:
    """One NEMO restart step's fields, stitched from the 16 per-rank tiles.

    Reuses ``rebuild_nemo_restart.rebuild`` (the SAME in-memory stitcher
    ``deep_box_heat_budget.py`` uses for RUN_90D_TWIN) -- no new stitching
    logic. Axis reshape ``(z,y,x) -> (y,x,z)`` (3-D) / no-op (2-D) is the
    halo=0 identity branch of ``nemo_io._to_latlon_lev``/``_strip_halo_2d``
    (private symbols, so inlined rather than imported -- see module
    docstring), NOT a re-derivation of the general halo-stripping logic.
    """
    from rebuild_nemo_restart import rebuild

    pattern = f"{run_twin_step1}/DINO_{kt:08d}_restart_*.nc"
    raw = rebuild(pattern, fields)
    out: dict[str, np.ndarray] = {}
    for name, arr in raw.items():
        arr = np.asarray(arr, dtype=np.float64)
        out[name] = np.moveaxis(arr, 0, -1) if arr.ndim == 3 else arr
    return out


def nemo_now_state_at(kt: int, *, run_twin_step1: str = RUN_TWIN_STEP1):
    """NEMO's now-level (Nnn: tn/sn/un/vn/sshn) state at step ``kt``."""
    from legoesm.ocean.fidelity.nemo_io import NemoState

    d = _rebuild_step(kt, ["tn", "sn", "un", "vn", "sshn"], run_twin_step1=run_twin_step1)
    return NemoState(T=d["tn"], S=d["sn"], u=d["un"], v=d["vn"], ssh=d["sshn"], rhd=None)


def nemo_before_state_at(kt: int, *, run_twin_step1: str = RUN_TWIN_STEP1):
    """NEMO's leap-frog before-level (Nbb: tb/sb/ub/vb/sshb/utau_b/vtau_b) at ``kt``."""
    from legoesm.ocean.fidelity.nemo_io import NemoBeforeState

    d = _rebuild_step(kt, ["tb", "sb", "ub", "vb", "sshb", "utau_b", "vtau_b"],
                       run_twin_step1=run_twin_step1)
    return NemoBeforeState(T=d["tb"], S=d["sb"], u=d["ub"], v=d["vb"], ssh=d["sshb"],
                            tau_x=d["utau_b"], tau_y=d["vtau_b"])


def nemo_en_at(kt: int, *, run_twin_step1: str = RUN_TWIN_STEP1) -> np.ndarray:
    """NEMO's TKE restart field (en) at ``kt``, (n_lat, n_lon, jpk)."""
    d = _rebuild_step(kt, ["en"], run_twin_step1=run_twin_step1)
    return d["en"]


def build_replay_ic(*, recipe: str = RECIPE, run_traj: str = RUN_TRAJ,
                     run_twin_step1: str = RUN_TWIN_STEP1,
                     surface_tendency_placement: str | None = None):
    """Bridge legoESM's day-0 (kt=230400) leap-frog-continuation state.

    Returns ``(g, br, cfg, st)`` -- ``g`` the raw ``NemoGrid`` (needed by
    callers for the 3-D tmask), ``br`` the bridge output (geometry/z_coord),
    ``cfg`` the (optionally placement-overridden) ``DINOConfig``, ``st`` the
    fully-bridged state (now + before + TKE levels), day-0-gated.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.experiments.dino import dino_config_for_recipe
    from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        bridge_before_state_topo,
        bridge_nemo_to_legoesm_topo,
    )
    from legoesm.ocean.fidelity.precision_gate import require_explicit_e3t_mode

    set_policy(PrecisionPolicy.fp64())
    require_explicit_e3t_mode(context="dino_1492 multistep replay")

    import sys
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _scripts_dir = os.path.dirname(_this_dir)
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from kamm_twin_90d import bridge_tke_from_restart, verify_day0_matches_restart

    g = read_nemo_mesh_mask(f"{run_traj}/mesh_mask.nc", nn_hls=0)
    s0 = nemo_now_state_at(IC_STEP, run_twin_step1=run_twin_step1)
    br = bridge_nemo_to_legoesm_topo(g, s0, periodic_i=True, full_step=True)
    cfg = dataclasses.replace(dino_config_for_recipe(recipe),
                               lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    if surface_tendency_placement is not None:
        cfg = dataclasses.replace(cfg, surface_tendency_placement=surface_tendency_placement)

    st = br.state
    verify_day0_matches_restart(st, s0, br.land_mask)

    before0 = nemo_before_state_at(IC_STEP, run_twin_step1=run_twin_step1)
    st = bridge_before_state_topo(br._replace(state=st), g, before0, periodic_i=True)

    en0 = nemo_en_at(IC_STEP, run_twin_step1=run_twin_step1)
    st = bridge_tke_from_restart(st, en0, br.land_mask)

    return g, br, cfg, st


def run_replay(n_steps: int, *, surface_tendency_placement: str | None = None,
               dt: float = 2700.0, run_traj: str = RUN_TRAJ,
               run_twin_step1: str = RUN_TWIN_STEP1):
    """Integrate ``n_steps`` from the bridged day-0 state; return per-step
    diagnostics for the 4 threading paths this lane covers.

    Returns a dict of per-step (length ``n_steps``) arrays:
      ``max_dT_now``/``max_dS_now``/``max_du_now``/``max_dv_now``/``max_deta_now``
          -- max|lego - NEMO| on the NOW level, wet cells (path 1)
      ``max_dT_before``/``max_dS_before``/``max_du_before``/``max_dv_before``
          -- max|lego - NEMO| on the leap-frog BEFORE level carried into the
          NEXT step's entry (path 1 + 2: the Asselin-filtered before-state)
      ``max_den`` -- max|lego.tke - NEMO.en| on the TKE carry (path 3)
      ``mean_T_now`` -- domain-mean wet-cell T (retention/sensitivity metric)
      ``max_de3t`` -- max|e3t(lego eta_now) - e3t(NEMO sshn)| per step, BOTH
          sides run through the SAME ``compute_layer_thickness`` on the SAME
          ``(H_bathy, z_coord)`` -- only the eta input differs (lego's own
          carried ``state.eta`` vs NEMO's restart ``sshn`` at the identical
          step). This is path 4: it isolates whether the ssh/e3t commit
          legoESM's tracer update actually weights against tracks NEMO's own
          ssh commit at the SAME step, rather than lagging or leading it by
          one time level (an ordering bug would show as a step-dependent,
          not merely magnitude-dependent, growth here -- see the module
          docstring's growth-vs-flat criterion).
      ``kt`` -- the NEMO step index compared at each entry
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing,
        dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    g, br, cfg, st = build_replay_ic(
        surface_tendency_placement=surface_tendency_placement,
        run_traj=run_traj, run_twin_step1=run_twin_step1)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)

    # Opt-in outer_integrator override for the nemo_mlf step-transcription A/B
    # (rung (b) of nemo_mlf_step_transcription_spec.md). Same pattern as
    # dino_year_screen_fullframe.py / kamm_twin_90d.py -- unset/"leapfrog"
    # leaves the recipe default untouched. nemo_mlf HARD-REQUIRES the NEMO
    # e3w(Kmm) divisor at construction, so force it here too.
    _oi = os.environ.get("DINO_OUTER_INTEGRATOR", "")
    if _oi:
        if _oi not in ("leapfrog", "nemo_mlf"):
            raise SystemExit(
                f"Unknown DINO_OUTER_INTEGRATOR={_oi!r}: expected "
                "'leapfrog' or 'nemo_mlf'")
        mc = mc._replace(
            outer_integrator=_oi,
            implicit_vmix_e3t_now_divisor=(
                True if _oi == "nemo_mlf" else mc.implicit_vmix_e3t_now_divisor))
        print(f"ABLATION: outer_integrator={mc.outer_integrator} "
              f"implicit_vmix_e3t_now_divisor={mc.implicit_vmix_e3t_now_divisor}")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)

    tmask3 = np.asarray(g.tmask) > 0.5
    umask3 = np.asarray(g.umask) > 0.5
    vmask3 = np.asarray(g.vmask) > 0.5
    wet2 = np.asarray(br.land_mask.data) > 0.5
    h_bathy = st.H_bathy.data

    print(f"REALIZED DTYPES: T={st.T.data.dtype} tke={st.tke.data.dtype} "
          f"eta={st.eta.data.dtype} u={st.u.data.dtype} H_bathy={h_bathy.dtype}")

    out = {k: [] for k in (
        "max_dT_now", "max_dS_now", "max_du_now", "max_dv_now", "max_deta_now",
        "max_dT_before", "max_dS_before", "max_du_before", "max_dv_before",
        "max_den", "mean_T_now", "max_de3t", "kt",
    )}

    placement = getattr(cfg, "surface_tendency_placement", "applied_now")
    for k in range(1, n_steps + 1):
        kt = IC_STEP + k
        ext_rate = None
        if placement == "leapfrog_rhs":
            st, ext_rate = apply_dino_lat_lon_surface_forcing(
                st, forcing, br.z_coord, cfg, dt, t_seconds=k * dt, return_rate=True)
        else:
            st = apply_dino_lat_lon_surface_forcing(
                st, forcing, br.z_coord, cfg, dt, t_seconds=k * dt)
        st = model.step(st, dt, surface_forcing=None, external_tracer_rate=ext_rate)

        # --- path 1: now-level state vs NEMO's restart at the SAME step ---
        ns = nemo_now_state_at(kt, run_twin_step1=run_twin_step1)

        # --- path 4: e3t/ssh commit ordering, cross-model -----------------
        # SAME (H_bathy, z_coord, min_water_column_m) on both sides -- only
        # the eta input differs (lego's own state.eta vs NEMO's sshn at the
        # IDENTICAL step kt). A stale/off-by-one eta commit would show as a
        # de3t curve that grows with k even though max_deta_now (the raw ssh
        # difference feeding it) stays flat/bounded.
        e3t_lego = compute_layer_thickness(
            st.eta.data, h_bathy, br.z_coord, min_water_column_m=mc.min_water_column_m)
        e3t_nemo = compute_layer_thickness(
            ns.ssh, h_bathy, br.z_coord, min_water_column_m=mc.min_water_column_m)
        out["max_de3t"].append(float(np.max(np.abs(
            np.asarray(e3t_lego)[tmask3] - np.asarray(e3t_nemo)[tmask3]))))

        t_now = np.asarray(st.T.data)
        out["max_dT_now"].append(float(np.max(np.abs(t_now[tmask3] - ns.T[tmask3]))))
        out["max_dS_now"].append(float(np.max(np.abs(
            np.asarray(st.S.data)[tmask3] - ns.S[tmask3]))))
        u_face = np.asarray(st.u.data)[:, 1:, :]
        v_face = np.asarray(st.v.data)[1:, :, :]
        out["max_du_now"].append(float(np.max(np.abs(u_face[umask3] - ns.u[umask3]))))
        out["max_dv_now"].append(float(np.max(np.abs(v_face[vmask3] - ns.v[vmask3]))))
        out["max_deta_now"].append(float(np.max(np.abs(
            np.asarray(st.eta.data)[wet2] - ns.ssh[wet2]))))
        out["mean_T_now"].append(float(np.mean(t_now[tmask3])))

        # --- paths 1+2: the ASSELIN-FILTERED before-level state carried
        # into step k+1's entry, vs NEMO's tb/sb/ub/vb at the SAME step ----
        nb = nemo_before_state_at(kt, run_twin_step1=run_twin_step1)
        t_before = np.asarray(st.T_before.data)
        out["max_dT_before"].append(float(np.max(np.abs(t_before[tmask3] - nb.T[tmask3]))))
        out["max_dS_before"].append(float(np.max(np.abs(
            np.asarray(st.S_before.data)[tmask3] - nb.S[tmask3]))))
        ub_face = np.asarray(st.u_before.data)[:, 1:, :]
        vb_face = np.asarray(st.v_before.data)[1:, :, :]
        out["max_du_before"].append(float(np.max(np.abs(ub_face[umask3] - nb.u[umask3]))))
        out["max_dv_before"].append(float(np.max(np.abs(vb_face[vmask3] - nb.v[vmask3]))))

        # --- path 3: prognostic TKE (en) carry ------------------------------
        # drop surface w-level (see bridge_tke_from_restart's convention)
        en_nemo = nemo_en_at(kt, run_twin_step1=run_twin_step1)[..., 1:]
        tke_lego = np.asarray(st.tke.data)
        out["max_den"].append(float(np.max(np.abs(tke_lego[wet2] - en_nemo[wet2]))))

        out["kt"].append(kt)

    return {k: np.array(v) for k, v in out.items()}
