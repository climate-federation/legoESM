#!/usr/bin/env python
"""LADDER: where inside NEMO's kt=1 barotropic solve does legoESM leave it?

The step-1 gate says sea level is the largest fraction of NEMO's own first
step (about 1%), and neither the surface-forcing arm nor ``mlf_baro_corr``
moved it.  Sea level after the step IS the barotropic loop's own output:
``ssh_nxt`` writes ``ssh(Kaa)`` at stpmlf.f90:248 and ``dyn_spg_ts``
OVERWRITES it at dynspg_ts.f90:1006 with the boxcar average, and the Euler
step skips ``ssh_atf`` (sshwzv.f90:443), so the restart's now-level ``sshn``
is exactly ``pssh(:,:,Kaa)``.

So this walks the loop from its inputs to its outputs and scores each rung
that a RACE-FREE oracle exists for.

WHY "RACE-FREE" IS THE WHOLE PROBLEM HERE.  The record's per-substep streams
cannot be used, and that is a property of the record, not of the comparison.
``ll_spg_dump`` (dynspg_ts.F90:206) and the substep-trajectory guard (:926)
test only ``kt == nit000``; neither tests ``narea``.  This record ran on 16
MPI ranks in one directory, so all 16 OPEN the same filename with
STATUS='REPLACE' and write concurrently.  Section 1 proves the damage three
ways rather than assuming it.  The rungs those streams would have scored are
printed UNMEASURED with the record that would close them -- never silently
dropped (Rule 1).

The rungs that ARE scorable come from the restart tiles, which are written
through XIOS with DOMAIN attributes and are not raced:

  R0  the layer thicknesses  every rung below is a depth-weighted mean or a
      the state sits in        face-column depth, so a wrong thickness reaches
                               all of them (Rule 2: the state can be right
                               while the boxes holding it are the wrong size).
                               NEMO's DINO carries TWO ladders -- the 1-D
                               reference e3t_1d and the 3-D e3t_0 that
                               zgr_sco_mi96 returns for its flat 4000 m column
                               -- and they part company below level 25.
                               legoESM runs on the first, NEMO on the second.
  R2  loop-entry seed        pssh/puu_b/pvv_b at Kbb (dynspg_ts.f90:494-503).
                             From rest these are identically zero, and NEMO's
                             own ub/vb confirm it.
  R4a barotropic velocity    puu_b(:,:,Kaa) -- the boxcar average of the
      average                substep velocities (dynspg_ts.f90:1005), sampled
                             into the restart as ``ubtacc_aa`` = uu_b(Naa) *
                             r1_Dt right after the dyn_spg call
                             (stpmlf.F90 trddump_acc_baro, trddump.F90:161).
  R4b barotropic sea level   pssh(:,:,Kaa) == restart ``sshn`` (above).
      average

Both R4 rungs are the loop's EXIT, so a break there says "somewhere in the
45 substeps" and no more.  Localising it needs the per-substep record that
section 1 shows this run cannot provide.  ``scripts/validate/ocean_fidelity/dino_1226/nemo_dino_kt1_rankdump/run.sh`` acquires it, as a makenemo
CONFIG COPY that cannot write inside the read-only oracle.

REMOVED 2026-09-10, and it is worth saying why rather than letting it vanish
from the diff: this script used to EMIT that re-run itself, and the script it
emitted edited ``cfgs/DINO/MY_SRC/dynspg_ts.F90`` IN PLACE behind an exit trap
and then ran ``makenemo -r DINO -n DINO``, which overwrites the oracle's own
``cfgs/DINO/BLD`` and ``bin/nemo.exe`` -- and no trap restores those.  A
reviewer found it still reachable after the config-copy acquisition landed.
The safe acquisition replaces it; this script no longer writes anything under
the oracle.

Rule 10: the legoESM side is the production solver's own arrays, captured
through the private ``_nemo_substep_trace_test_hook`` (no numerical selector
is introduced) and through a wrapper that records the solver's arguments.
The loop's time-averaged velocity is not returned by the solver, so it is
rebuilt from the traced per-substep velocities; section 3 CALIBRATES that
rebuild by applying the identical method to sea level, whose average the
solver DOES return, and requiring bit equality.

Usage (GPU only -- this card's XLA CPU compile crashes on this machine):

    CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\
    python scripts/validate/ocean_fidelity/dino_1226/\\
        spg_kt1_barotropic_ladder.py [--plant]

``--plant`` moves one wet cell of EVERY rung that is currently AT BAR -- the
carried mesh ladder and all three loop-entry seeds -- by 1 ulp.  All four MUST
then read DEBT.  Planting into a row that already reads DEBT would prove
nothing, which is why the plant does not touch one.
"""
from __future__ import annotations

import argparse
import glob
import os
import struct
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from rebuild_nemo_restart import rebuild                       # noqa: E402

RUN_KT1 = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
           "RUN_FROMREST_KT1")
RN_DT = 2700.0                      # namdom rn_Dt
R1_DT = 1.0 / RN_DT                 # stpmlf.F90:131-133 on the Euler step
NN_E = 23                           # ln_bt_auto resolved value (ocean.output:1102)
ICYCLE = 45                         # forward-start window (ocean.output:1265)


# ---------------------------------------------------------------- section 1
def audit_record(run_dir: str) -> dict:
    """Is this record's dyn_spg_ts instrumentation usable?  Three checks.

    FAILS CLOSED.  ``out["checks_run"]`` counts how many of the three actually
    produced evidence; a missing stream, an unreadable file or a netCDF4
    failure leaves its check unrun, and the caller must read fewer than three
    as INDETERMINATE, never as "coherent".  An audit that says a record is
    usable when it simply could not look is the same defect as a gate that
    passes because it scored nothing.
    """
    out = {"checks_run": 0}
    # (a) the substep trajectory's own header disagrees with its length.
    path = os.path.join(run_dir, "substep_dump.bin")
    if os.path.exists(path):
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            jpi, jpj, icy = struct.unpack("<3i", fh.read(12))
        need = 12 + icy * (4 + 7 * jpi * jpj * 8)
        # the jn markers must read 1..icy under the header's own stride
        with open(path, "rb") as fh:
            raw = fh.read()
        markers = []
        stride = 4 + 7 * jpi * jpj * 8
        off = 12
        for _ in range(icy):
            if off + 4 > len(raw):
                break
            markers.append(struct.unpack("<i", raw[off:off + 4])[0])
            off += stride
        out["substep_header"] = (jpi, jpj, icy)
        out["substep_size_bytes"] = size
        out["substep_size_implied"] = need
        out["substep_markers_ok"] = markers == list(range(1, icy + 1))
        out["checks_run"] += 1
    # (b) two streams that must satisfy a closed identity, do not.
    #     From rest at jn=1: ssh_frc == 0 (no E-P) so ssha_e(1) == 0, hence
    #     zu_spg == 0; the seed is rest so dyn_cor_2D and the bottom stress
    #     are 0 too.  dynspg_ts.f90:839-845 then reduces to
    #         un_e(after jn=1) = rDt_e * zu_frc * ssumask.
    rdt_e = RN_DT / NN_E
    try:
        zu = np.fromfile(os.path.join(run_dir, "spg_dump_zu_frc.bin"),
                         dtype="<f8")
        ub1 = np.fromfile(os.path.join(run_dir, "spg_dump_ub_substep1.bin"),
                          dtype="<f8")
        sfrc = np.fromfile(os.path.join(run_dir, "spg_dump_ssh_frc.bin"),
                           dtype="<f8")
        ssh1 = np.fromfile(os.path.join(run_dir, "spg_dump_ssh_substep1.bin"),
                           dtype="<f8")
        nj = 29
        ni = 30
        ub1 = ub1.reshape(nj, ni)[2:-2, 2:-2].ravel()
        out["ssh_frc_zero"] = bool((sfrc == 0).all())
        out["ssh_substep1_zero"] = bool((ssh1 == 0).all())
        pred = rdt_e * zu
        out["identity_cells_unequal"] = int((pred != ub1).sum())
        out["identity_n"] = int(pred.size)
        out["identity_max_abs"] = float(np.abs(pred - ub1).max())
        out["identity_field_max"] = float(np.abs(ub1).max())
        out["checks_run"] += 1
    except (OSError, ValueError) as exc:                 # pragma: no cover
        out["identity_error"] = str(exc)
    # (c) different streams carry different ranks' tiles.  Land patterns of
    #     the DINO basin differ between the western and eastern tile columns;
    #     a coherent record cannot show one stream on each.
    try:
        import netCDF4 as nc
        masks = {}
        for p in sorted(glob.glob(os.path.join(run_dir, "mesh_mask_*.nc"))):
            r = int(p[-7:-3])
            ds = nc.Dataset(p)
            masks[r] = {k: np.squeeze(ds.variables[k][:]).astype(bool)
                        for k in ("tmaskutil", "umaskutil")}
            ds.close()

        def which(fname, key, interior):
            a = np.fromfile(os.path.join(run_dir, fname), dtype="<f8")
            a = (a.reshape(25, 26) if interior
                 else a.reshape(29, 30)[2:-2, 2:-2])
            nz = a != 0.0
            return [r for r in sorted(masks)
                    if np.array_equal(nz, masks[r][key])]
        out["tiles_zu_frc"] = which("spg_dump_zu_frc.bin", "umaskutil", True)
        out["tiles_qns"] = which("sbc_dump_qns.bin", "tmaskutil", False)
        out["checks_run"] += 1
    except Exception as exc:                             # pragma: no cover
        out["tile_error"] = str(exc)
    # (d) THE RECORD MAY ALREADY BE RANK-TAGGED.  Checks (a)-(c) all look at
    #     the OLD, un-suffixed filenames; on a record acquired by the config
    #     copy they simply do not exist, and the audit would then report
    #     INDETERMINATE -- "could not look" -- for a record that is in fact
    #     clean.  That is the wrong verdict and it would keep the two rungs
    #     waived forever.  So the rank-tagged layout is detected explicitly,
    #     and it is only accepted when the file COUNT matches the run's own
    #     decomposition (layout.dat's jpnij) times the substep count.
    tagged = sorted(glob.glob(os.path.join(run_dir, "substep_r*_s*.bin")))
    if tagged:
        jpnij = None
        lay = os.path.join(run_dir, "layout.dat")
        if os.path.exists(lay):
            with open(lay) as fh:
                for ln in fh:
                    parts = ln.split()
                    if len(parts) >= 6 and parts[0].isdigit():
                        jpnij = int(parts[0])
                        break
        ranks = sorted({os.path.basename(t).split("_")[1] for t in tagged})
        steps = sorted({os.path.basename(t).split("_")[2] for t in tagged})
        out["rank_tagged_files"] = len(tagged)
        out["rank_tagged_ranks"] = len(ranks)
        out["rank_tagged_substeps"] = len(steps)
        out["rank_tagged_jpnij"] = jpnij
        out["rank_tagged_complete"] = bool(
            jpnij is not None and len(ranks) == jpnij
            and len(tagged) == len(ranks) * len(steps))
        out["checks_run"] += 1
    return out


#: The acquisition that produces a race-free record.  A CONFIG COPY: it never
#: writes inside cfgs/DINO or src/, and it refuses rather than trying.
ACQUISITION = ("scripts/validate/ocean_fidelity/dino_1226/"
               "nemo_dino_kt1_rankdump/run.sh")


# ---------------------------------------------------------------- helpers
def _mesh_field(run_dir: str, name: str) -> np.ndarray:
    """Stitch one mesh_mask variable over the 16 tiles (haloless, no race)."""
    import netCDF4 as nc
    out = None
    for p in sorted(glob.glob(os.path.join(run_dir, "mesh_mask_*.nc"))):
        ds = nc.Dataset(p)
        x0, y0 = (int(v) for v in ds.DOMAIN_position_first)
        x1, y1 = (int(v) for v in ds.DOMAIN_position_last)
        nxg, nyg = (int(v) for v in ds.DOMAIN_size_global)
        a = np.squeeze(np.asarray(ds.variables[name][:], dtype=np.float64))
        if out is None:
            out = np.full(a.shape[:-2] + (nyg, nxg), np.nan)
        out[..., y0 - 1:y1, x0 - 1:x1] = a
        ds.close()
    if out is None:
        raise SystemExit(f"no mesh_mask tiles under {run_dir}")
    return out


class Row:
    def __init__(self):
        self.rows = []
        self.fail = False

    def score(self, name, built, oracle, mask, floor=None):
        b = np.asarray(built, dtype=np.float64)[mask]
        o = np.asarray(oracle, dtype=np.float64)[mask]
        if b.size == 0:
            # An empty comparison is not agreement.  A mask that selects
            # nothing would otherwise read AT BAR at n=0.
            self.fail = True
            self.rows.append(dict(name=name, n=0, cells=None, maxabs=None,
                                  rms=None, floor=None, status="UNMEASURED",
                                  reason="the mask selected no cells"))
            return self.rows[-1]
        ne = int((b != o).sum())
        d = b - o
        row = dict(name=name, n=int(b.size), cells=ne,
                   maxabs=float(np.abs(d).max()) if b.size else 0.0,
                   rms=float(np.sqrt(np.mean(d ** 2))) if b.size else 0.0,
                   floor=floor, status="AT BAR" if ne == 0 else "DEBT")
        if ne:
            self.fail = True
        self.rows.append(row)
        return row

    def unmeasured(self, name, reason):
        self.rows.append(dict(name=name, n=0, cells=None, maxabs=None,
                              rms=None, floor=None, status="UNMEASURED",
                              reason=reason))

    def render(self):
        print(f"{'rung':38s}{'cells !=':>10s}{'max|d|':>13s}"
              f"{'rms':>13s}{'NEMO step':>13s}  status")
        for r in self.rows:
            if r["status"] == "UNMEASURED":
                print(f"{r['name']:38s}{'-':>10s}{'-':>13s}{'-':>13s}"
                      f"{'-':>13s}  UNMEASURED")
                print(f"{'':38s}  needs: {r['reason']}")
                continue
            fl = "-" if r["floor"] is None else f"{r['floor']:13.4e}"
            print(f"{r['name']:38s}{r['cells']:>10d}{r['maxabs']:13.4e}"
                  f"{r['rms']:13.4e}{fl:>13s}  {r['status']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=RUN_KT1)
    ap.add_argument("--plant", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--save-trace", default=None,
                    help="npz path for legoESM's per-substep trajectory")
    args = ap.parse_args()

    print("=" * 78)
    print("1. RECORD AUDIT -- are this run's dyn_spg_ts streams usable?")
    print("=" * 78)
    a = audit_record(args.run_dir)
    raced = False
    if "substep_header" in a:
        jpi, jpj, icy = a["substep_header"]
        print(f"  substep_dump.bin header says jpi={jpi} jpj={jpj} "
              f"icycle={icy}")
        print(f"    -> that header implies {a['substep_size_implied']} bytes; "
              f"the file is {a['substep_size_bytes']}")
        print(f"    -> substep markers read 1..{icy} in order: "
              f"{a['substep_markers_ok']}")
        if (a["substep_size_implied"] != a["substep_size_bytes"]
                or not a["substep_markers_ok"]):
            raced = True
    if "identity_cells_unequal" in a:
        print(f"  ssh_frc identically zero: {a['ssh_frc_zero']}   "
              f"ssh after substep 1 identically zero: "
              f"{a['ssh_substep1_zero']}")
        print("  closed identity  un_e(after jn=1) = rDt_e * zu_frc * ssumask")
        print(f"    -> cells unequal {a['identity_cells_unequal']}"
              f"/{a['identity_n']}   max|d| {a['identity_max_abs']:.4e}"
              f"   (the field's own max is {a['identity_field_max']:.4e})")
        if a["identity_cells_unequal"]:
            raced = True
    if "tiles_zu_frc" in a:
        print(f"  land pattern of spg_dump_zu_frc.bin matches ranks "
              f"{a['tiles_zu_frc']}")
        print(f"  land pattern of sbc_dump_qns.bin  matches ranks "
              f"{a['tiles_qns']}")
        if set(a["tiles_zu_frc"]).isdisjoint(a["tiles_qns"]):
            raced = True
    if "rank_tagged_files" in a:
        print(f"  RANK-TAGGED substep streams present: "
              f"{a['rank_tagged_files']} files = {a['rank_tagged_ranks']} "
              f"ranks x {a['rank_tagged_substeps']} substeps; layout.dat "
              f"says jpnij={a['rank_tagged_jpnij']}; complete: "
              f"{a['rank_tagged_complete']}")
    print()
    _n = int(a.get("checks_run", 0))
    _tagged_ok = bool(a.get("rank_tagged_complete"))
    if raced and not _tagged_ok:
        _verdict = ("RACED -- these streams are a 16-rank interleave and "
                    "cannot be scored")
    elif _tagged_ok:
        # A rank-tagged record does not need checks (a)-(c): those exist to
        # detect an interleave, and one rank per file cannot interleave.  The
        # completeness check above is what stands in their place, and it is
        # against the run's OWN decomposition rather than a constant.
        _verdict = ("RANK-TAGGED and complete -- one file per (rank, "
                    "substep), so there is no interleave to detect")
        raced = False
    elif _n < 3:
        _verdict = (f"INDETERMINATE -- only {_n}/3 checks could run, so this "
                    "audit did not look; treat the streams as unusable")
        raced = True
    else:
        _verdict = "coherent -- all 3 checks ran and none found damage"
    print("  VERDICT: " + _verdict)
    if raced:
        print("  cause: dynspg_ts.F90:206 and :926 guard on kt only, never "
              "on narea;")
        print("         all 16 ranks OPEN the same path with "
              "STATUS='REPLACE'.")
        print("  fix:   " + ACQUISITION + " acquires a clean record")
    if args.audit_only:
        return 0

    # ------------------------------------------------------------ oracle
    print()
    print("=" * 78)
    print("2. THE LADDER")
    print("=" * 78)
    import jax
    if not jax.config.jax_enable_x64:
        # BEFORE set_policy: PrecisionPolicy.fp64() turns x64 on itself, so a
        # check after it can never fire.
        raise SystemExit("JAX x64 is disabled; re-run with JAX_ENABLE_X64=1")
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64())                            # Rule 1c
    import jax.numpy as jnp
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocean_model
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        compute_layer_thickness, min_cell_to_uface)

    pattern = os.path.join(args.run_dir, "DINO_00000001_restart_*.nc")
    if not glob.glob(pattern):
        raise SystemExit(f"no restart tiles match {pattern}")
    R = rebuild(pattern, ["sshn", "ubtacc_aa", "vbtacc_aa", "ub", "vb",
                          "sshb"])
    for k in ("sshn", "ubtacc_aa", "vbtacc_aa"):
        if k not in R:
            raise SystemExit(f"restart carries no {k!r}: this record predates "
                             "the trddump barotropic accumulator")
    # ``ubtacc_aa`` is a SUM over steps (trddump.F90:161).  It is this step's
    # puu_b(Kaa)*r1_Dt only if exactly one step was folded in; the file's own
    # counter says so, and a multi-step record must not be read as a kt=1
    # oracle.
    import netCDF4 as _nc
    _ns = set()
    for _p in sorted(glob.glob(pattern)):
        _ds = _nc.Dataset(_p)
        if "nacc_steps" in _ds.variables:
            _ns.add(float(np.squeeze(_ds.variables["nacc_steps"][:])))
        _ds.close()
    print(f"  trddump accumulator folded {sorted(_ns)} step(s) "
          "(must be exactly [1.0])")
    if _ns != {1.0}:
        raise SystemExit("ubtacc_aa is not a single-step accumulator here")

    # ------------------------------------------------------- legoESM side
    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    state0 = dm.dino_lat_lon_state(grid, z, cfg)
    forcing = dm.dino_lat_lon_surface_forcing_arrays(grid, cfg)
    sf_step = (dm.dino_step_surface_forcing(forcing)
               if getattr(cfg, "wind_through_step", False) else None)
    mc, _ = dm.dino_lat_lon_model_config(grid, cfg, physics=True)
    print(f"  card: filter={mc.barotropic.barotropic_time_filter!r} "
          f"n_substeps={mc.barotropic.n_barotropic_substeps} "
          f"face_depth={mc.barotropic.barotropic_face_depth!r} "
          f"continuity="
          f"{mc.barotropic.barotropic_continuity_evaluation!r} "
          f"pgf={mc.barotropic.barotropic_pgf_evaluation!r} "
          f"seed={mc.barotropic.barotropic_seed_evaluation!r}")

    original = ocean_model.barotropic_substeps_latlon_cgrid
    cap = {}

    import inspect
    sig = inspect.signature(original)

    def wrapper(*a, **k):
        if "trace" in cap:
            return original(*a, **k)
        ba = sig.bind(*a, **k)
        ba.apply_defaults()
        for nm in ("dt_s", "n_substeps", "F_slow_eta", "F_slow_u", "F_slow_v",
                   "add_barotropic_coriolis"):
            if nm in ba.arguments:
                cap[nm] = ba.arguments[nm]
        k2 = dict(k)
        k2["_nemo_substep_trace_test_hook"] = True
        state_new, transports, trace = original(*a, **k2)
        cap["trace"] = [np.asarray(t) for t in trace]
        cap["eta_out"] = np.asarray(state_new.eta.data)
        return state_new, transports

    model = LatLonCGridOceanModel(grid, z, mc)
    st, rate = dm.apply_dino_lat_lon_surface_forcing(
        state0, forcing, z, cfg, RN_DT, t_seconds=RN_DT, return_rate=True)
    ocean_model.barotropic_substeps_latlon_cgrid = wrapper
    try:
        with jax.disable_jit():
            model.step(st, dt=RN_DT, surface_forcing=sf_step,
                       external_tracer_rate=rate)
    finally:
        ocean_model.barotropic_substeps_latlon_cgrid = original
    if "trace" not in cap:
        raise SystemExit("the barotropic solver was never called")

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import _compute_weights
    n_sub = int(cap["n_substeps"])
    w_filter, w_total, w_transport, n_loop = _compute_weights(
        mc, n_sub, np.float64,
        substep_scale=getattr(mc.barotropic, "barotropic_substep_scale", 1)
        if hasattr(mc.barotropic, "barotropic_substep_scale") else 1)
    w_filter = np.asarray(w_filter, dtype=np.float64)
    w_total = float(np.asarray(w_total))
    print(f"  loop: dt_s={float(np.asarray(cap['dt_s'])):.9f} s  "
          f"n_substeps={n_sub}  n_loop={n_loop}  "
          f"sum(w_filter)={w_filter.sum():.17g}  w_total={w_total:.17g}")
    print(f"  NEMO: rDt_e={RN_DT / NN_E:.9f} s  nn_e={NN_E}  icycle={ICYCLE}")

    # trace layout is pinned by _run_substep_loop's `trace` tuple
    TR = ("eta_entry", "u_entry", "v_entry", "eta_mid", "u_mid", "v_mid",
          "flux_u", "flux_v", "eta_continuity", "eta_pgf", "pgf_u", "pgf_v",
          "slow_u", "slow_v", "drag_u", "drag_v", "u_exit", "v_exit",
          "eta_exit")
    tr = dict(zip(TR, cap["trace"]))
    if tr["eta_exit"].shape[0] != n_loop:
        raise SystemExit(f"trace has {tr['eta_exit'].shape[0]} substeps, "
                         f"loop count is {n_loop}")

    # The rebuild below is the VELOCITY-average form.  The solver has a second
    # form (barotropic_latlon_cgrid.py:1683-1689) that accumulates
    # w*U*H_face and divides the completed mean by a face depth; sea level has
    # no such branch, so section 3's calibration would still pass bit-exactly
    # while the velocity rebuild was wrong.  Refuse rather than report.
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        nemo_flux_form_update_active)   # noqa: F401  (import-time contract)
    _primary_transport = (
        getattr(mc, "momentum_time_integrator", "euler") == "rk3_ws"
        and getattr(mc, "momentum_advection", "vector_invariant") == "flux_form"
        and mc.barotropic.barotropic_time_filter in (
            "nemo_boxcar_ab3", "nemo_boxcar1_ab3"))
    if _primary_transport:
        raise SystemExit(
            "this card resolves to the solver's TRANSPORT primary average; "
            "the velocity rebuild below would be the wrong formula and "
            "section 3's sea-level calibration cannot see it")

    def boxcar(stack):
        """Reproduce the solver's own sequential accumulate-then-divide."""
        acc = np.zeros(stack.shape[1:], dtype=np.float64)
        for i in range(stack.shape[0]):
            acc = acc + w_filter[i] * stack[i]
        return acc / w_total

    # --- section 3: calibrate the rebuild on a quantity the solver returns
    eta_avg_rebuilt = boxcar(tr["eta_exit"])
    eta_cells = int((eta_avg_rebuilt != cap["eta_out"]).sum())
    print()
    print("3. REBUILD CALIBRATION (the method that produces the R4a number)")
    print(f"   sea-level average rebuilt from the trace vs the solver's own "
          f"returned array: {eta_cells} cells unequal, "
          f"max|d| {np.abs(eta_avg_rebuilt - cap['eta_out']).max():.3e}")
    if eta_cells:
        raise SystemExit(
            "the rebuild does not reproduce the solver's own average, so the "
            "velocity number it would produce is not the model's")

    u_avg = boxcar(tr["u_exit"])
    v_avg = boxcar(tr["v_exit"])

    g = ndm.nemo_dino_mesh()
    twet = g.tmask[:, :, 0] > 0.5
    uwet = g.umask[:, :, 0] > 0.5
    vwet = g.vmask[:, :, 0] > 0.5

    # ---------------------------------------------------------- R0
    # THE BOXES BEFORE THE STATE IN THEM (Rule 2).  Every rung below is a
    # depth-weighted mean or a face-column depth, so a wrong layer thickness
    # reaches all of them.
    #
    # NEMO's DINO has TWO vertical ladders and they are not the same one.
    # usr_def_zgr takes the ld_zco branch (ln_zco_nam = .true.) and calls
    # zgr_sco_mi96 on a FLAT column, zflat(:,:) = zHmax = 4000 m
    # (usrdef_zgr.F90:107-118) -- so the 3-D e3t_0 it returns is horizontally
    # UNIFORM (measured: column-to-column spread exactly 0.0 at every level)
    # but it is NOT the 1-D reference ladder e3t_1d.  The two agree for
    # k = 1..25 and then part company: e3t_0/e3t_1d = 0.979, 0.945, 0.924,
    # 0.915, 0.919, 0.935, 0.965, 1.009, 1.070, 1.148 at k = 26..35.
    # legoESM runs on e3t_1d.  NEMO runs on e3t_0.
    #
    # RETRACTED, 2026-09-10: an earlier reading of this called it per-column
    # stretching to the local bathymetry, citing ocean.output:402 ("zgr_lib:
    # zgr_sco : define full vertical s-coord. system using the 2d bathymetry
    # field") and ln_hpg_sco = T.  A reviewer refuted it and the measurement
    # agrees: that log line is printed INSIDE zgr_sco_mi96, which both
    # branches call, and ln_hpg_sco names the pressure-gradient scheme, not
    # the coordinate.  Prose is a pointer, never a citable fact.  The
    # NUMBERS below are unchanged; only the mechanism was wrong, and the
    # correct one makes the gap a 1-D ladder mismatch -- horizontally
    # uniform, deep levels only.
    e3t0 = np.moveaxis(_mesh_field(args.run_dir, "e3t_0"), 0, -1)
    tmask3 = np.moveaxis(_mesh_field(args.run_dir, "tmask"), 0, -1) > 0.5
    e3u0 = np.moveaxis(_mesh_field(args.run_dir, "e3u_0"), 0, -1)
    umask3 = np.moveaxis(_mesh_field(args.run_dir, "umask"), 0, -1) > 0.5
    hu0 = (e3u0 * umask3).sum(axis=-1)
    _h_live = compute_layer_thickness(
        st.eta.data, st.H_bathy.data, z,
        min_water_column_m=mc.min_water_column_m)
    h_live = np.asarray(_h_live)
    Hu_live = np.asarray(jnp.sum(min_cell_to_uface(_h_live), axis=-1))[:, 1:]

    tab = Row()
    tab.score("R0a live layer thickness  e3t_0",
              h_live, e3t0, tmask3)
    tab.score("R0b U-face column depth   hu_0",
              Hu_live, hu0, uwet, floor=float(np.sqrt(np.mean(hu0[uwet] ** 2))))
    # The card DOES carry NEMO's own ladder -- it just does not run on it.
    ne3t = getattr(z, "nemo_e3t_0", None)
    if ne3t is not None:
        ne3t = np.asarray(ne3t)
        if args.plant:
            # Non-vacuity.  Plant into the two rungs that are AT BAR -- a
            # plant on a rung that is already DEBT proves nothing about the
            # gate.  Both must flip.
            ne3t = ne3t.copy()
            _p = np.argwhere(tmask3)[0]
            ne3t[_p[0], _p[1], _p[2]] = np.nextafter(
                ne3t[_p[0], _p[1], _p[2]], np.inf)
        tab.score("R0c carried mesh ladder   e3t_0", ne3t, e3t0, tmask3)
    # R2 -- the loop-entry seed.  From rest NEMO's is identically zero.
    _seed_eta = np.asarray(tr["eta_entry"][0])
    if args.plant:
        _seed_eta = _seed_eta.copy()
        _q = np.argwhere(twet)[0]
        _seed_eta[_q[0], _q[1]] = np.nextafter(_seed_eta[_q[0], _q[1]], np.inf)
    tab.score("R2 seed eta   pssh(Kbb)", _seed_eta, R["sshb"], twet)
    # NEMO does not write puu_b(Kbb); it writes the 3-D ub/vb the depth mean
    # is built from (dynatf_qco.F90:222-235).  Build the mean here with
    # NEMO's own e3u_0/hu_0 weights rather than asserting zero -- a hard-wired
    # zero would read AT BAR even if the record's ub were not zero.
    e3v0 = np.moveaxis(_mesh_field(args.run_dir, "e3v_0"), 0, -1)
    vmask3 = np.moveaxis(_mesh_field(args.run_dir, "vmask"), 0, -1) > 0.5
    hv0 = (e3v0 * vmask3).sum(axis=-1)
    ub3 = np.moveaxis(R["ub"], 0, -1)
    vb3 = np.moveaxis(R["vb"], 0, -1)
    seed_u_oracle = np.where(
        hu0 > 0, (e3u0 * ub3 * umask3).sum(axis=-1) / np.where(hu0 > 0, hu0, 1.0), 0.0)
    seed_v_oracle = np.where(
        hv0 > 0, (e3v0 * vb3 * vmask3).sum(axis=-1) / np.where(hv0 > 0, hv0, 1.0), 0.0)
    _seed_u = np.asarray(tr["u_entry"][0])[:, 1:]
    _seed_v = np.asarray(tr["v_entry"][0])[1:, :]
    if args.plant:
        _seed_u = _seed_u.copy()
        _seed_v = _seed_v.copy()
        _a = np.argwhere(uwet)[0]
        _b = np.argwhere(vwet)[0]
        _seed_u[_a[0], _a[1]] = np.nextafter(_seed_u[_a[0], _a[1]], np.inf)
        _seed_v[_b[0], _b[1]] = np.nextafter(_seed_v[_b[0], _b[1]], np.inf)
    tab.score("R2 seed U     puu_b(Kbb)", _seed_u, seed_u_oracle, uwet)
    tab.score("R2 seed V     pvv_b(Kbb)", _seed_v, seed_v_oracle, vwet)
    # R1/R3 -- their disposition is DECIDED BY the audit, not hard-coded.
    # If a later record passes the audit these rungs become scoreable, and a
    # gate that kept printing UNMEASURED would hide that.
    if raced:
        reason = ("a rank-tagged spg dump (" + ACQUISITION + "); "
                  "this record's "
                  "streams are a 16-rank interleave (section 1)")
        tab.unmeasured("R1 frozen forcing zu_frc/zv_frc/ssh_frc", reason)
        tab.unmeasured("R3 per-substep ssha_e/ua_e/va_e (x45)", reason)
    else:
        tab.rows.append(dict(
            name="R1/R3 spg streams", n=0, cells=None, maxabs=None, rms=None,
            floor=None, status="UNMEASURED",
            reason="NOTHING -- section 1 says this record's streams are "
                   "usable, so these rungs are now scoreable and this gate "
                   "has not been extended to score them. Extend it."))
        tab.fail = True
    # R4 -- the loop exit, from race-free restart fields
    floor_u = float(np.sqrt(np.mean((R["ubtacc_aa"][uwet]) ** 2)))
    floor_v = float(np.sqrt(np.mean((R["vbtacc_aa"][vwet]) ** 2)))
    floor_e = float(np.sqrt(np.mean((R["sshn"][twet]) ** 2)))
    tab.score("R4a barotropic U avg  puu_b(Kaa)",
              u_avg[:, 1:] * R1_DT, R["ubtacc_aa"], uwet, floor=floor_u)
    tab.score("R4a barotropic V avg  pvv_b(Kaa)",
              v_avg[1:, :] * R1_DT, R["vbtacc_aa"], vwet, floor=floor_v)
    tab.score("R4b barotropic eta avg pssh(Kaa)",
              cap["eta_out"], R["sshn"], twet, floor=floor_e)
    print()
    tab.render()
    print()
    print("  'NEMO step' is the rms of NEMO's own answer for that field, so a "
          "residual is read as a fraction of the motion it belongs to.")

    if args.save_trace:
        os.makedirs(os.path.dirname(args.save_trace) or ".", exist_ok=True)
        np.savez_compressed(
            args.save_trace,
            w_filter=w_filter, w_total=w_total,
            dt_s=float(np.asarray(cap["dt_s"])), n_loop=n_loop,
            **{k: v for k, v in tr.items()})
        print(f"  legoESM per-substep trajectory -> {args.save_trace}")

    if tab.fail:
        print("\nGATE FAIL: at least one rung is off the bar (0 cells "
              "unequal).")
        return 1
    print("\nGATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
