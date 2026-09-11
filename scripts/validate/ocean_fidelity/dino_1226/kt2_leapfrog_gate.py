#!/usr/bin/env python
"""NEMO's kt=2 LEAP-FROG step on the standalone DINO card.

Preregistered in ``PREREG_kt1_kmm_stretch_and_rdt.md`` (read it first).

WHY kt=2.  Three statements this branch transcribes are DEGENERATE on the
from-rest first step, so every number measured so far is silent about them:

  rDt         ``domain.f90:288`` sets ``rDt = 2*rn_Dt`` for the modified
              leap-frog; ``stpmlf.f90:131-133`` reduces it to ``rn_Dt`` while
              ``l_1st_euler`` and ``:617-620`` restores it at the END of that
              step.  At kt=1 a card that used ``rn_Dt`` everywhere would score
              identically.  ``traldf_iso.f90:829-830`` reads ``rDt`` and
              ``r1_Dt``, so this is where that statement can be wrong.

  ssh_atf     ``stpmlf.f90:425`` Asselin-filters ``ssh(:,:,Nnn)`` BEFORE
              ``tra_ldf`` at ``:504``, while ``r3t(:,:,Nnn)`` is not refreshed
              until ``:571``.  The filter is guarded ``IF(.NOT.l_1st_euler)``
              (``sshwzv.f90:443``), so at kt=1 the two "now" heights coincide
              and the ``Kmm``-stretch transcription is untested.

  akz         the stabilising correction fires on 0 of 342134 wet cells at
              kt=1, so ``traldf_iso_a33``'s own ``e3w`` and the implicit half
              of the explicit/implicit split are multiplied by nothing.

WHAT IT DOES.  Feeds legoESM NEMO's OWN kt=1 restart as the BEFORE (Kbb) and
NOW (Kmm) levels, runs ONE step through the model's own path, and scores the
after-state against NEMO's kt=2 restart -- plus the ``tra_ldf`` tendency
against NEMO's own ``ttrd_ldf``/``strd_ldf`` at kt=2.

THE BAR IS EXACT: zero cells unequal, per field (Rule 1b).  Rows that are not
at the bar say DEBT.

Usage
-----
    CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\
        python scripts/validate/ocean_fidelity/dino_1226/kt2_leapfrog_gate.py \\
            --run-dir /data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt2_trends

``--run-dir`` is what ``nemo_dino_kt2_trends/run.sh`` produced; it holds the
``DINO_00000002_restart_*.nc`` tiles (the after state and the trends).  It does
NOT hold a kt=1 restart -- with ``nn_stock = 2`` NEMO sets ``nitrst = kt +
nn_stock - 1`` at kt=1 (``restart.f90:112``) and writes only at ``nitrst``
(``:121``).  The BEFORE/NOW state therefore comes from ``--kt1-dir``, the
certified ``RUN_FROMREST_KT1``, and that substitution is underwritten by a
MEASUREMENT rather than by determinism as a belief: two independent runs of
this configuration wrote bit-identical kt=1 restarts (``read_rankdump.py
--twin-check`` on the kt=1 slopes record against ``RUN_FROMREST_KT1``: 16
tiles, every variable, 0 fatal, 0 admitted).

NON-VACUITY.  ``--plant-euler`` drops the before-level fields so the step takes
the EULER branch (``rdt = rn_Dt`` instead of ``2*rn_Dt``).  Every row must then
move; a gate whose rows survive that plant is not measuring a leap-frog step at
all, which is the one thing this record exists to test.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from rebuild_nemo_restart import rebuild                       # noqa: E402

RN_DT = 2700.0                   # namdom rn_Dt, cfgs/DINO/*/namelist_cfg:116

STATE_FIELDS = ("tn", "sn", "sshn", "un", "vn", "tb", "sb", "sshb", "ub",
                "vb", "utau_b", "vtau_b", "en", "avm_k", "avt_k", "dissl")

#: Rule 1 (coverage) over the kt=1 restart: every variable NEMO's step 2 READS
#: back out of it, with what carries it on the legoESM side.  Diagnostic
#: trend/accumulator arrays (``?trd_*``, ``?trdacc_*``, ``*acc_*``, ``*_stg``)
#: are outputs of the step, never inputs to the next one, and are excluded by
#: prefix rather than one by one.
CARRIES = {
    "tn": "state.T", "sn": "state.S", "sshn": "state.eta",
    "un": "state.u", "vn": "state.v",
    "tb": "state.T_before", "sb": "state.S_before",
    "sshb": "state.eta_before", "ub": "state.u_before", "vb": "state.v_before",
    "utau_b": "state.tau_x_prev", "vtau_b": "state.tau_y_prev",
    "en": "state.tke", "avm_k": "state.tke_avm", "avt_k": "state.tke_avt",
    "dissl": "state.tke_dissl",
    "rhd": None, "rdt": None, "kt": None, "ndastp": None, "adatrj": None,
    "ntime": None, "time_counter": None, "nav_lon": None, "nav_lat": None,
    "nav_lev": None,
    "qns_b": None, "emp_b": None, "sfx_b": None, "sbc_hc_b": None,
    "sbc_sc_b": None, "qsr_hc_b": None, "fraqsr_1lev": None,
}
TREND_FIELDS = ("ttrd_ldf", "strd_ldf", "ttrd_tot", "strd_tot")


def _score(name, lego, nemo, wet, step):
    """One row: cells unequal at the EXACT bar, plus the size of NEMO's own
    step for that field so the residual is read against the motion it is a
    fraction of (Rule 3)."""
    d = np.abs(np.asarray(lego) - np.asarray(nemo))[wet]
    n = int((d != 0.0).sum())
    rms = float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan")
    print(f"  {name:14s}{n:>10d}{d.max() if d.size else np.nan:14.4e}"
          f"{rms:14.4e}{step:14.4e}  {'AT BAR' if n == 0 else 'DEBT'}")
    return 0 if n == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="the DINO_KT2_TRENDS record directory")
    ap.add_argument("--kt1-dir", default=None,
                    help="where the kt=1 restart lives (default: --run-dir if "
                         "it has one, else the certified RUN_FROMREST_KT1)")
    ap.add_argument("--plant-euler", action="store_true",
                    help="drop the before levels so the step takes the EULER "
                         "branch (rdt = rn_Dt); every row MUST then move")
    a = ap.parse_args()

    CERTIFIED_KT1 = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
                     "DINO/RUN_FROMREST_KT1")
    kt1_dir = a.kt1_dir
    if kt1_dir is None:
        kt1_dir = (a.run_dir
                   if glob.glob(os.path.join(
                       a.run_dir, "DINO_00000001_restart_*.nc"))
                   else CERTIFIED_KT1)
    k1 = os.path.join(kt1_dir, "DINO_00000001_restart_*.nc")
    k2 = os.path.join(a.run_dir, "DINO_00000002_restart_*.nc")
    print(f"BEFORE/NOW (kt=1) restart from {kt1_dir}"
          + ("" if kt1_dir == a.run_dir else
             "  <- NOT --run-dir: nn_stock=2 means NEMO wrote only the kt=2 "
             "restart (restart.f90:112, :121)"))
    for pat in (k1, k2):
        if not glob.glob(pat):
            raise SystemExit(
                f"no tiles match {pat}.\n"
                "Produce the record first:\n"
                "  scripts/validate/ocean_fidelity/dino_1226/"
                "nemo_dino_kt2_trends/run.sh")
    if len(glob.glob(k1)) != len(glob.glob(k2)):
        raise SystemExit(
            f"{len(glob.glob(k1))} kt=1 tiles but {len(glob.glob(k2))} kt=2 "
            "tiles -- the two sides are not the same decomposition")
    R1 = rebuild(k1, list(STATE_FIELDS))
    R2 = rebuild(k2, list(STATE_FIELDS) + list(TREND_FIELDS))

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())                          # Rule 1c
    import jax
    import numpy as _np
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    from legoesm.ocean.fidelity.nemo_io import (
        NemoBeforeState, NemoState, read_nemo_mesh_mask)
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        bridge_nemo_to_legoesm_topo, bridge_before_state_topo)
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    import legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid as gmmod

    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    mc, _ = dm.dino_lat_lon_model_config(grid, cfg, physics=True)
    model = LatLonCGridOceanModel(grid, z, mc)

    # The BEFORE/NOW state is NEMO's own kt=1 restart, bridged the way every
    # other restart-seeded probe on this branch bridges it -- not rebuilt.
    # The REBUILT global mesh_mask, which is what nemo_dino_mesh_gate.py
    # certifies against.  The from-rest records hold only per-rank
    # ``mesh_mask_00NN.nc`` tiles, and bridging tile 0000 as if it were the
    # domain would have silently built 1/16 of the mesh.
    mesh = os.path.join(a.run_dir, "mesh_mask.nc")
    if not os.path.exists(mesh):
        mesh = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
                "RUN_TRAJ/mesh_mask.nc")
    g = read_nemo_mesh_mask(mesh, nn_hls=0)
    # The kt=1 state comes from the STITCHED tiles, not from a single file:
    # NEMO wrote 16 per-rank restarts and no rebuilt one exists, and reading
    # tile 0000 as if it were the domain would have silently bridged 1/16 of
    # the ocean.  ``rebuild`` returns (nlev, nlat, nlon) haloless global
    # fields; ``NemoState`` wants (nlat, nlon, nlev), which is exactly ``O3``.
    def _O3(k, R):
        return _np.moveaxis(R[k], 0, -1)

    now = NemoState(T=_O3("tn", R1), S=_O3("sn", R1), u=_O3("un", R1),
                    v=_O3("vn", R1), ssh=R1["sshn"], rhd=None)
    br = bridge_nemo_to_legoesm_topo(g, now, periodic_i=True, full_step=True,
                                     omega=cfg.omega)
    before = NemoBeforeState(
        T=_O3("tb", R1), S=_O3("sb", R1), u=_O3("ub", R1), v=_O3("vb", R1),
        ssh=R1["sshb"], tau_x=R1.get("utau_b"), tau_y=R1.get("vtau_b"))

    # ---- Rule 1: every kt=1 restart variable the NEXT step reads, disposed.
    import netCDF4 as _nc
    _d = _nc.Dataset(sorted(glob.glob(k1))[0])
    _skip = ("utrd_", "vtrd_", "ttrd_", "strd_", "utrdacc_", "vtrdacc_",
             "uacc_", "vacc_", "ubtacc_", "vbtacc_", "nacc_", "rn_acc_")
    _vars = [v for v in sorted(_d.variables)
             if not v.startswith(_skip) and not v.endswith("_stg")]
    _d.close()
    unknown = [v for v in _vars if v not in CARRIES]
    # bad_cov is the Rule-1 count: an UNCLASSIFIED variable must FAIL the
    # gate, not merely be printed.  The first version printed it and never
    # touched the exit path, and a reviewer showed that four real NEMO
    # carries (ssha, sshbb, ub2_b, vb2_b) could be added to the restart and
    # the gate would still say GATE PASS.
    bad_cov = len(unknown)
    print(f"\nCOVERAGE of the kt=1 restart (Rule 1): {len(_vars)} variables "
          "the next step can read; trend/accumulator outputs excluded by "
          "prefix")
    if unknown:
        print(f"  UNCLASSIFIED (hard failure, {bad_cov} of them): {unknown}")
        print("    ^^ each of these is either a carry the next step reads or "
              "a waiver nobody wrote; Rule 1 admits no third option.")
    print("  bridged onto the state: "
          + ", ".join(v for v in _vars if CARRIES.get(v)))
    print("  NOT bridged: "
          + ", ".join(v for v in _vars if CARRIES.get(v) is None
                      and v in CARRIES))
    print("    of these, the ones NEMO's step 2 genuinely reads are the "
          "SURFACE carries qns_b/emp_b/sfx_b/sbc_hc_b/sbc_sc_b/qsr_hc_b/"
          "fraqsr_1lev; legoESM re-evaluates its surface forcing from the "
          "card every step instead of carrying them, so the STATE rows below "
          "are NOT attributable to a leap-frog statement while that stands.")

    # The TKE closure's integrator memory IS carried by legoESM's state, so it
    # is bridged with the harness that already exists (kamm_twin_90d's
    # --bridge-tke), not re-implemented.  Whether the coefficients travel with
    # it is the CARD's own resolved choice, printed rather than assumed.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "kamm_twin_90d", os.path.join(_HERE, "kamm_twin_90d.py"))
    _ktw = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_ktw)
    _carry = (getattr(cfg, "tke_preclosure_coeff_source", None)
              == "carried_previous_step")
    print(f"  TKE bridge: tke_preclosure_coeff_source="
          f"{getattr(cfg, 'tke_preclosure_coeff_source', None)!r} -> "
          f"coefficients {'carried' if _carry else 'NOT carried'}")
    st = bridge_before_state_topo(br._replace(state=br.state), g, before,
                                  periodic_i=True)
    st = _ktw.bridge_tke_from_restart(
        st, _O3("en", R1), br.land_mask,
        restart_avm=(_O3("avm_k", R1) if _carry else None),
        restart_avt=(_O3("avt_k", R1) if _carry else None),
        restart_dissl=(_O3("dissl", R1) if _carry else None))
    _wetc = _np.asarray(br.land_mask) > 0.5
    print(f"  state.tke seeded from NEMO's en: max|d| over wet columns = "
          f"{float(_np.max(_np.abs(_np.asarray(st.tke.data)[_wetc] - _O3('en', R1)[..., 1:][_wetc]))):.3e}")
    st_lf = st
    if a.plant_euler:
        print("EULER PLANT ACTIVE: this run scores the leap-frog step AND an "
              "arm with the before levels dropped (rdt = rn_Dt); every scored "
              "row MUST differ between them, or the gate is not measuring a "
              "leap-frog step at all.")

    # Rule 10: PRINT the timestep the step will actually use, do not infer it.
    euler = st_lf.u_before is None
    rdt = (1.0 if euler else 2.0) * RN_DT
    print(f"\nSTEP: euler_start={euler}  rdt={rdt}  "
          f"(NEMO kt=2: rDt = 2*rn_Dt = {2 * RN_DT})")
    if rdt != 2 * RN_DT:
        print("  ^^ the step is NOT the leap-frog step this record scores; "
              "the kt=1 restart did not carry the before levels")
        return 1
    st = st_lf

    forcing = dm.dino_lat_lon_surface_forcing_arrays(grid, cfg)
    sf_step = (dm.dino_step_surface_forcing(forcing)
               if getattr(cfg, "wind_through_step", False) else None)
    st2, rate = dm.apply_dino_lat_lon_surface_forcing(
        st, forcing, z, cfg, RN_DT, t_seconds=2 * RN_DT, return_rate=True)

    real = gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid
    calls: list = []

    def spy(q, *aa, **kw):
        r = real(q, *aa, **kw)
        out = r[0] if (isinstance(r, tuple) and kw.get("return_bolus")) else r
        calls.append((_np.asarray(q), _np.asarray(out), (q,) + tuple(aa),
                      dict(kw)))
        return r

    gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid = spy
    try:
        with jax.disable_jit():
            after = model.step(st2, dt=RN_DT, surface_forcing=sf_step,
                               external_tracer_rate=rate)
    finally:
        gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid = real

    gmesh = ndm.nemo_dino_mesh()
    wet3 = gmesh.tmask > 0.5
    wet2 = wet3[..., 0]

    O3 = _O3           # the stitched (nlev, nlat, nlon) -> (nlat, nlon, nlev)

    rows: dict[str, float] = {}
    bad = bad_cov
    print("\nSTATE at the end of kt=2 (legoESM vs NEMO's kt=2 restart)")
    print(f"  {'field':14s}{'cells!=':>10s}{'max':>14s}{'rms':>14s}"
          f"{'NEMO step':>14s}")
    uwet = gmesh.umask > 0.5
    vwet = gmesh.vmask > 0.5

    def _lego(state, name):
        """legoESM's field on NEMO's own staggering.

        legoESM carries one REDUNDANT face (the periodic image west of cell 0,
        and the closed southern wall); NEMO's ``un``/``vn`` are the east/north
        faces.  Same inverse as nemo_dino_step1_gate.py:275-278.
        """
        d = _np.asarray(getattr(state, name).data)
        if name == "u":
            return d[:, 1:, :]
        if name == "v":
            return d[1:, :, :]
        return d

    for lego_name, nemo_name, w in (("T", "tn", wet3), ("S", "sn", wet3),
                                    ("eta", "sshn", wet2),
                                    ("u", "un", uwet), ("v", "vn", vwet)):
        obj = _lego(after, lego_name)
        nemo = R2[nemo_name] if w is wet2 else O3(nemo_name, R2)
        prev = R1[nemo_name] if w is wet2 else O3(nemo_name, R1)
        step = float(_np.abs(nemo - prev)[w].max())
        rows[lego_name] = float(
            _np.sqrt(_np.mean((_np.asarray(obj) - nemo)[w] ** 2)))
        bad += _score(lego_name, obj, nemo, w, step)
    print("  ^ NOT ATTRIBUTABLE to any leap-frog statement while the surface "
          "carries above are unbridged; the per-operator row below is.")

    print("\ntra_ldf at kt=2 (legoESM's own tendency vs NEMO's ttrd_ldf)")
    if not calls:
        print("  UNMEASURED: the isoneutral operator was never called")
        bad += 1
    else:
        for tag, key in (("T", "ttrd_ldf"), ("S", "strd_ldf")):
            if key not in R2:
                print(f"  UNMEASURED {key}: the kt=2 restart does not carry "
                      "it (was ln_tra_trd on?)")
                bad += 1
                continue
            # The operator is fed the BEFORE (Kbb) tracer, not the now one
            # (stpmlf.f90:504 passes Kbb=Nbb), so "nearest to T-now" is not
            # "NEMO's tra_ldf call".  Identify by an EXACT wet-masked match
            # against the Kbb field the step carries, and REFUSE when nothing
            # matches -- kt=1's own note records that scoring the wrong call
            # moved a ratio from 0.9967 to 5.53.
            src = _np.asarray(getattr(st2, tag + "_before").data
                              if getattr(st2, tag + "_before", None) is not None
                              else getattr(st2, tag).data)
            hits = [i for i in range(len(calls))
                    if _np.array_equal(calls[i][0][wet3], src[wet3])]
            print(f"  call identification [{tag}]: "
                  + "  ".join(f"call {i}: max|q-Kbb| "
                              f"{_np.abs(calls[i][0] - src)[wet3].max():.3e}"
                              for i in range(len(calls))))
            if len(hits) != 1:
                print(f"  ^^ {len(hits)} captured calls match the Kbb {tag} "
                      "exactly; the scored call is not identified and no row "
                      "below is about NEMO's tra_ldf")
                bad += 1
                continue
            idx = hits[0]
            tend = calls[idx][1]
            nemo = _np.nan_to_num(O3(key, R2))
            den = float(nemo[wet3] @ nemo[wet3])
            ratio = float(tend[wet3] @ nemo[wet3]) / den if den else float("nan")
            res = float(_np.sqrt(_np.mean((tend[wet3] - nemo[wet3]) ** 2)))
            n = int((tend[wet3] != nemo[wet3]).sum())
            print(f"  {tag} tra_ldf [call {idx}]  ratio {ratio:.9f}  "
                  f"res rms {res:.4e}  NEMO rms "
                  f"{float(_np.sqrt(_np.mean(nemo[wet3] ** 2))):.4e}  "
                  f"cells!= {n}/{int(wet3.sum())}  "
                  f"{'AT BAR' if n == 0 else 'DEBT'}")
            if n:
                bad += 1
        kw = calls[0][3]
        print(f"  the dt the operator received: {kw.get('dt')!r}  "
              f"(NEMO rDt at kt=2 = {2 * RN_DT})")
        if kw.get("dt") != 2 * RN_DT and not a.plant_euler:
            print("  ^^ the operator is NOT on NEMO's rDt "
                  "(traldf_iso.f90:829-830)")
            bad += 1

    # The akz census: this is the ONE thing the kt=1 record structurally could
    # not see, so it is printed whether or not it is at the bar.
    kwd = {**calls[0][3], "return_diagnostics": True,
           "return_operand_diagnostics": True}
    r = real(*calls[0][2], **kwd)
    # The DINO card resolves gm_bolus_advection='through_fct', so the captured
    # call carries return_bolus=True and the operator returns a THREE-tuple
    # when the diagnostics are also asked for (gm_redi_latlon_cgrid.py:2787).
    diags = r[-1]
    if not isinstance(diags, dict):
        print(f"\n  akz census FAILED: the operator returned "
              f"{type(r).__name__} of length {len(r)}, whose last member is "
              f"{type(diags).__name__}, not the diagnostics dict")
        bad += 1
    elif "zfw_operands" not in diags:
        print("\n  akz census FAILED: no 'zfw_operands' in the diagnostics -- "
              "that block exists only when msc_stabilize is on "
              "(gm_redi_latlon_cgrid.py:2764), so the stabilising correction "
              "is OFF while NEMO's namelist_cfg:267 turns it on")
        bad += 1
    else:
        akz = _np.asarray(diags["zfw_operands"]["akz"])
        w = wet3[..., :akz.shape[-1]]
        fired = int((akz[w] > 0.0).sum())
        tot = int(w.sum())
        print(f"\n  akz branch census at kt=2: the stabiliser fires on "
              f"{fired}/{tot} wet cells -- "
              + ("the A33 e3w (traldf_iso.f90:831-833) and the implicit half "
                 "of the split are now SCORED by the rows above"
                 if fired else
                 "still UNMEASURED: the branch is empty at kt=2 as well, so "
                 "those two operands remain multiplied by nothing"))

    if a.plant_euler:
        st_eu = st_lf._replace(u_before=None, v_before=None, T_before=None,
                               S_before=None, eta_before=None)
        st2e, ratee = dm.apply_dino_lat_lon_surface_forcing(
            st_eu, forcing, z, cfg, RN_DT, t_seconds=2 * RN_DT,
            return_rate=True)
        with jax.disable_jit():
            after_e = model.step(st2e, dt=RN_DT, surface_forcing=sf_step,
                                 external_tracer_rate=ratee)
        print("\nPLANT (euler arm): every scored row must MOVE")
        moved = 0
        for lego_name, nemo_name, w in (("T", "tn", wet3), ("S", "sn", wet3),
                                        ("eta", "sshn", wet2),
                                        ("u", "un", uwet), ("v", "vn", vwet)):
            nemo = R2[nemo_name] if w is wet2 else O3(nemo_name, R2)
            r = float(_np.sqrt(_np.mean(
                (_lego(after_e, lego_name) - nemo)[w] ** 2)))
            ok = r != rows[lego_name]
            moved += int(ok)
            print(f"  {lego_name:6s} leap-frog rms {rows[lego_name]:.6e}  "
                  f"euler rms {r:.6e}  {'MOVED' if ok else 'DID NOT MOVE'}")
        if moved != len(rows):
            print("  ^^ a row that does not move between a leap-frog and an "
                  "Euler step is not measuring the timestep, and every rDt "
                  "row above rests on it")
            return 1
        print("  every row moved, as required")
        return 0
    print(f"\n{'GATE PASS' if bad == 0 else f'GATE FAIL ({bad} rows)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
