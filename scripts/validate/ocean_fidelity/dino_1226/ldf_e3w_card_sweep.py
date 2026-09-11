#!/usr/bin/env python
"""Rule 12 for the two ``traldf_iso`` transcription fixes: which cards move?

SECTION A -- the A33 ``e3w`` divisor (PR #1728 round 3).
SECTION B -- the stretch's TIME LEVEL (round 4): every ``e3``/``r3`` operand of
``traldf_iso`` is indexed ``Kmm`` (``traldf_iso.f90:231-232``, ``:284``,
``:292``, ``:305``, ``:823``, ``:828``), and the shared step now hands the
operator the step-entry height instead of the post-barotropic one.


The fix swaps the A33 vertical-flux divisor from the interface midpoint
``0.5*(e3t(k-1)+e3t(k))`` to NEMO's ``e3w_0(k)*(1+r3t)`` with
``e3w_0(k) = gdept_0(k) - gdept_0(k-1)`` (``traldf_iso.f90:285``, ``:831-833``,
``domzgr_substitute.h90:131``), resolved by ``nemo_iso_a33_e3w`` and shared by
BOTH halves of the explicit/implicit A33 split.

This enumerates every DINO recipe and every ocean recipe in the catalog and
reports, per card:

  reaches?   does it run the isoneutral operator whose A33 block changed
  msc        ``ln_traldf_msc`` -- with msc=F the explicit A33 coefficient is
             identically zero and the divisor cannot matter
  e3w moves  max relative change of the divisor this card would see
  raises?    whether ``nemo_e3w0_reference`` now REFUSES a coordinate that
             previously got the midpoint silently (a new hard error in a
             previously-working path is a regression, and it is reported here
             rather than discovered in a run)

A card with ``reaches=yes`` and ``e3w moves > 0`` has its trajectory changed
and must be registered.
"""
import sys

import numpy as np


def main() -> int:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    import jax.numpy as jnp
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        nemo_iso_a33_e3w)

    print(f"{'card':30s}{'scheme':14s}{'msc':>5s}{'reaches':>9s}"
          f"{'max rel d(e3w)':>16s}  note")
    moved, failed = [], []
    for name in sorted(dm.DINO_RECIPES):
        try:
            cfg = dm.dino_config_for_recipe(name)
            if name.startswith("nemo_dino"):
                cfg = dm.nemo_faithful_dino_config(base=cfg)
            grid = dm.dino_lat_lon_grid(cfg)
            z = dm.dino_lat_lon_vertical(grid, cfg)
        except Exception as exc:                       # noqa: BLE001
            print(f"{name:30s}{'?':14s}{'?':>5s}{'BUILD FAIL':>9s}"
                  f"{'-':>16s}  {type(exc).__name__}: {exc}")
            failed.append(name)
            continue
        scheme = getattr(cfg, "gm_redi_slope_scheme", "?")
        msc = scheme == "nemo_iso_lap"     # the card's msc_stabilize=True arm
        reaches = scheme == "nemo_iso_lap"
        # The card's OWN horizontal shape: nemo_e3w_0 is a 3-D mesh field,
        # so a stand-in shape would only measure a broadcast error.
        _ny, _nx = np.asarray(grid.lat).size, np.asarray(grid.lon).size
        e3t = (jnp.asarray(z.dz_ref)[None, None, :]
               * jnp.ones((_ny, _nx, 1), dtype=jnp.float64))
        jac = jnp.ones((_ny, _nx), dtype=jnp.float64)
        note = ""
        try:
            new = np.asarray(nemo_iso_a33_e3w(z, e3t, jac, jnp.float64))
        except Exception as exc:                       # noqa: BLE001
            print(f"{name:30s}{scheme:14s}{str(msc):>5s}{str(reaches):>9s}"
                  f"{'RAISES':>16s}  {type(exc).__name__}")
            failed.append(name)
            continue
        e3t_n = np.asarray(e3t)
        old = 0.5 * (np.roll(e3t_n, 1, 2) + e3t_n)
        old[..., 0] = e3t_n[..., 0]
        rel = np.abs(new[..., 1:] - old[..., 1:]) / np.maximum(
            np.abs(new[..., 1:]), 1e-300)
        mx = float(rel.max())
        if reaches and mx > 0.0:
            moved.append(name)
            note = "REGISTERED: trajectory moves"
        elif not reaches:
            note = "does not run the changed operator"
        elif mx == 0.0:
            note = "byte-unchanged (midpoint IS e3w_0 here)"
        print(f"{name:30s}{scheme:14s}{str(msc):>5s}{str(reaches):>9s}"
              f"{mx:16.4e}  {note}")

    # --- the NEMO test-case cards (LOCK_EXCHANGE zco, OVERFLOW zps) -------
    from legoesm.ocean.fidelity import nemo_testcase_recipe as ntc
    for case in ("LOCK_EXCHANGE-zco", "OVERFLOW-zps"):
        try:
            card = ntc.build_nemo_testcase_card(case)
        except Exception as exc:                           # noqa: BLE001
            print(f"{case:30s}{'?':14s}{'?':>5s}{'BUILD FAIL':>9s}"
                  f"{'-':>16s}  {type(exc).__name__}: {exc}")
            failed.append(case)
            continue
        gmc = getattr(getattr(card, "model_config", None), "gm_redi", None)
        sch = getattr(gmc, "slope_scheme", "n/a")
        m = bool(getattr(gmc, "msc_stabilize", False))
        zc = card.recipe.z_coord if hasattr(card, "recipe") else None
        try:
            _n = np.asarray(nemo_iso_a33_e3w(
                zc, jnp.asarray(np.broadcast_to(
                    np.asarray(zc.dz_ref), (2, 2, zc.n_levels)).copy()),
                jnp.ones((2, 2), dtype=jnp.float64), jnp.float64))
            _e = np.broadcast_to(np.asarray(zc.dz_ref),
                                 (2, 2, zc.n_levels)).copy()
            _o = 0.5 * (np.roll(_e, 1, 2) + _e)
            _o[..., 0] = _e[..., 0]
            gap = float(np.abs(_n[..., 1:] - _o[..., 1:]).max())
            tag = f"{gap:16.4e}"
        except Exception as exc:                           # noqa: BLE001
            tag, gap = f"{'RAISES':>16s}", -1.0
            failed.append(case)
        note = ("does not run the changed block (ln_traldf_msc=F)"
                if not (m and sch == "nemo_iso_lap")
                else "REGISTERED: trajectory moves")
        print(f"{case:30s}{str(sch):14s}{str(m):>5s}"
              f"{str(m and sch == 'nemo_iso_lap'):>9s}{tag}  {note}")
        if m and sch == "nemo_iso_lap" and gap > 0.0:
            moved.append(case)

    # --- the enumeration itself, grep-backed rather than asserted ---------
    import subprocess
    print("\nEvery place ln_traldf_msc (GMRediConfig.msc_stabilize) is "
          "turned ON in production code -- the ONLY cards that execute the "
          "changed A33 block at all:")
    out = subprocess.run(
        ["grep", "-rn", "--include=*.py", "--include=*.yaml",
         "msc_stabilize", "packages/", "scripts/"],
        capture_output=True, text=True).stdout
    for ln in out.splitlines():
        if "=True" in ln.replace(" ", "") or ": True" in ln:
            print("  " + ln.strip())

    # =================================================================
    # SECTION B -- the stretch's TIME LEVEL
    # =================================================================
    # The Kmm change is scoped to the ``nemo_iso_lap`` branch of
    # gm_redi_tracer_tendency_latlon, and BOTH model call sites pass it
    # unconditionally -- so "reaches" is exactly "runs nemo_iso_lap", and the
    # number of operands that move depends on the card's flux-face mode:
    #   nemo_qco_live   the u/v faces were ALREADY on the Kmm height, so the
    #                   four VOLUME operands move (e3t x2, A33 e3w, ze3w_2)
    #   tpoint_jacobian e3u_flux = e3v_flux = e3t, so SIX operands move
    print("\nSECTION B -- the stretch's TIME LEVEL (Kmm), per card")
    print(f"{'card':30s}{'scheme':14s}{'flux faces':18s}"
          f"{'reaches':>9s}{'operands':>10s}  note")
    moved_b = []
    for name in sorted(dm.DINO_RECIPES):
        try:
            cfg = dm.dino_config_for_recipe(name)
            if name.startswith("nemo_dino"):
                cfg = dm.nemo_faithful_dino_config(base=cfg)
        except Exception as exc:                       # noqa: BLE001
            print(f"{name:30s}{'?':14s}{'?':18s}{'BUILD FAIL':>9s}"
                  f"{'-':>10s}  {type(exc).__name__}")
            failed.append(name)
            continue
        # Rule 10: the card's RESOLVED GMRediConfig, built the way the driver
        # builds it -- not the ExperimentConfig's string and not a getattr
        # default.  An earlier draft of this section read the defaults and
        # reported "tpoint_jacobian" for a card that selects nemo_qco_live.
        try:
            _gg = dm.dino_lat_lon_grid(cfg)
            _mc, _ = dm.dino_lat_lon_model_config(_gg, cfg)
            gmc = _mc.gm_redi
        except Exception as exc:                       # noqa: BLE001
            print(f"{name:30s}{'?':14s}{'?':18s}{'CFG FAIL':>9s}"
                  f"{'-':>10s}  {type(exc).__name__}: {exc}")
            failed.append(name)
            continue
        if gmc is None:
            print(f"{name:30s}{'(no gm_redi)':14s}{'-':18s}{'False':>9s}"
                  f"{0:>10d}  card runs no GM/Redi at all")
            continue
        sch = gmc.slope_scheme
        ffm = gmc.redi_flux_face_thickness_evaluation
        reaches = (sch == "nemo_iso_lap")
        nop = (0 if not reaches else (4 if ffm == "nemo_qco_live" else 6))
        note = ("does not run the changed operator" if not reaches
                else "REGISTERED: trajectory moves")
        if reaches:
            moved_b.append(name)
        print(f"{name:30s}{str(sch):14s}{str(ffm):18s}{str(reaches):>9s}"
              f"{nop:>10d}  {note}")
    for case in ("LOCK_EXCHANGE-zco", "OVERFLOW-zps"):
        try:
            card = ntc.build_nemo_testcase_card(case)
        except Exception:                                  # noqa: BLE001
            continue
        gmc = getattr(getattr(card, "model_config", None), "gm_redi", None)
        sch = "n/a" if gmc is None else gmc.slope_scheme
        ffm = ("-" if gmc is None
               else gmc.redi_flux_face_thickness_evaluation)
        reaches = (sch == "nemo_iso_lap")
        nop = (0 if not reaches else (4 if ffm == "nemo_qco_live" else 6))
        if reaches:
            moved_b.append(case)
        print(f"{case:30s}{str(sch):14s}{str(ffm):18s}{str(reaches):>9s}"
              f"{nop:>10d}  "
              f"{'REGISTERED: trajectory moves' if reaches else 'does not run the changed operator'}")
    print("\nEvery place slope_scheme='nemo_iso_lap' is selected in "
          "production code -- the ONLY cards Section B can reach:")
    out_b = subprocess.run(
        ["grep", "-rn", "--include=*.py", "--include=*.yaml",
         "nemo_iso_lap", "packages/", "scripts/"],
        capture_output=True, text=True).stdout
    for ln in out_b.splitlines():
        if "slope_scheme" in ln:
            print("  " + ln.strip())

    # NON-VACUITY for Section B, functional rather than structural: the new
    # argument must be INERT when it names the height the operator already
    # had, and LOAD-BEARING when it names a different one.  A sweep that
    # cannot tell those apart says nothing about which cards move.
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        gm_redi_tracer_tendency_latlon as _gmt)
    _c = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    _g = dm.dino_lat_lon_grid(_c)
    _mcb, _ = dm.dino_lat_lon_model_config(_g, _c)
    _z = dm.dino_lat_lon_vertical(_g, _c)
    _st = dm.dino_lat_lon_state(_g, _z, _c)
    _kw = dict(eos=_mcb.eos, eos_linear=_mcb.eos_linear,
               mask=_st.land_mask.data, u_mask=_st.u_mask.data,
               v_mask=_st.v_mask.data, rho_0=_mcb.constants.rho_0,
               g=_mcb.constants.g, omega=_mcb.omega, dt=2700.0,
               eos_depth=getattr(_mcb, "eos_depth", "insitu"),
               native_bolus_slope_eta=_st.eta.data)
    _eta = _st.eta.data
    _args = (_st.T.data, _st.S.data, _eta, _st.H_bathy.data, _g, _z,
             _mcb.gm_redi)
    _base = np.asarray(_gmt(*_args, **_kw)[0])
    _same = np.asarray(_gmt(*_args, redi_kmm_eta=_eta, **_kw)[0])
    _other = np.asarray(_gmt(*_args, redi_kmm_eta=_eta + 1.0, **_kw)[0])
    _d_same = float(np.max(np.abs(_same - _base)))
    _d_other = float(np.max(np.abs(_other - _base)))
    print(f"\nSECTION B non-vacuity: redi_kmm_eta=eta is inert "
          f"(max|diff| = {_d_same:.3e}, bar 0.0); redi_kmm_eta=eta+1 m moves "
          f"the tendency (max|diff| = {_d_other:.3e}, must be > 0)")
    if _d_same != 0.0:
        print("  ^^ the new argument is NOT inert when it names the height "
              "the operator already had; every 'does not move' row above is "
              "unsafe")
        failed.append("section-B-inertness")
    if _d_other <= 0.0:
        print("  ^^ the new argument does NOT reach the operator; every "
              "'REGISTERED' row above is unsupported")
        failed.append("section-B-liveness")
    print(f"\nSection B: {len(moved_b)} card(s) move: {moved_b}")

    print(f"\n{len(moved)} card(s) move: {moved}")
    if failed:
        print(f"{len(failed)} card(s) could not be scored: {failed}")
    # A sweep that can never report a move is not a sweep.  Plant: hand the
    # resolver a coordinate with the mesh reference switched off and require
    # the reported move to collapse to zero.
    from legoesm.ocean.experiments import dino as _dm
    _c = _dm.nemo_faithful_dino_config(
        base=_dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    _g = _dm.dino_lat_lon_grid(_c)
    _z = _dm.dino_lat_lon_vertical(_g, _c)._replace(
        nemo_e3w_mesh_reference=False, t_depth_ref=None)
    _ny, _nx = np.asarray(_g.lat).size, np.asarray(_g.lon).size
    _e3t = (jnp.asarray(_z.dz_ref)[None, None, :]
            * jnp.ones((_ny, _nx, 1), dtype=jnp.float64))
    _new = np.asarray(nemo_iso_a33_e3w(_z, _e3t, jnp.ones((_ny, _nx),
                                                          jnp.float64),
                                       jnp.float64))
    _e = np.asarray(_e3t)
    _old = 0.5 * (np.roll(_e, 1, 2) + _e)
    _old[..., 0] = _e[..., 0]
    _pm = float(np.abs(_new - _old).max())
    print(f"PLANT: with the mesh reference switched off the resolver returns "
          f"the midpoint, max|new - old| = {_pm:.3e} (bar 0.0)")
    if _pm != 0.0:
        print("  ^^ the midpoint arm is not the midpoint; the sweep's zero "
              "rows prove nothing")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
