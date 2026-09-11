#!/usr/bin/env python
"""Rule 12 for the ``traldf_iso`` A33 ``e3w`` fix: which cards does it move?

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
