"""Dump one two-leaf multilayer land step on fixed inputs, for a bit-for-bit
comparison of two checkouts (run once with each on PYTHONPATH, then compare).

Columns: vegetated (LAI 3, 0.5) and bare/glacier, dry/wet soil, snow feedback
on and off.  Every output leaf of the state, the surface response and the
diagnostics is written in float64.

Run: PYTHONPATH=<checkout paths> JAX_ENABLE_X64=1 python \
        scripts/validate/land_step_golden_dump.py --out a.npz
     python scripts/validate/land_step_golden_dump.py --compare a.npz b.npz
"""
from __future__ import annotations

import argparse

import numpy as np


def dump(out: str) -> None:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.boundary_data.gap_fill import bare_canopy_params
    from legoesm.land.multilayer_land import (
        MultiLayerLandConfig, init_multilayer_land_state,
        step_multilayer_land_with_diagnostics)
    from legoesm.land.soil_albedo import SOIL_COLOR_ALBEDO
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig

    n = 3
    dv, dn, sv, sn = SOIL_COLOR_ALBEDO[4]
    lp = bare_canopy_params(n)._replace(
        LAI=jnp.asarray([3.0, 0.5, 0.0]), CI=jnp.asarray([0.8, 0.9, 1.0]),
        ALB_VIS=jnp.full(n, 0.1), ALB_NIR=jnp.full(n, 0.2),
        ALB_VIS_DRY=jnp.asarray([dv, dv, 0.8]), ALB_VIS_SAT=jnp.asarray([sv, sv, 0.8]),
        ALB_NIR_DRY=jnp.asarray([dn, dn, 0.6]), ALB_NIR_SAT=jnp.asarray([sn, sn, 0.6]))
    f = AtmToSurface(
        sw_down=jnp.full(n, 300.0), lw_down=jnp.full(n, 350.0),
        precip_total=jnp.full(n, 1e-5), precip_snow=jnp.zeros(n),
        T_lowest=jnp.full(n, 295.0), q_lowest=jnp.full(n, 0.008),
        u_lowest=jnp.full(n, 5.0), v_lowest=jnp.full(n, 2.0),
        p_lowest=jnp.full(n, 95000.0), p_surface=jnp.full(n, 100000.0),
        rho_lowest=jnp.full(n, 1.15), cos_zenith=jnp.full(n, 0.5),
        co2_ppmv=jnp.full(n, 400.0), has_radiation=jnp.ones(n),
        has_precipitation=jnp.ones(n))
    res = {}
    for fb in (True, False):
        cfg = MultiLayerLandConfig(snow_albedo_feedback=fb,
                                   surface_scheme=TwoLeafCanopyConfig())
        st = init_multilayer_land_state(n, cfg, T_init=293.0)
        st = st._replace(theta_soil=st.theta_soil.at[:, 0].set(
            jnp.asarray([0.05, 0.30, 0.05])))
        step = jax.jit(lambda s, cfg=cfg: step_multilayer_land_with_diagnostics(
            s, f, cfg, 1.0, 1800.0, lat=jnp.full(n, 0.4), land_params=lp))
        for k in range(3):
            st, resp, _, sfc = step(st)
        for tag, tree in (("state", st), ("resp", resp), ("sfc", sfc)):
            for i, leaf in enumerate(jax.tree.leaves(tree)):
                res[f"fb{int(fb)}_{tag}_{i}"] = np.asarray(leaf, dtype=np.float64)
    np.savez(out, **res)
    print(f"# {len(res)} arrays -> {out}")


def compare(a: str, b: str) -> int:
    A, B = np.load(a), np.load(b)
    if sorted(A.files) != sorted(B.files):
        print("# DIFFERENT array sets"); return 1
    bad = [k for k in A.files if not np.array_equal(A[k], B[k], equal_nan=True)]
    print(f"# {len(A.files)} arrays compared, {len(bad)} differ" + (f": {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out")
    p.add_argument("--compare", nargs=2)
    a = p.parse_args(argv)
    if a.compare:
        return compare(*a.compare)
    dump(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
