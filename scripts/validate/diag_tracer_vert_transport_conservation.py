"""What the non-conservative tracer operator costs, band by band.

On the hybrid lane the conservative Simmons-Burridge lever is read by the
TEMPERATURE path only (primitive_eq_mpas.py:666-667, ``conservative=
(_vert_scheme == 'sb')``).  The tracer path calls
``vertical_advection_hybrid`` with no scheme argument at all
(primitive_eq_mpas.py:794-795), so every tracer -- including all six
condensate species -- rides the non-conservative upwind advective operator
unconditionally, whatever the deck selects.

The advective form ``-F df/dp`` does not satisfy the discrete product rule; the
conservative ``vertical_advection_hybrid_sb`` telescopes to the (zero) boundary
fluxes.  The conservation statement is about the FLUX FORM
``adv_k - f_k*(mdot_{k+1/2}-mdot_{k-1/2})/dp_k``, NOT about the advection
operator on its own: the operator alone has a non-zero column integral for
BOTH forms, so comparing raw global tendencies is not a conservation test.
Only the flux-form column integral printed at the end is one.

This probe applies BOTH operators to the same state and the same mass flux and
reports, per species, the band-integrated tendency from each and their
difference, then the flux-form global integral as the conservation test.

WHAT IT DOES NOT ESTABLISH.  The flux-form residual is the INSTANTANEOUS
vertical-advection contribution at one state.  It is not the model's total
water non-conservation per day: surface-pressure hyperdiffusion, the time
integration stages, positivity repair and any mass fixer all contribute and are
outside this probe.  Nor can a single snapshot exclude a cause, because altered
vapour transport feeds back on condensation and circulation over days.

NUMBERS ONLY -- no verdict.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

_DAY = 86400.0


def _load_run_amip():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_ra_tvt", root / "scripts" / "run" / "run_amip.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["_ra_tvt"] = m
    spec.loader.exec_module(m)
    return m


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--restart", required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--band", nargs=2, type=float, default=[500.0, 800.0],
                    metavar=("P_LO_HPA", "P_HI_HPA"))
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return ap


def masked_int(rate, dp, area_w, g, mask):
    return float(np.sum(area_w[:, None] * np.where(mask, rate * dp, 0.0)) / g)


def main():
    a = build_arg_parser().parse_args()
    import jax.numpy as jnp
    from legoesm import constants

    ra = _load_run_amip()
    argv = ["--config", a.config] + list(a.extra)
    parser = ra.build_arg_parser()
    from legoesm.driver.run_config_yaml import load_yaml_config
    keys = load_yaml_config(a.config, parser)
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    args = parser.parse_args(argv)
    args = ra._postprocess_args(args, parser, argv)
    ra._apply_spectral_scheme_fallback(args, argv, parser)
    config = ra.build_config_from_args(args)

    from legoesm.driver.model_driver import ModelDriver
    driver = ModelDriver(config)
    driver.setup()
    step0, day0 = driver.load_checkpoint(a.restart)
    print(f"[{a.label}] restored step={step0} day={day0}", flush=True)

    state, model = driver.state, driver.model
    sc = model.sigma_coord
    mesh = model.mesh
    from legoesm.grids.vertical import HybridSigmaPressureCoordinate
    hybrid = isinstance(sc, HybridSigmaPressureCoordinate)
    print(f"[{a.label}] vertical coordinate hybrid={hybrid}; "
          f"deck vert_advection_scheme="
          f"{getattr(model.config, 'vert_advection_scheme', '?')!r}")
    if not hybrid:
        raise SystemExit(f"[{a.label}] this probe targets the HYBRID lane; the "
                         f"sigma lane already passes the scheme to tracers")

    T = jnp.asarray(state.T.data)
    ncol, nlev = T.shape
    area = np.asarray(mesh.areaCell).reshape(ncol)
    A = area / area.sum()
    p_s = jnp.asarray(state.p_s.data)
    p_full = sc.pressure_at_full(p_s).reshape(ncol, nlev)
    p_half = sc.pressure_at_half(p_s).reshape(ncol, nlev + 1)
    dp = np.asarray(p_half[:, 1:] - p_half[:, :-1])
    g = constants.g

    # Mass flux exactly as the core builds it on the hybrid branch
    # (primitive_eq_mpas.py:644-656): flux-form continuity cumsum.
    # dp at edges via the SAME cell-to-edge average the core uses
    # (primitive_eq_mpas.py:491-495), then flux-form div(u*dp) and the shared
    # cumsum (:625, :644-656).
    from legoesm.core.operators_voronoi import (
        divergence_cell_3d, cell_to_edge_avg_3d)
    from legoesm.grids.vertical import compute_mass_flux_from_cumsum
    u_3d = jnp.asarray(state.u.data)
    dp_j = jnp.asarray(dp)
    dp_edge = cell_to_edge_avg_3d(dp_j, mesh)
    div_dp_3d = divergence_cell_3d(u_3d * dp_edge, mesh)
    cumsum_dp = jnp.cumsum(div_dp_3d, axis=-1)
    D_total = cumsum_dp[..., -1:]
    mass_flux = compute_mass_flux_from_cumsum(cumsum_dp, D_total, sc)
    print(f"[{a.label}] mass flux built from flux-form continuity; "
          f"|mdot|max {float(jnp.max(jnp.abs(mass_flux))):.4e} Pa/s, "
          f"top/surface interface values "
          f"{float(jnp.max(jnp.abs(mass_flux[..., 0]))):.2e}/"
          f"{float(jnp.max(jnp.abs(mass_flux[..., -1]))):.2e} (must be 0)")

    from legoesm.grids.vertical import (
        vertical_advection_hybrid, vertical_advection_hybrid_sb)

    lo, hi = a.band[0] * 100.0, a.band[1] * 100.0
    pfn = np.asarray(p_full)
    band = (pfn >= lo) & (pfn < hi)
    allm = np.ones_like(band, dtype=bool)

    print(f"[{a.label}] --- vertical transport of each tracer, "
          f"{a.band[0]:.0f}-{a.band[1]:.0f} hPa.  UNITS: q_* rows are "
          f"kg/m2/day; N_* rows are per-MASS number, so their dp/g integral is "
          f"#/m2/day -- do NOT read the N_* rows as water.")
    print(f"[{a.label}] These are ADVECTION-OPERATOR tendencies, not completed "
          f"band inventory tendencies; the continuity term cancels only in the "
          f"scheme-to-scheme DIFFERENCE column, which is the trustworthy one.")
    print(f"[{a.label}] {'species':8s} {'advective(now)':>15s} "
          f"{'conservative(sb)':>17s} {'difference':>13s} "
          f"{'GLOBAL adv':>13s} {'GLOBAL sb':>13s}")
    names = list(state.tracers.keys()) if state.tracers else []
    cond = [n for n in names if n in ("q_c", "q_r", "q_i", "q_s", "q_g")]
    band_tot = {"adv": 0.0, "sb": 0.0}
    for n in names:
        r = state.tracers[n]
        q = jnp.asarray(r.data if hasattr(r, "data") else r).reshape(ncol, nlev)
        adv = np.asarray(vertical_advection_hybrid(q, mass_flux, p_s, sc))
        sb = np.asarray(vertical_advection_hybrid_sb(q, mass_flux, p_s, sc))
        b_a = masked_int(adv, dp, A, g, band) * _DAY
        b_s = masked_int(sb, dp, A, g, band) * _DAY
        g_a = masked_int(adv, dp, A, g, allm) * _DAY
        g_s = masked_int(sb, dp, A, g, allm) * _DAY
        if n in cond:
            band_tot["adv"] += b_a
            band_tot["sb"] += b_s
        if n.startswith("N_") and abs(g_a) == 0.0 and abs(b_a) == 0.0:
            print(f"[{a.label}] {n:8s} identically zero in this state -- "
                  f"inactive or masked field, reported rather than tabulated")
            continue
        unit = "#/m2/day" if n.startswith("N_") else "kg/m2/day"
        print(f"[{a.label}] {n:8s} {b_a:+15.6e} {b_s:+17.6e} "
              f"{b_a - b_s:+13.6e} {g_a:+13.6e} {g_s:+13.6e}  [{unit}]")

    # TOTAL CONDENSATE, because a single species can carry the opposite sign to
    # the sum and a conclusion drawn from q_c alone will be wrong (GLM review,
    # 2026-09-23: the q_c row says sb removes MORE, the condensate sum says sb
    # ADDS more).
    if cond:
        print(f"[{a.label}] --- band TOTAL CONDENSATE "
              f"({'+'.join(cond)}) [kg/m2/day]")
        print(f"[{a.label}]   advective (in use) {band_tot['adv']:+13.6e}   "
              f"conservative (sb) {band_tot['sb']:+13.6e}   "
              f"difference {band_tot['sb'] - band_tot['adv']:+13.6e}")

    # Conservation test on the total water path, the quantity that must not be
    # created or destroyed by vertical transport.
    water = [n for n in names if n.startswith("q_")]
    if water:
        qt = sum(jnp.asarray(state.tracers[n].data).reshape(ncol, nlev)
                 for n in water)
        adv = np.asarray(vertical_advection_hybrid(qt, mass_flux, p_s, sc))
        sb = np.asarray(vertical_advection_hybrid_sb(qt, mass_flux, p_s, sc))
        # The conservative property is of the FLUX FORM, i.e. of
        #   adv_k - f_k * (mdot_{k+1/2} - mdot_{k-1/2}) / dp_k
        # (vertical.py, vertical_advection_hybrid_sb docstring).  Testing the
        # advection operator ALONE is not a conservation test for either form,
        # because the continuity term that completes the divergence lives
        # elsewhere in the tendency.  Add it here so the test is well posed.
        mf = np.asarray(mass_flux)
        dmdot = mf[..., 1:] - mf[..., :-1]              # (ncol, nlev)
        qtn = np.asarray(qt)
        flux_adv = adv - qtn * dmdot / np.clip(dp, 1e-10, None)
        flux_sb = sb - qtn * dmdot / np.clip(dp, 1e-10, None)
        scale = masked_int(np.abs(adv), dp, A, g, allm) * _DAY
        print(f"[{a.label}] --- total water ({'+'.join(water)}): GLOBAL column "
              f"integral of the FLUX-FORM vertical tendency [kg/m2/day].  A "
              f"conservative operator must give ~0 here; the advection term "
              f"alone is NOT a conservation test for either form.")
        for nm, fl in (("advective (in use now)", flux_adv),
                       ("conservative (sb)", flux_sb)):
            v = masked_int(fl, dp, A, g, allm) * _DAY
            print(f"[{a.label}]   {nm:24s} {v:+13.6e}   "
                  f"= {abs(v) / max(scale, 1e-30):.3e} of gross transport")


if __name__ == "__main__":
    main()
