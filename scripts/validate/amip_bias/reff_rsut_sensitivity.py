"""#1521's own cheapest discriminator: rsut sensitivity to droplet r_eff.

The issue attributes most of the +36.6 W/m2 rsut excess to a too-small droplet
effective radius (7.84 um diagnosed vs 11-14 observed).  The triage's named
next step, run here: sweep the RADIATION KERNEL over r_eff on the model's OWN
cloud fields -- everything except the radius handed to the kernel byte-
identical -- and read d(rsut)/d(r_eff).

Columns come from a saved production MPAS checkpoint (state the model actually
occupies, not an invented profile).  Cloud water paths, cloud fraction and ice
radius are computed ONCE with the production cloud config and held fixed; only
``cloud_r_eff_liq`` changes between arms:

    arm PSD(ckpt) : the PSD radius computed from the CHECKPOINT's N_c.  A
                 baseline for the deltas, NOT the radius production's kernel
                 saw: the run sets aerosol_ccn, which replaces radiation's
                 droplet number from the external aerosol (codex review), so
                 no claim is made that this row equals the run's 7.84 um.
    arm in-cloud : the PSD radius with the condensate reconstructed IN-CLOUD
                 (q_c/cf), which is the pairing an AMBIENT droplet number
                 requires.  Production sets aerosol_ccn, whose AOD-derived
                 number is in-cloud but takes the grid-mean branch, so this
                 arm measures the size of that mismatch on the model's own
                 cloud-fraction distribution rather than from a global mean.
                 Reviewed by GLM before it was written; it also warned that
                 a global-mean cube root OVERSTATES the effect, because
                 x^(1/3) is concave and real cloud fractions are bimodal --
                 which is exactly why this is measured per column here.
    arms 8/10/12/14 um : uniform overrides -- THE measurement

READING (pre-registered): the issue needs roughly -15 to -20 W/m2 available
from moving 7.84 -> 11-14 um.  d(rsut)/d(r_eff) of order -3 to -5 W/m2/um over
that range CONFIRMS the radius attribution is the right SIZE; a sensitivity
well under -1.5 W/m2/um means the radius cannot buy the residual and the
attribution needs rework.

FRAME, stated because it bounds the claim (codex review): this solve is a
SINGLE HOMOGENEOUS COLUMN per cell -- production additionally applies
two_region inhomogeneity and max_random McICA subcolumns, both of which THIN
the effective optical response.  So the sensitivity here is the
PLANE-PARALLEL-side value, an UPPER bound on production's; GLM's calibration
against the run's own LWP-halving test (-23 W/m2 for -0.69 in ln tau) gives
the matching lower anchor.  The absolute rsut is also not production's (default
gases/aerosols, fixed 0.06 ocean albedo, 8-point local-time quadrature).
Only the DIFFERENCES between arms are the measurement.  Numbers only.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="MPAS checkpoint_day_NNNN.npz")
    p.add_argument("--mesh-level", type=int, default=5)
    p.add_argument("--radii-um", type=float, nargs="+",
                   default=[8.0, 10.0, 12.0, 14.0])
    p.add_argument("--n-times", type=int, default=8,
                   help="local-time quadrature points over the diurnal cycle")
    p.add_argument("--cover-schemes", nargs="*", default=[],
                   choices=("sundqvist", "xu_randall"),
                   help="also re-solve the radiation with the cloud COVER "
                        "closure swapped, everything else held fixed. Cover "
                        "changes the water paths as well as the fraction, so "
                        "this reports rsut AND rlut per closure.")
    args = p.parse_args(argv)

    import jax
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        _CLOUD_R_EFF_MAX_M, _INHOM_CF_FLOOR, compute_cloud_properties)
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.atmosphere.physics._shared import compute_rho
    from legoesm.grids.voronoi import create_voronoi_mesh

    d = np.load(args.checkpoint, allow_pickle=True)
    T = jnp.asarray(d["T"])                       # (ncol, nlev)
    p_s = jnp.asarray(d["p_s"])
    q_v = jnp.asarray(d["trc_q_v"])
    q_c = jnp.asarray(d["trc_q_c"])
    q_i = jnp.asarray(d["trc_q_i"])
    n_c = jnp.asarray(d["trc_N_c"])
    n_i = jnp.asarray(d["trc_N_i"])
    # meta_vgrid row 1 is the HALF-level sigma ladder (nlev+1 entries,
    # 0 -> 1); full levels are its midpoints (codex review — row [1][:-1]
    # silently shifted every pressure by half a layer).  Row 0 is the hybrid
    # A-coefficient ladder, all zero on this sigma configuration; assert that
    # so a hybrid checkpoint cannot be mis-read here.
    _vg = np.asarray(d["meta_vgrid"], dtype=np.float64)
    assert _vg.shape[0] == 2 and float(np.abs(_vg[0]).max()) == 0.0, (
        "meta_vgrid row 0 is not all-zero: hybrid checkpoint, this probe "
        "only handles the sigma ladder")
    sig_half_np = _vg[1]
    sig_full = jnp.asarray(0.5 * (sig_half_np[:-1] + sig_half_np[1:]))
    ncol, nlev = T.shape
    day = float(d["day"]) if "day" in d else 12.0

    mesh = create_voronoi_mesh(args.mesh_level)
    lat = np.asarray(mesh.latCell, dtype=np.float64)          # radians
    assert lat.shape[0] == ncol, (lat.shape, ncol)

    p_full = sig_full[None, :] * p_s[:, None]
    p_half = jnp.asarray(np.maximum(sig_half_np, 1e-5))[None, :] * p_s[:, None]
    dp = p_half[:, 1:] - p_half[:, :-1]
    T_sfc = T[:, -1]

    # Checkpoint number convention: per-mass [1/kg]; cloud_fraction wants
    # per-volume [1/m^3] (the driver bridges via N * rho).
    conv = str(d["number_convention"]) if "number_convention" in d else "per_mass"
    rho = compute_rho(T, p_full, q_v)
    n_c_vol = n_c * rho if conv == "per_mass" else n_c
    n_i_vol = n_i * rho if conv == "per_mass" else n_i

    # Production cloud config (config/amip/amip_production.yaml).
    ccfg = CloudConfig(scheme="sundqvist", rh_crit=0.85, q_c_diagnostic=5.0e-6,
                       saturation_scheme="mixed_phase")
    cp = compute_cloud_properties(
        T=T, p_full=p_full, q_v=q_v, dp=dp, config=ccfg,
        q_cloud=q_c, q_ice=q_i, n_cloud=n_c_vol, n_ice=n_i_vol)
    kw = cp.to_rrtmg_kwargs()
    r_psd = kw["cloud_r_eff_liq"]

    # IN-CLOUD PAIRING ARM (#1521, 2026-09-10).  The PSD radius goes as
    # (q_c / N_c)^(1/3), so the ratio is meaningful only if BOTH are on the
    # same footing.  compute_cloud_properties reconstructs in-cloud
    # condensate ONLY when the droplet number is dead (``n_cloud <= 1.0``,
    # the specified-constant fallback).  Production instead sets
    # ``aerosol_ccn``, whose AOD-derived number is ~1e8 and is an AMBIENT --
    # i.e. IN-CLOUD -- concentration, so it takes the other branch and is
    # paired with GRID-MEAN condensate.  Reviewed by GLM before this arm was
    # written: "Using ambient CCN directly as droplet number therefore
    # carries the in-cloud convention... The branch predicate is wrong:
    # 'prognostic vs. specified' is a proxy for the real question, is the
    # source number cf-diluted?"
    #
    # Dividing q_c by cf multiplies the radius by cf^(-1/3) EXACTLY (r_eff is
    # a pure cube root in q_c at fixed N_c), so the corrected field needs no
    # second cloud-property solve -- but it does need the SAME clamp the PSD
    # obeys, or the arm manufactures radii the model could never emit.
    cf_psd = jnp.clip(cp.cloud_fraction, _INHOM_CF_FLOOR, 1.0)
    r_incloud = jnp.minimum(r_psd * cf_psd ** (-1.0 / 3.0),
                            _CLOUD_R_EFF_MAX_M)

    # Diurnal quadrature: Jan declination, hour angles at n_times local times.
    doy = 1.0 + day
    decl = -23.44 * np.cos(2 * np.pi * (doy + 10.0) / 365.0) * np.pi / 180.0
    hours = np.arange(args.n_times) * 24.0 / args.n_times + 24.0 / (2 * args.n_times)
    cosz_t = []
    for h in hours:
        H = (h - 12.0) * np.pi / 12.0
        mu = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(H)
        cosz_t.append(np.clip(mu, 0.0, 1.0))

    solver = RRTMGP.from_legoesm_config(RRTMGPConfig(
        gpoint_batch_size=16, gpoint_checkpoint=False, include_clouds=True))
    area_w = jnp.asarray(np.cos(lat) * 0.0 + 1.0)   # SCVT ~equal-area
    area_w = area_w / jnp.sum(area_w)

    def _toa(r_eff_liq, kwx=None):
        """Diurnally-averaged TOA (rsut, rlut) for one set of cloud fields."""
        kwx = kw if kwx is None else kwx
        sw = 0.0
        lw = 0.0
        for mu in cosz_t:
            mu_j = jnp.asarray(np.maximum(mu, 0.0))
            out = solver.solve_columns(
                T=T, p_full=p_full, p_half=p_half, sfc_temperature=T_sfc,
                q_v=q_v, cos_zenith=jnp.maximum(mu_j, 1e-4),
                sfc_albedo=0.06, sfc_emissivity=0.97,
                cloud_path_liq=kwx["cloud_path_liq"],
                cloud_path_ice=kwx["cloud_path_ice"],
                cloud_r_eff_liq=r_eff_liq,
                cloud_r_eff_ice=kwx.get("cloud_r_eff_ice"))
            # Night columns: SW up is 0 when mu ~ 0; the 1e-4 floor keeps the
            # solver defined and contributes ~0 flux.
            day_mask = jnp.asarray(mu > 0.0)
            sw = sw + jnp.sum(area_w * jnp.where(
                day_mask, out.sw_flux_up[:, 0], 0.0))
            lw = lw + jnp.sum(area_w * out.lw_flux_up[:, 0])
        n = len(cosz_t)
        return float(sw / n), float(lw / n)

    def rsut_mean(r_eff_liq):
        return _toa(r_eff_liq)[0]

    print(f"{args.checkpoint}: {ncol} columns, {nlev} levels, day {day:g}, "
          f"{args.n_times}-point diurnal quadrature")
    r_mean = float(jnp.mean(jnp.where(kw["cloud_path_liq"] > 1e-4,
                                      r_psd, jnp.nan)) )
    print(f"model's own PSD r_eff, cloudy-column mean: "
          f"{np.nanmean(np.asarray(jnp.where(kw['cloud_path_liq'] > 1e-4, r_psd, jnp.nan))) * 1e6:.2f} um")
    _cloudy = np.asarray(kw["cloud_path_liq"]) > 1e-4
    _cf_np = np.asarray(cf_psd)
    _r_a = np.asarray(r_psd)
    _r_b = np.asarray(r_incloud)
    print(f"cloud fraction over cloudy points: mean {_cf_np[_cloudy].mean():.3f}, "
          f"median {np.median(_cf_np[_cloudy]):.3f}, "
          f"5th pct {np.percentile(_cf_np[_cloudy], 5):.3f}")
    print(f"in-cloud pairing would move r_eff "
          f"{_r_a[_cloudy].mean() * 1e6:.2f} -> {_r_b[_cloudy].mean() * 1e6:.2f} um "
          f"(cloudy-point mean; ratio {(_r_b[_cloudy] / _r_a[_cloudy]).mean():.3f}, "
          f"cap binds on {100.0 * float((_r_b >= _CLOUD_R_EFF_MAX_M)[_cloudy].mean()):.2f}% "
          f"of cloudy points)")
    # CONTROL: the correction must be the IDENTITY where the layer is
    # overcast.  If this prints anything but ~1.000 the arm is measuring its
    # own arithmetic rather than the pairing.
    _oc = _cloudy & (_cf_np > 0.999)
    if _oc.any():
        print(f"  control, overcast points (cf > 0.999): ratio "
              f"{(_r_b[_oc] / _r_a[_oc]).mean():.6f} (must be 1.000000)")
    else:
        print("  control SKIPPED: no overcast cloudy points in this checkpoint")
    print()
    print(f"{'arm':>10s} {'rsut [W/m2]':>12s} {'delta vs PSD':>13s}")
    base = rsut_mean(r_psd)
    print(f"{'PSD(ckpt)':>10s} {base:12.3f} {0.0:13.3f}")
    prev_r, prev_v = None, None
    r_ic = rsut_mean(r_incloud)
    print(f"{'in-cloud':>10s} {r_ic:12.2f} {r_ic - base:13.2f}"
          f"   <- pairing correction, NOT a uniform radius override")
    for r_um in args.radii_um:
        v = rsut_mean(jnp.full_like(r_psd, r_um * 1e-6))
        print(f"{r_um:8.1f}um {v:12.3f} {v - base:13.3f}")
        if prev_r is not None:
            print(f"{'':>10s}   d(rsut)/d(r_eff) "
                  f"{(v - prev_v) / (r_um - prev_r):+.3f} W/m2/um "
                  f"over {prev_r:g}-{r_um:g} um")
        prev_r, prev_v = r_um, v

    # COVER-CLOSURE ARMS.  Swapping the closure moves the cloud FRACTION and,
    # through the radiative condensate floor and the in-cloud scaling, the
    # water PATHS too -- so each arm needs its own compute_cloud_properties
    # solve and its own PSD radius, not just a different fraction.  Both
    # numbers are reported: a closure that trades cover for optical depth can
    # brighten in the shortwave and still change the longwave more.
    if args.cover_schemes:
        print()
        print(f"{'cover':>12s} {'rsut [W/m2]':>12s} {'rlut [W/m2]':>12s} "
              f"{'cf mean':>9s} {'LWP':>8s} {'IWP':>8s}")
        for sch in args.cover_schemes:
            cfg_s = ccfg._replace(scheme=sch)
            cp_s = compute_cloud_properties(
                T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg_s,
                q_cloud=q_c, q_ice=q_i, n_cloud=n_c_vol, n_ice=n_i_vol)
            kw_s = cp_s.to_rrtmg_kwargs()
            sw_s, lw_s = _toa(kw_s["cloud_r_eff_liq"], kw_s)
            _cf_s = float(np.mean(np.asarray(cp_s.cloud_fraction)))
            _lwp = float(np.mean(np.sum(np.asarray(kw_s["cloud_path_liq"]), axis=1)))
            _iwp = float(np.mean(np.sum(np.asarray(kw_s["cloud_path_ice"]), axis=1)))
            print(f"{sch:>12s} {sw_s:12.3f} {lw_s:12.3f} {_cf_s:9.4f} "
                  f"{_lwp * 1e3:8.2f} {_iwp * 1e3:8.2f}")
        print("  LWP/IWP are grid-mean column paths [g/m2] as handed to the "
              "solver; cf mean is over all layers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
