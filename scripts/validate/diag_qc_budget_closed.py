"""Closed cloud-water budget over a pressure band, both arms, one state.

Every source and sink of ``q_c`` that the scheme applies, named, area- and
mass-weighted over a pressure band, with the residual against the scheme's own
``dq_c_dt`` printed.  SCOPE, because the residual is easy to over-read: the
closure proves the published terms reproduce the expression that assembles
``dq_c_dt``, i.e. that no IMPLEMENTED term was left out of the publication.  It
CANNOT detect a process missing from both -- MG2, for instance, sediments cloud
droplets (micro_mg2_0.F90:2338) and this scheme's cloud-water tendency has no
sedimentation term at all.  Nor does one call account for transport, the other
parameterizations, or later state corrections.  The terms are the APPLIED
(post-donor-clamp) ones published by ``MorrisonConfig.publish_qc_budget``; a
re-derivation outside the scheme would report PRE-clamp rates and could not
close (codex review, 2026-09-23).

Also measures the three named departures from the MG2 oracle, because each is a
candidate for the CAM6 arm's cloud-water deficit and each is cheap to test on
the same state:

1. MG2 caps in-cloud cloud water at 5e-3 kg/kg (micro_mg2_0.F90:1226); we do
   not.  NOTE the divisor below is the MICROPHYSICS in-cloud fraction, which is
   1.0 when the sub-grid closure is off -- it is not the diagnosed cloud cover,
   and a null result here says nothing about the cap after a faithful in-cloud
   coupling.  Where the cap would bite, our KK2000 rates (autoconversion ~ q_c^2.47)
   are evaluated at an in-cloud water content MG2 would never feed them.
2. MG2 scales vapour deposition by the liquid-lifetime fraction its limiter
   produced (:1588-1592); our clamp does not couple to deposition.
3. Our sedimentation acts on PRE-source pools (morrison.py:29-36); MG2
   sediments the POST-source hydrometeors (:2204-2211).

READ THE ABSOLUTE COLUMN, NOT THE FRACTION.  A fraction of the reservoir is
normalised by the very quantity under investigation, so an arm with an
anomalously small reservoir shows large fractional sinks by construction.  On
the 2026-09-23 day-5 states that inverted the story: the CAM6 arm removes only
0.54x the baseline's cloud water in absolute terms and condenses 2.23x as much,
so its cold-sector microphysics is a NET SOURCE of +0.090 kg/m2/day while the
baseline's is a NET SINK of -0.112.  Microphysics alone would GROW the CAM6
reservoir and SHRINK the baseline's, which is the opposite of the observed
deficit.  The band also does not balance, so terms of order 0.1 kg/m2/day from
transport and the other parameterizations are present and outside this probe.

SCALE FIRST, because it decides whether this instrument can answer anything.
A 31.5 % deficit on a band reservoir of 2.7e-02 kg/m2 sustained over 20 days is
an imbalance of about 4.2e-04 kg/m2/day.  The instantaneous terms below are
1e-01 to 4e-01, i.e. roughly 600x larger, and they cancel: measured on the
baseline's own day-5 and day-10 checkpoints, band-restricted, the true 5-day
mean rate is -5.2e-04 kg/m2/day against an instantaneous -2.8e-01, a factor of
535.  A single-state budget therefore CANNOT attribute this deficit, and no
amount of extra terms changes that.  Use this probe to compare PROCESSES
between arms, never to close a multi-day inventory.

WHAT "SUM" IS NOT (codex review, 2026-09-23).  It is a sum of instantaneous
process tendencies, NOT the band inventory derivative.  It omits q*d(dp)/dt,
the change in which layers fall inside the band as surface pressure moves, the
tracer vertical damper, and post-step adjustment and repair increments.  The
band is also selected by layer MIDPOINT pressure, so the 32- and 36-level grids
integrate slightly different effective pressure intervals; the sign and size of
that bias are unmeasured and it affects every term and the reservoir alike.

UPSTREAM TERMS.  ``--upstream`` adds the non-microphysical terms that change
band cloud water, so the band budget can be closed rather than only its
microphysics part: horizontal advection of q_c, vertical advection of q_c
(both operators, since the conservative lever does not reach tracers on the
hybrid lane), convective detrainment into q_c, and the sedimentation flux
across the two band edges.  Cloud droplets do not sediment in this scheme, so
that last term is reported for the condensate that does.

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
        "_ra_qcb", root / "scripts" / "run" / "run_amip.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["_ra_qcb"] = m
    spec.loader.exec_module(m)
    return m


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--restart", required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--band", nargs=2, type=float, default=[500.0, 800.0],
                    metavar=("P_LO_HPA", "P_HI_HPA"))
    ap.add_argument("--upstream", action="store_true",
                    help="add horizontal/vertical transport, convective "
                         "detrainment and band-edge sedimentation")
    ap.add_argument("--no-graupel", action="store_true",
                    help="MG2-faithful counterfactual: CAM6's MG2 carries no "
                         "graupel category at all (micro_mg_cam.F90:139), so "
                         "run the same state with do_graupel off and compare.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return ap


def band_rate(rate, dp, area_w, g, mask):
    """Area-weighted, mass-weighted column integral inside ``mask``."""
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
    T = jnp.asarray(state.T.data)
    ncol, nlev = T.shape
    mesh = model.mesh
    area = np.asarray(mesh.areaCell).reshape(ncol)
    A = area / area.sum()
    p_s = jnp.asarray(state.p_s.data)
    p_full = sc.pressure_at_full(p_s).reshape(ncol, nlev)
    p_half = sc.pressure_at_half(p_s).reshape(ncol, nlev + 1)
    dp = np.asarray(p_half[:, 1:] - p_half[:, :-1])
    g = constants.g

    def trac(n):
        if state.tracers is None or n not in state.tracers:
            return jnp.zeros_like(T)
        r = state.tracers[n]
        d = r.data if hasattr(r, "data") else r
        return jnp.asarray(d).reshape(ncol, nlev)

    q_v, q_c, q_r = trac("q_v"), trac("q_c"), trac("q_r")
    from legoesm.atmosphere.physics._shared import (
        compute_rho as _rho_of, compute_layer_dz as _dz_of)
    rho = _rho_of(T, p_full, q_v=q_v)
    dz = _dz_of(T, p_half, q_v=q_v)

    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    from legoesm.atmosphere.physics.microphysics.integration import (
        number_per_mass_to_per_volume)
    hyd = HydrometeorState(
        q_c=q_c, q_r=q_r, q_i=trac("q_i"), q_s=trac("q_s"), q_g=trac("q_g"),
        N_c=number_per_mass_to_per_volume(trac("N_c"), rho),
        N_r=number_per_mass_to_per_volume(trac("N_r"), rho),
        N_i=trac("N_i"))

    from legoesm.driver.physics_pipeline import _resolve_microphysics
    micro_fn, mcfg = _resolve_microphysics(config)
    dt_phys = float(config.dycore.dt) * int(
        getattr(config, "physics_update_steps", 1) or 1)
    n_macmic = int(getattr(config, "cld_macmic_num_steps", 1) or 1)
    dt_micro = dt_phys / max(1, n_macmic)
    print(f"[{a.label}] micro dt {dt_micro:.1f} s "
          f"({dt_phys:.1f} s physics / {n_macmic} macro-micro sub-step(s))")

    mcfg_b = mcfg._replace(publish_qc_budget=True)
    if a.no_graupel:
        mcfg_b = mcfg_b._replace(do_graupel=False)
        print(f"[{a.label}] COUNTERFACTUAL: do_graupel=False (MG2 carries no "
              f"graupel category; its riming of cloud water by snow, psacws, "
              f"sends the rimed mass to SNOW -- micro_mg2_0.F90:1894)")
    out = micro_fn(T, q_v, hyd, p_full, p_half, rho, dz, dt_micro, mcfg_b)
    if out.qc_budget is None:
        raise SystemExit(f"[{a.label}] scheme published no qc_budget; this "
                         f"probe requires the Morrison family")

    lo, hi = a.band[0] * 100.0, a.band[1] * 100.0
    pfn = np.asarray(p_full)
    band = (pfn >= lo) & (pfn < hi)

    # Temperature sectors.  Riming is gated below freezing by a SIGMOID in the
    # port (not a hard switch, so it is small but NOT structurally zero above
    # 273.15 K), and a band-integrated comparison dilutes it with warm layers.
    # The 273.15 K cut classifies supercooled liquid; mixed-phase physics spans
    # roughly 235-273 K, so this split is coarse (codex review, 2026-09-23).
    Tn = np.asarray(T)
    T_FREEZE = float(constants.T_freeze)
    sectors = (("whole band", band),
               ("cold (T<273.15 K)", band & (Tn < T_FREEZE)),
               ("warm (T>=273.15 K)", band & (Tn >= T_FREEZE)))
    for sname, mask in sectors:
        res = band_rate(np.asarray(q_c), dp, A, g, mask)
        if res <= 0.0:
            print(f"[{a.label}] --- {sname}: no cloud water")
            continue
        print(f"[{a.label}] --- {a.band[0]:.0f}-{a.band[1]:.0f} hPa, {sname}: "
              f"cloud water {res:.6e} kg/m2, "
              f"rain {band_rate(np.asarray(q_r), dp, A, g, mask):.6e} kg/m2, "
              f"{int(np.sum(mask))} cells")
        print(f"[{a.label}]   CLOSED cloud-water budget [kg/m2/day, "
              f"+ = source of cloud water]")
        tot = 0.0
        sink_rate = 0.0
        for name, term in out.qc_budget.items():
            v = band_rate(np.asarray(term), dp, A, g, mask) * _DAY
            tot += v
            if v < 0.0:
                sink_rate += -v / res
            print(f"[{a.label}]     {name:22s} {v:+13.6e} kg/m2/day   "
                  f"{v / res:+9.4f} /day of THIS ARM's reservoir "
                  f"({res:.4e} kg/m2)")
        net = band_rate(np.asarray(out.dq_c_dt), dp, A, g, mask) * _DAY
        denom = max(abs(band_rate(np.asarray(t), dp, A, g, mask) * _DAY)
                    for t in out.qc_budget.values()) or 1.0
        gross_sink = sum(
            -band_rate(np.asarray(t), dp, A, g, mask) * _DAY
            for t in out.qc_budget.values()
            if band_rate(np.asarray(t), dp, A, g, mask) < 0.0)
        print(f"[{a.label}]     {'GROSS SINK (absolute)':22s} {gross_sink:13.6e} "
              f"kg/m2/day  <- compare THIS between arms, not the fractions")
        print(f"[{a.label}]     {'SUM OF TERMS':22s} {tot:+13.6e}")
        print(f"[{a.label}]     {'scheme dq_c_dt':22s} {net:+13.6e}")
        print(f"[{a.label}]     {'RESIDUAL':22s} {tot - net:+13.6e}   "
              f"= {abs(tot - net) / denom:.3e} of the largest term")
        # NOT a residence time: the terms are summed over cells before their
        # sign is classified, so a cell where a term is a source cancels one
        # where it is a sink.  It is a band-integrated fractional turnover of
        # the negative-signed terms, useful only for comparing the same term
        # between arms (codex review, 2026-09-23).
        print(f"[{a.label}]     negative-term fractional turnover "
              f"{sink_rate:8.3f} per day  (NOT a gross sink, NOT a residence "
              f"time -- signs are classified after summing over cells)")

    res = band_rate(np.asarray(q_c), dp, A, g, band)

    if a.upstream:
        print(f"[{a.label}] === UPSTREAM terms on band cloud water "
              f"[kg/m2/day, + = source of band q_c] ===")
        u = {}
        # 1. transport.  The conservative lever does not reach tracers on the
        #    hybrid lane, so report what the model USES and what it would give.
        from legoesm.grids.vertical import (
            vertical_advection_hybrid, vertical_advection_hybrid_sb,
            compute_mass_flux_from_cumsum, HybridSigmaPressureCoordinate)
        from legoesm.core.operators_voronoi import (
            divergence_cell_3d, cell_to_edge_avg_3d)
        u_3d = jnp.asarray(state.u.data)
        dp_edge = cell_to_edge_avg_3d(jnp.asarray(dp), mesh)
        cumsum_dp = jnp.cumsum(divergence_cell_3d(u_3d * dp_edge, mesh), axis=-1)
        if isinstance(sc, HybridSigmaPressureCoordinate):
            mf = compute_mass_flux_from_cumsum(cumsum_dp, cumsum_dp[..., -1:], sc)
            u["vertical transport (in use)"] = band_rate(
                np.asarray(vertical_advection_hybrid(q_c, mf, p_s, sc)),
                dp, A, g, band) * _DAY
            u["vertical transport (conservative)"] = band_rate(
                np.asarray(vertical_advection_hybrid_sb(q_c, mf, p_s, sc)),
                dp, A, g, band) * _DAY
        else:
            # Sigma lane: the core builds sigma-dot from the SAME cumsum
            # (primitive_eq_mpas.py:687-689) and DOES pass the selector to
            # tracers (:799-800), so report the scheme the deck actually runs.
            from legoesm.grids.vertical import (
                vertical_advection, compute_sigma_dot_from_cumsum)
            sig_dot = compute_sigma_dot_from_cumsum(
                cumsum_dp, cumsum_dp[..., -1:], p_s, sc)
            _sch = getattr(model.config, "vert_advection_scheme", "upwind")
            u[f"vertical transport (in use, {_sch})"] = band_rate(
                np.asarray(vertical_advection(q_c, sig_dot, sc, scheme=_sch)),
                dp, A, g, band) * _DAY
        from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
            tracer_horizontal_advection)
        dqh = tracer_horizontal_advection(q_c[..., None], u_3d, mesh)[..., 0]
        u["horizontal transport"] = band_rate(
            np.asarray(dqh), dp, A, g, band) * _DAY

        # 2. convective detrainment into q_c.  This is a q_c SOURCE APPLIED
        #    OUTSIDE morrison's dq_c_dt (convection/integration.py:786-827), so
        #    the microphysics budget above is NOT the complete q_c budget and
        #    this term must be added (GLM review, 2026-09-23).  CLUBB is NOT a
        #    second external source on this deck: its closure liquid is
        #    discarded and only the fraction is published, measured at 95-98 %
        #    missing (diag_clubb_liquid_handoff.py).
        try:
            from legoesm.driver.physics_pipeline import _resolve_convection
            conv_fn, ccfg = _resolve_convection(config)
        except Exception as exc:                      # pragma: no cover
            print(f"[{a.label}]   convective detrainment: UNAVAILABLE ({exc})")
            conv_fn = None
        if conv_fn is not None:
            print(f"[{a.label}]   convective detrainment: run "
                  f"scripts/validate/diag_convective_detrainment.py for the "
                  f"per-band split; it is the committed instrument for this "
                  f"term and is not duplicated here")

        # 3. sedimentation across the two band edges.  Cloud droplets do not
        #    sediment in this scheme, so q_c has NO edge flux; the falling
        #    species do, and they are what rime q_c away inside the band.
        print(f"[{a.label}]   sedimentation of q_c across the band edges: "
              f"structurally ZERO -- cloud droplets do not sediment here "
              f"(morrison.py has no sed term in dq_c_dt).  The falling species "
              f"enter as riming partners, already inside the microphysics "
              f"terms above.")
        for k, v in u.items():
            print(f"[{a.label}]   {k:36s} {v:+13.6e}   "
                  f"{v / max(res, 1e-30):+9.4f} /day of the band reservoir")
        tot_up = sum(v for k, v in u.items()
                     if "conservative" not in k)
        print(f"[{a.label}]   {'TRANSPORT TOTAL (as the model runs it)':36s} "
              f"{tot_up:+13.6e}")

    # --- departure 1: MG2 in-cloud cloud-water cap ------------------------
    from legoesm.atmosphere.physics.microphysics.morrison import (
        resolve_morrison_flavor)
    from legoesm.thermo import saturation_mixing_ratio
    cfg = resolve_morrison_flavor(mcfg)
    if getattr(cfg, "subgrid_autoconversion", False):
        q_sat = saturation_mixing_ratio(T, p_full)
        arg = (1.0 - q_v / jnp.maximum(q_sat, 1.0e-10)) / max(
            1.0 - cfg.subgrid_rh_crit, 1.0e-6)
        cf = jnp.clip(jnp.where(arg > 0.0, 1.0 - jnp.sqrt(jnp.where(
            arg > 0.0, arg, 1.0)), 1.0), cfg.subgrid_cf_min, 1.0)
    else:
        cf = jnp.ones_like(q_c)
    q_c_ic = np.asarray(q_c / cf)
    MG2_CAP = 5.0e-3   # micro_mg2_0.F90:1226
    over = (q_c_ic > MG2_CAP) & band
    n_band = int(np.sum(band))
    qc_np = np.asarray(q_c)
    # Cloud water sitting in cells whose in-cloud value exceeds MG2's cap.
    mass_over = band_rate(np.where(over, qc_np, 0.0), dp, A, g,
                          np.ones_like(band, dtype=bool))
    print(f"[{a.label}] --- departure 1: MG2 in-cloud cap {MG2_CAP:.1e} kg/kg")
    print(f"[{a.label}]   in-cloud q_c exceeds the cap in {int(np.sum(over))} "
          f"of {n_band} band cells "
          f"({100.0 * np.sum(over) / max(n_band, 1):.3f} %)")
    print(f"[{a.label}]   cloud water in those cells: {mass_over:.6e} kg/m2 "
          f"= {100.0 * mass_over / max(res, 1e-30):.3f} % of the band reservoir")
    print(f"[{a.label}]   in-cloud q_c: mean {np.mean(q_c_ic[band]):.4e}  "
          f"p99 {np.percentile(q_c_ic[band], 99):.4e}  "
          f"max {np.max(q_c_ic[band]):.4e} kg/kg")
    print(f"[{a.label}]   cloud fraction in band: mean "
          f"{float(jnp.mean(cf[jnp.asarray(band)])):.4f}  "
          f"min {float(jnp.min(cf[jnp.asarray(band)])):.4f}")


if __name__ == "__main__":
    main()
