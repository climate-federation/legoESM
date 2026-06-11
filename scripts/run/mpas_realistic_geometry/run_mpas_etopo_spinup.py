"""MPAS realistic-bathymetry GO spinup on ico4 with ETOPO.

P6 of ``docs/ocean_experiments/realistic_geometry_mpas_plan.md`` —
the headline validation experiment.  Loads ETOPO bathymetry onto a
Voronoi mesh, applies MEO r-factor smoothing, builds the
partial-cell coordinate, and integrates a wind-driven GO spinup
with the full P3-P5 partial-cell stack active (Adcroft PGF, hybrid
vertex thickness, donor-cell continuity, min-rule edge thickness,
bottom drag at maxLevelEdgeBot).

Mirrors the lat-lon Phase 4 Wolfe-Cessi spinup
(``run_phase4_wolfe_cessi_spinup.py``); the lat-lon equivalent ran
100 yrs and confirmed equilibrated state (max|u|=1.21 m/s).

Default config is **smoke-test scale** (subdivision=4, 30 days) so
a developer can verify the script runs end-to-end without burning
HPC budget.  Production runs use the documented flags:

    # 5-yr ico4 GO spinup (the plan-spec headline run):
    JAX_ENABLE_X64=1 python scripts/run/mpas_realistic_geometry/run_mpas_etopo_spinup.py \\
        --subdivision 7 --years 5 --diag-every-days 30

    # Smoke test (default):
    JAX_ENABLE_X64=1 python scripts/run/mpas_realistic_geometry/run_mpas_etopo_spinup.py

Acceptance gates (production run):
- Integration completes ``years`` sim-years without NaN.
- ``max|u|`` stays bounded (< 2 m/s) once spun up.
- Time-mean V_baro grid-noise σ < 3× the implicit-CN flat-bottom
  baseline (1.92e-2 m/s per ``project_mpas_barotropic_noise.md``).
- AMOC magnitude in the observed range (15-25 Sv).
- No spurious deep flow under western boundary topography (visual).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
os.environ.setdefault("JAX_ENABLE_X64", "1")
jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.bathymetry import BathymetryConfig
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.init_mpas import (
    attach_static_rho_ref_z,
    rest_state_mpas_ocean,
)
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.lateral_mixing.config import (
    GMRediConfig,
    LateralMixingConfig,
)
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig,
    SurfaceForcingConfig,
)
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
)


def make_physics(tau_max: float):
    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="prescribed",
            prescribed=PrescribedForcingConfig(
                wind_profile="cosine_latitude", tau_max=tau_max,
            ),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
    )


def run(args):
    print(f"[mpas-etopo] subdivision={args.subdivision}, "
          f"H_max={args.H_max} m, n_levels={args.n_levels}")
    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
    z_coord = create_ocean_z_star(n_levels=args.n_levels, H_max=args.H_max)

    if args.etopo_path:
        bathy_cfg = BathymetryConfig(
            source="file", path=args.etopo_path,
            H_max=args.H_max, H_min=args.H_min,
            smoothing_passes=args.smoothing_passes,
            enforce_straits=True,
            r_factor_max=args.r_factor_max,
            depth_is_negative=True,
        )
        print(f"[mpas-etopo] loading ETOPO from {args.etopo_path}")
    else:
        # Idealized fallback (no ETOPO file available): test the full
        # pipeline with the lat-threshold land mask.
        bathy_cfg = None
        print("[mpas-etopo] no --etopo-path given; using idealized bathymetry")

    if bathy_cfg is not None:
        state = rest_state_mpas_ocean(
            mesh, z_coord, bathymetry=bathy_cfg,
            T_water_init_C=args.T_surf, T_deep=args.T_deep, S_uniform=args.S_uniform,
        )
    else:
        state = rest_state_mpas_ocean(
            mesh, z_coord, H_max=args.H_max,
            T_water_init_C=args.T_surf, T_deep=args.T_deep, S_uniform=args.S_uniform,
        )

    H_bathy = state.H_bathy.data
    pc_coord = create_partial_cell_coordinate(z_coord, H_bathy)
    print(f"[mpas-etopo] nCells={mesh.nCells}, ocean cells = "
          f"{int(jnp.sum(state.land_mask.data > 0.5))}, "
          f"H range = [{float(jnp.min(jnp.where(state.land_mask.data > 0.5, H_bathy, jnp.inf))):.0f}, "
          f"{float(jnp.max(H_bathy)):.0f}] m")

    cfg = MPASOceanConfig(
        barotropic_solver=args.barotropic_solver,
        A_h=args.A_h, A_v=args.A_v, K_v=1.0e-4, K_h=args.K_h,
        B_h=args.B_h, C_smag=args.C_smag,
        K_zeta_bih=args.k_zeta_bih,
        # apvm_dt: -1 sentinel → use baroclinic dt (MPAS-O default);
        # 0 disables; positive value uses that timescale directly.
        apvm_dt=args.apvm_dt if args.apvm_dt >= 0 else args.dt,
        bottom_drag_r=args.bottom_drag_r,
        # Distributed-BBL drag (Killworth & Edwards 1999, MOM6
        # ``BBL_thick_min``).  Required on partial-cell ETOPO: the
        # thinnest partial-cell bottom is ~0.3 m; legacy single-cell
        # drag CFL-violates at ``r·dt > h_bot``.  H_BBL=50 m gives a
        # well-behaved ~1% damping per step at thin cells; on deep
        # cells (h_bot ≥ 50) recovers the legacy single-cell form.
        # See plan-doc §8a.
        bottom_drag_bbl_thickness=args.bbl_thickness,
        physics=make_physics(args.tau_max),
        barotropic_implicit_pcg_tol=1.0e-10,
        barotropic_implicit_pcg_maxiter=300,
        min_water_column_m=1.0,
        # PGF scheme:
        #   "centered" — bare gradient on dz_ref; produced day-5
        #     bottom-trapped instability on ETOPO+ico4 (see plan doc).
        #   "adcroft"  — known-bad on legoesm (260x over-correction).
        #   "smc03"    — Shchepetkin & McWilliams 2003 density-Jacobian
        #     PGF; the real partial-cell PGF (per ROMS/CROCO/NEMO).
        #     This is the path the lat-lon equivalent used to reach
        #     100-yr stable spinups.  See
        #     ``docs/ocean_experiments/density_jacobian_pgf_mpas.md``.
        pgf_scheme=args.pgf_scheme,
        pv_scheme="enstrophy",
        tracer_advection="upwind",
        # Barotropic-mode lateral viscosity on u_bar.  Two flavors:
        #   * ``barotropic_u_viscosity`` (m²/s) — del2 (harmonic).
        #     Damps all scales uniformly; magnitudes needed to control
        #     the partial-cell mode (~3e6) over-damp real mesoscale
        #     flow.  Default 0.
        #   * ``barotropic_u_biharmonic`` (m⁴/s) — del4 (biharmonic).
        #     Scale-selective: damps grid-scale much harder than
        #     mesoscale.  PREFERRED for partial-cell ETOPO (see
        #     dycore-expert review in plan-doc §8c).  Default 1e16.
        # See ``project_mpas_etopo_instability.md`` for diagnostic.
        barotropic_u_viscosity=args.baro_u_viscosity,
        barotropic_u_biharmonic=args.baro_u_biharmonic,
        # Depth-dependent ρ_ref(z) for the baroclinic PGF.  On partial-
        # cell ETOPO this cuts the rest-state PGF residual ~24× (ocean-
        # expert review 2026-05-03; see plan-doc §8c).  The residual is
        # the seed of the partial-cell rotational u_bar mode; smaller
        # seed + del2 viscosity buys substantially longer stable runs.
        use_baroclinic_rho_ref=args.use_baroclinic_rho_ref,
        use_static_baroclinic_rho_ref=args.use_static_baroclinic_rho_ref,
        # Integrate baroclinic pressure cumsum against actual partial-
        # cell thickness h_k (NEMO ln_hpg_zps / MITgcm convention).
        # REQUIRED for the AC face correction to cancel the step-edge
        # artifact correctly — see plan-doc §8d.
        use_h_actual_pgf=args.use_h_actual_pgf,
        # Equatorial viscosity boost — extra damping at low latitudes
        # where the implicit-CN solver's Coriolis predictor-corrector
        # fails (f→0 at equator).  Targets the equatorial Pacific
        # bottom-trapped mode diagnosed in
        # project_mpas_etopo_instability.md.  Mirrors the lat-lon
        # ``A_h_lat_scaling`` (cos²(lat)) mechanism.
        equatorial_visc_boost=args.equatorial_visc_boost,
        # Threshold for the ``vertex_thickness_hybrid`` min-rule
        # fallback in the TRiSK PV term ``q = ζ/h_v``.  Audit
        # 2026-05-04: ``alpha=0.5`` (legacy) amplifies q by O(h_max/
        # h_min) at deep partial-cell step vertices (median 2.6×, p90
        # 12×, max ~870× on ETOPO+ico4) — exactly the topographic
        # regime where the bottom-trapped instability lives.
        # ``alpha=0.0`` always uses the kite-area mean (Petersen 2015)
        # and is the recommended setting for partial-cell ETOPO.
        vertex_thickness_alpha=args.vertex_thickness_alpha,
        # GM/Redi — production-grade parameterization of baroclinic
        # eddy buoyancy fluxes.  Targets the §8j late-phase mode
        # (positive-feedback baroclinic instability seeded by the
        # partial-cell PGF residual).  GM kappa damps available
        # potential energy by flattening isopycnals before they
        # grow large enough to feed the mode.  Lat-lon production
        # uses kappa=800 with Visbeck adaptive 200-2000 (PR #231);
        # we start with a fixed kappa for diagnostic clarity.
        # MPAS supports only ``slope_scheme="centered"``.
        gm_redi=(
            GMRediConfig(
                kappa_GM=args.gm_redi_kappa,
                kappa_Redi=args.gm_redi_kappa,
                slope_scheme="centered",
            )
            if args.gm_redi_kappa > 0
            else None
        ),
    )
    # Freeze the per-level reference density ρ_ref(z) on the state
    # before the first step.  No-op when --no-use-static-baroclinic-rho-
    # ref.  Must be done AFTER pc_coord is built so that step-edge cells
    # are excluded from the wet-cell mean.
    state = attach_static_rho_ref_z(state, mesh, pc_coord, cfg)
    if state.rho_ref_z is not None:
        rrz = state.rho_ref_z.data
        print(f"[mpas-etopo] static ρ_ref(z) attached: "
              f"min={float(jnp.min(rrz)):.3f}, "
              f"max={float(jnp.max(rrz)):.3f}, "
              f"surface={float(rrz[0]):.3f}, "
              f"bottom={float(rrz[-1]):.3f} kg/m³")
    model = MPASOceanModel(mesh, pc_coord, cfg)

    n_days = args.years * 365.25
    n_steps = int(n_days * 86400.0 / args.dt)
    diag_stride = max(1, int(args.diag_every_days * 86400.0 / args.dt))
    print(f"[mpas-etopo] dt={args.dt}s, n_steps={n_steps}, "
          f"diag every {args.diag_every_days} days")

    t0 = time.time()
    s = state
    for k in range(n_steps):
        s = model.step(s, dt=args.dt)
        if (k + 1) % diag_stride == 0:
            sim_days = (k + 1) * args.dt / 86400.0
            mu = float(jnp.max(jnp.abs(s.u.data)))
            me = float(jnp.max(jnp.abs(s.eta.data)))
            mt = float(jnp.max(s.T.data))
            elapsed = time.time() - t0
            print(f"  t={sim_days:7.1f}d : max|u|={mu:.3e} m/s, "
                  f"max|eta|={me:.3e} m, max T={mt:.2f} C, "
                  f"wall={elapsed:.1f}s")
            if not (np.isfinite(mu) and np.isfinite(me)):
                raise RuntimeError(
                    f"State went non-finite at t={sim_days}d — "
                    f"investigate stability before proceeding"
                )

    print(f"[mpas-etopo] done in {time.time() - t0:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subdivision", type=int, default=4,
                   help="Voronoi subdivision level (4=2562, 7=163842 cells)")
    p.add_argument("--n-levels", dest="n_levels", type=int, default=20)
    p.add_argument("--H-max", dest="H_max", type=float, default=5500.0)
    p.add_argument("--H-min", dest="H_min", type=float, default=10.0)
    p.add_argument("--T-surf", dest="T_surf", type=float, default=20.0)
    p.add_argument("--T-deep", dest="T_deep", type=float, default=2.0)
    p.add_argument("--S-uniform", dest="S_uniform", type=float, default=35.0)
    p.add_argument("--years", type=float, default=0.083,
                   help="Simulation years (default ~30 days for smoke test)")
    p.add_argument("--dt", type=float, default=500.0,
                   help="Baroclinic timestep [s]; ico-4 + ETOPO + "
                        "centered PGF NaNs above ~525s (TBD: identify "
                        "which mode is the limiter — barotropic CFL "
                        "is well-resolved at 600s with implicit-CN, so "
                        "this is most likely a partial-cell-specific "
                        "stiff mode the implicit solver doesn't damp).")
    p.add_argument("--diag-every-days", dest="diag_every_days", type=float, default=5.0)
    p.add_argument("--A-h", dest="A_h", type=float, default=1.0e6,
                   help="Lateral viscosity [m²/s].  Default 1e6 for "
                        "ico-4 + ETOPO; the legacy 1e5 gives a 30-day "
                        "diffusion timescale (dx²/A_h ≈ 30d on ~500km "
                        "cells) which is far too weak to damp the "
                        "bottom-trapped O(1-day) topographic mode.  "
                        "See plan-doc §8a A_h sensitivity scan.")
    p.add_argument("--K-h", dest="K_h", type=float, default=1.0e3)
    p.add_argument("--C-smag", dest="C_smag", type=float, default=0.0,
                   help="Smagorinsky coefficient [dimensionless], typical "
                        "0.01-0.15.  Flow-dependent biharmonic viscosity "
                        "``-del2(A_smag·del2(u))`` with A_smag=(C·Δ)²·|D|. "
                        "Targets the §8j late-phase mode where strain "
                        "rate is high.")
    p.add_argument("--A-v", dest="A_v", type=float, default=1.0e-3,
                   help="Vertical viscosity [m²/s].  Default 1e-3.  "
                        "Targets the §8j bottom-trapped momentum mode "
                        "(damps vertical shear of u directly).")
    p.add_argument("--bottom-drag-r", dest="bottom_drag_r", type=float,
                   default=1.1e-3,
                   help="Bottom drag coefficient [m/s].  Default 1.1e-3. "
                        "Lat-lon production uses 2.5e-3.  Target the "
                        "§8j bottom-trapped momentum mode.")
    p.add_argument("--gm-redi-kappa", dest="gm_redi_kappa", type=float,
                   default=0.0,
                   help="GM/Redi diffusivity [m²/s].  0 disables (default). "
                        "Lat-lon production uses 800.  Targets the §8j "
                        "late-phase baroclinic mode by parameterizing "
                        "eddy flux of buoyancy.  MPAS uses centered "
                        "slope scheme.")
    p.add_argument("--B-h", dest="B_h", type=float, default=0.0,
                   help="Biharmonic ∇⁴ viscosity on 3D velocity [m⁴/s]. "
                        "Scale-selective deep-mode damping.  Lat-lon "
                        "production stack uses 5e9 (PR #231).  Targets "
                        "the late-phase baroclinic mode identified in "
                        "plan-doc §8j (audit-fixed, but PGF-residual-"
                        "seeded baroclinic instability at sub-deformation-"
                        "radius scales — lives at grid scale on ico-4).")
    p.add_argument("--bbl-thickness", dest="bbl_thickness", type=float, default=50.0,
                   help="BBL thickness [m] for distributed bottom drag. "
                        "Default 50; set 0 to revert to legacy single-cell "
                        "drag (CFL-violates on partial-cell ETOPO).")
    p.add_argument("--tau-max", dest="tau_max", type=float, default=0.05)
    p.add_argument("--smoothing-passes", dest="smoothing_passes", type=int, default=2)
    p.add_argument("--r-factor-max", dest="r_factor_max", type=float,
                   default=0.2,
                   help="MEO r-factor cap on bathymetry [0,1].  Tighter "
                        "= more smoothing of step edges = smaller PGF "
                        "residual.  Default 0.2 (lat-lon production "
                        "scale; sweep showed 0.2 is the sweet spot, "
                        "0.15 is non-monotonically worse).")
    p.add_argument("--etopo-path", dest="etopo_path", type=str, default=None,
                   help="Path to ETOPO/GEBCO NetCDF; idealized if omitted")
    p.add_argument("--pgf-scheme", dest="pgf_scheme", type=str,
                   default="adcroft",
                   choices=["centered", "adcroft", "smc03", "ahh08", "zero"],
                   help="PGF scheme.  Default 'adcroft' = centered + "
                        "Adcroft-Campin 2004 face correction with "
                        "h_actual integration (the standard MITgcm/"
                        "MOM6/NEMO ln_hpg_zps recipe for z*+partial "
                        "cells).  'smc03' was the previous default; on "
                        "lat-lon and MPAS ETOPO the two schemes produce "
                        "identical answers to 4 sig figs (lat-lon AC vs "
                        "SMC03 30-day comparison) so 'adcroft' is "
                        "preferred for code clarity / production "
                        "alignment.  See plan-doc §8d.  "
                        "'ahh08' = Adcroft-Hallberg-Hill 2008 analytic "
                        "FV PGF; gives machine-zero rest-state residual "
                        "on partial cells (vs adcroft's 1.4e-6).  "
                        "Requires eos='wright'.  See plan-doc §8i.")
    p.add_argument("--baro-u-viscosity", dest="baro_u_viscosity",
                   type=float, default=3.0e6,
                   help="Harmonic ∇² damping on u_bar [m²/s].  Default "
                        "3e6 cuts 30-day eta growth 3.6× vs disabled "
                        "and is the most effective single-knob lever "
                        "we've found for the partial-cell rotational "
                        "u_bar mode.  See plan-doc §8b.")
    p.add_argument("--baro-u-biharmonic", dest="baro_u_biharmonic",
                   type=float, default=0.0,
                   help="Biharmonic ∇⁴ damping on u_bar [m⁴/s].  Scale-"
                        "selective; in our 90-day tau=0 sweep it was "
                        "less effective than del2 at controlling early "
                        "eta growth.  Default 0; opt in if you want a "
                        "less-aggressive damping profile.")
    p.add_argument("--apvm-dt", dest="apvm_dt", type=float, default=0.0,
                   help="Anticipated PV Method timescale [s].  Targets "
                        "the TRiSK PV-flux null branch.  In our 90-day "
                        "sweep it was a no-op for the partial-cell "
                        "u_bar mode (the mode doesn't go through PV).  "
                        "Default 0 (disabled).  Set positive or pass "
                        "the baroclinic dt to enable.")
    p.add_argument("--k-zeta-bih", dest="k_zeta_bih", type=float,
                   default=0.0,
                   help="Biharmonic dissipation on relative vorticity ζ "
                        "[m⁴/s].  Same reasoning as --apvm-dt: targets "
                        "PV-flux null mode, ineffective on this u_bar "
                        "instability.  Default 0 (disabled).")
    p.add_argument("--equatorial-visc-boost",
                   dest="equatorial_visc_boost",
                   type=float, default=5.0,
                   help="Multiplicative boost on lateral viscosity at "
                        "low latitudes: A_eff = A · (1 + boost · "
                        "cos²(lat_edge)).  Targets the equatorial "
                        "Pacific bottom-trapped mode where f→0 nullifies "
                        "the implicit-CN Coriolis stabilizer.  Default "
                        "5.0; 0 disables.  Mirrors the lat-lon "
                        "A_h_lat_scaling mechanism.")
    p.add_argument("--use-baroclinic-rho-ref",
                   dest="use_baroclinic_rho_ref",
                   action=argparse.BooleanOptionalAction,
                   default=False,
                   help="DYNAMIC ρ_ref(z) — wet-cell mean recomputed every "
                        "tendency call.  Known to NaN at day 60 on ETOPO+"
                        "ico4 because ρ_ref chases T,S drift (positive "
                        "feedback).  Use --use-static-baroclinic-rho-ref "
                        "instead.  Kept only for diagnostic comparison.")
    p.add_argument("--use-static-baroclinic-rho-ref",
                   dest="use_static_baroclinic_rho_ref",
                   action=argparse.BooleanOptionalAction,
                   default=False,
                   help="STATIC ρ_ref(z) — frozen at init from EOS(T_init, "
                        "S_init) averaged over wet cells per level.  Used "
                        "as ρ'=ρ-ρ_ref(z) in the baroclinic PGF.  Cuts "
                        "the partial-cell PGF residual (the seed of the "
                        "bottom-trapped rotational mode) ~24× without "
                        "the dynamic version's drift feedback (Option B "
                        "from project_mpas_etopo_instability.md §8g).  "
                        "Mutually exclusive with --use-baroclinic-rho-ref.")
    p.add_argument("--barotropic-solver",
                   dest="barotropic_solver", type=str,
                   default="implicit_cn",
                   choices=["implicit_cn", "explicit_substep"],
                   help="Barotropic solver. ``implicit_cn`` (default) is "
                        "the production stack used for §8a-§8j.  "
                        "``explicit_substep`` is the §8i probe 2 alternative "
                        "(now safe on partial cells per the Audit 1 fix in "
                        "barotropic_mpas.py).")
    p.add_argument("--vertex-thickness-alpha",
                   dest="vertex_thickness_alpha", type=float,
                   default=0.5,
                   help="Threshold for the vertex_thickness_hybrid min-rule "
                        "fallback (use_min triggered when h_min < alpha*h_max). "
                        "Default 0.5 (legacy).  Set 0.0 to disable the min "
                        "branch entirely (kite-mean only) — the audit "
                        "2026-05-04 finding for partial-cell ETOPO.  See "
                        "plan-doc §8j (when written).")
    p.add_argument("--use-h-actual-pgf",
                   dest="use_h_actual_pgf",
                   action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Integrate the baroclinic pressure cumsum "
                        "against the actual partial-cell thickness h_k "
                        "rather than the reference dz_ref.  This is the "
                        "standard MITgcm/MOM6/NEMO ln_hpg_zps recipe "
                        "and is REQUIRED for the AC face correction to "
                        "cancel the step-edge artifact (without it, AC "
                        "over-corrects 260×).  Default True; pass --no-"
                        "use-h-actual-pgf to revert to dz_ref.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
