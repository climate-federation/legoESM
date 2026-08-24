"""Marine stratocumulus on the spectral TRUE-LES core + swappable microphysics
(Morrison/M2005 default; non-drizzling RF01).

Three decks, selected with ``--case``: DYCOMS-II RF01 (default, described
below), DYCOMS-II RF02 (``rf02``, the DRIZZLING flight of the same campaign)
and ASTEX flight 209. All three set ``dolongwave = .true., doradsimple =
.true.``, so all three are driven by the SAME Stevens (2005) simple longwave
with the same constants -- gSAM's rad_simple hardcodes them and applies them to
every deck that selects it. They differ in Coriolis (ASTEX has none),
subsidence divergence, domain depth, droplet concentration and the height the
initial theta perturbation is seeded below; see ``_STRATOCUMULUS_CASES``.

DYCOMS-II RF02 (Ackerman et al. 2009, MWR 137; gSAM ``CASES/DYCOMS_RF02``) is
the same nocturnal marine stratocumulus regime as RF01 but DRIZZLING: its deck
sets ``doprecip = .true.`` and a droplet concentration of 55 cm^-3 against
RF01's non-precipitating 140 cm^-3. Its inversion sits at 795 m (theta_l =
288.3 K, q_t = 9.45 g/kg below it), its geostrophic wind is SHEARED
(u_g = 3.0 + 4.3 z_km, v_g = -9.0 + 5.6 z_km, read off the deck's lsf rather
than assumed uniform) and its surface fluxes are SHF = 16, LHF = 93 W/m^2.

DYCOMS-II RF01 (Stevens et al. 2005, MWR 133; gSAM ``CASES/DYCOMS_RF01``):

* IC: the gSAM ``snd`` — NOTE its temperature column is LIQUID-WATER potential
  temperature θ_l (289 K mixed layer, 9 g/kg q_t, sharp inversion at 840 m;
  the 600–840 m cloud layer is SATURATED). The driver saturation-adjusts
  (θ_l, q_t) → (θ, q_v, q_c) with the SHARED Tetens saturation (no re-derived
  Clausius–Clapeyron) and seeds N_c = 140 cm⁻³ in cloud (the RF01 droplet
  concentration).
* Radiation: the Stevens et al. (2005) PARAMETERIZED nocturnal LW —
  F(z) = F0·e^(−Q(z,top)) + F1·e^(−Q(0,z)) + a·ρ_i·c_p·D·[(z−z_i)^{4/3}/4 +
  z_i·(z−z_i)^{1/3}] for z > z_i, with Q = κ∫ρ q_l dz, κ=85 m²/kg, F0=70,
  F1=22 W/m², a=1, D=3.75e-6 s⁻¹; heating dT/dt = −(1/ρc_p)·∂F/∂z. This IS
  the case's cooling engine (cloud-top radiative driving); SAM runs it as
  ``doradsimple``.
* Subsidence: w_ls = −D·z on θ and q_t (slots 0+1).
* Surface: fixed fluxes SHF=15, LHF=115 W/m²; Coriolis f=0.376e-4 toward
  (u_g, v_g) = (7, −5.5) m/s.

Reference targets (Stevens 2005, hours 2–4): LWP ~50–80 g/m², cloud cover
~100 %, z_i ≈ 840–870 m (entrainment ~0.4 cm/s), well-mixed θ_l/q_t, ⟨w²⟩
single peak ~0.4 m²/s² mid-BL.

Example (GPU):
    JAX_PLATFORMS=cuda .venv/bin/python scripts/run/run_dycoms_les.py \\
        --nx 96 --ny 96 --nz 192 --hours 4 --f32
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

_F32 = "--f32" in sys.argv
import jax  # noqa: E402

if not _F32:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402
from legoesm.atmosphere.dynamics.les.spectral_les_moist import (  # noqa: E402
    make_anelastic_reference,
    make_les_microphysics_fn,
)
from legoesm.atmosphere.forcing.sam_case_forcing import (  # noqa: E402
    read_sam_lsf,
    read_sam_sfc,
    read_sam_snd,
    surface_at_day,
)
from legoesm.atmosphere.physics.microphysics.config import (  # noqa: E402
    MicrophysicsConfig,
    MorrisonConfig,
)
from legoesm.atmosphere.physics.radiation.simple_lw import (  # noqa: E402
    SimpleLWConfig,
    simple_lw_temperature_tendency,
)

from legoesm import constants  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import les_record  # noqa: E402
from legoesm.atmosphere.forcing.sam_case_forcing import resolve_sam_case_dir  # noqa: E402
from legoesm.atmosphere.les_suite.scm_coupling import (  # noqa: E402
    liquid_water_theta,
    saturation_adjust,
)


def _record_theta_l(st, ref):
    """3D liquid-water potential temperature θ_l to RECORD as moist truth. The spectral moist
    LES prognoses ACTUAL θ (``state.theta``; the IC saturation-adjusts θ_l→θ), so θ_l is
    DERIVED from θ + the LES cloud liquid q_c (slot 1) — else the score compares LES θ to SCM
    θ_l. q_c only (not rain), matching ``scm_final_moist_on``. Shared canonical reduction."""
    exner = np.asarray(ref.exner_c)[None, None, :]
    return np.asarray(liquid_water_theta(
        np.asarray(st.theta), np.asarray(st.tracers[..., 1]), exner))

# Default case dir: external LEGOESM_GSAM_ROOT if set, else the repo-local
# cache (scripts/data/fetch_les_forcing.py); --case-dir overrides. See
# resolve_sam_case_dir.
_DEFAULT_CASE = resolve_sam_case_dir("DYCOMS_RF01")
# The two marine-stratocumulus decks this driver can run. Both set
# `dolongwave = .true., doradsimple = .true.` in their prm, i.e. both are
# driven by the SAME Stevens (2005) simple longwave with the SAME constants --
# gSAM's rad_simple hardcodes them and applies them to every deck that selects
# it, so ASTEX legitimately runs the RF01 numbers.
#
# What DOES differ is per case, and each entry is read off that deck rather
# than inherited from DYCOMS:
#   f_cor        ASTEX209 sets `docoriolis = .false.`, so its LES has none.
#   subsidence_divergence_s   DYCOMS subsides as w = -D z with D = 3.75e-6.
#                ASTEX's lsf gives w = -0.010 m/s at 2000 m, i.e. D = 5.0e-6.
#                This is NOT the longwave's D: gSAM's rad_simple hardcodes
#                f0 = 3.75e-6 in its clear-sky term for every deck, and the
#                subsidence comes from the lsf file. The two coincide for
#                DYCOMS and differ for ASTEX, so they are separate here --
#                reusing one for the other is a silent wrong forcing.
#   lz_m         ASTEX's inversion sits near 637 m but its sounding runs to
#                1637 m, so the domain has to clear it.
# The DYCOMS entry reproduces the historical hardcoded values exactly.
#   perturb_z_m  the initial theta noise is seeded BELOW this height, which is
#                the case's inversion: seeding above it puts noise in the free
#                troposphere the case does not perturb. RF01's 840 m was a
#                literal in build(); RF02's inversion is at 795 m.
_STRATOCUMULUS_CASES = {
    "dycoms": {
        "gsam_dir": "DYCOMS_RF01",
        "f_cor": 0.376e-4,   # the RF01 driver's own value
        "subsidence_divergence_s": 3.75e-6,
        "lz_m": 1500.0,
        "n_c_m3": 140.0e6,
        "perturb_z_m": 840.0,
        # Printed next to the run's own diagnostics. Per case, because these
        # ARE RF01's numbers: they were printed for every deck, so an ASTEX or
        # RF02 run was scored in the log against a case it is not.
        "ref_targets": "LWP 50-80 g/m^2, cloud cover ~1.0, z_i 840-870 m "
                       "(Stevens et al. 2005, hours 2-4)",
        "note": "Stevens et al. 2005 RF01 nocturnal stratocumulus.",
    },
    "rf02": {
        "gsam_dir": "DYCOMS_RF02",
        # READ OFF THIS DECK, not inherited from RF01. RF01's prm sets
        # `fcor = 0.376e-4` explicitly -- a value that is NOT 2*Omega*sin(31.5)
        # -- while RF02's prm carries no fcor at all and sets
        # `latitude0 = 31.5`, leaving gSAM to derive f from the latitude. So
        # the two flights of the same campaign really do run different
        # rotation rates, and copying RF01's would halve RF02's.
        #
        # A reviewer argued the RF02 intercomparison inherits RF01's 0.376e-4
        # and that the community decks hardcode it. Checked against two other
        # RF02 implementations rather than settled by argument: PyCLES sets
        # `coriolis_param = 2.0*omega*sin(31.5*pi/180)` in the forcing class it
        # shares between RF01 and RF02, and DALES' cases/dycoms_rf02 namelist
        # sets `xlat = 31.5, lcoriol = .true.` with no explicit f. Three
        # independent codes, the same latitude-derived value.
        "f_cor": 2.0 * constants.Omega * float(np.sin(np.deg2rad(31.5))),
        # lsf: w_ls = -0.00598 m/s at 1595 m, i.e. D = 3.75e-6 -- the same
        # divergence as RF01. Still declared per case: the equality is a fact
        # about these two decks, not a rule (ASTEX's is 5.0e-6).
        "subsidence_divergence_s": 3.75e-6,
        # The deck's own stretched grid tops at 1459 m and its sounding runs to
        # 1595 m, so a 1500 m lid is spanned by the sounding.
        "lz_m": 1500.0,
        # prm &MICRO_DRIZZLE Nc0 = 55 cm^-3, and the RF02 SCM/LES
        # intercomparison specifies the same. RF01 runs 140 cm^-3.
        #
        # This driver has no equivalent of gSAM's `doprecip` switch -- Morrison's
        # warm-rain process is always active -- so the droplet concentration is
        # the only lever the case has on its defining drizzle, and it is a real
        # one: on the SAME RF02 column, 55 cm^-3 produces 5.3x the rain
        # tendency that 140 cm^-3 does (the KK2000-type N_c^-1.79), measured in
        # tests/unit/test_stratocumulus_les_case_selector.py.
        "n_c_m3": 55.0e6,
        # snd: theta_l steps 288.300 -> 296.710 K between 795 and 800 m.
        "perturb_z_m": 795.0,
        # No targets registered: the RF02 intercomparison's ensemble LWP /
        # cloud-cover / entrainment numbers are not in this repo, and RF01's
        # do not transfer -- RF02 is deeper, moister and drizzling.
        "ref_targets": None,
        "note": "Ackerman et al. 2009 RF02 nocturnal DRIZZLING stratocumulus; "
                "prescribed surface fluxes (SHF 16, LHF 93 W/m^2), sheared "
                "geostrophic wind from the deck lsf.",
    },
    "astex": {
        "gsam_dir": "ASTEX209",
        "f_cor": 0.0,                 # prm: docoriolis = .false.
        "subsidence_divergence_s": 5.0e-6,   # lsf: w=-0.010 m/s at 2000 m
        "lz_m": 2500.0,
        "n_c_m3": 100.0e6,            # ASTEX intercomparison N_c
        # UNCHANGED from the literal build() used for every deck, so this diff
        # does not move ASTEX. It is almost certainly too high -- ASTEX's
        # inversion is near 637 m -- but correcting it is a separate change
        # with its own reference run.
        "perturb_z_m": 840.0,
        "ref_targets": None,
        "note": "ASTEX flight 209 stratocumulus (Sc-to-Cu transition deck).",
    },
}
# The longwave fit lives in legoesm.atmosphere.physics.radiation.simple_lw,
# which the single-column model calls too. Its constants -- including its own
# divergence -- are hardcoded in gSAM's rad_simple for EVERY deck that selects
# it, so this one instance serves both cases. The SUBSIDENCE divergence is per
# case and comes from _STRATOCUMULUS_CASES; do not substitute one for the other.
_SIMPLE_LW = SimpleLWConfig()
_QT_INV = _SIMPLE_LW.qt_inversion_kg_kg   # z_i: q_t isoline [kg/kg]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=sorted(_STRATOCUMULUS_CASES),
                   default="dycoms",
                   help="which marine-stratocumulus deck to run. All are "
                        "driven by the same Stevens (2005) simple longwave; "
                        "they differ in Coriolis, subsidence divergence, "
                        "domain depth, droplet concentration and the height "
                        "the initial perturbation is seeded below.")
    p.add_argument("--case-dir", default=None,
                   help="override the deck directory --case resolves to.")
    p.add_argument("--nx", type=int, default=96)
    p.add_argument("--ny", type=int, default=96)
    p.add_argument("--nz", type=int, default=192,
                   help="dz=7.8 m at Lz=1500 (official 5 m needs nz=300, "
                        "beyond the projection's nz≲200 compile cliff).")
    p.add_argument("--Lx", type=float, default=3360.0)
    p.add_argument("--Ly", type=float, default=3360.0)
    p.add_argument("--Lz", type=float, default=None,
                   help="domain depth [m]; defaults to the case's own.")
    p.add_argument("--z0", type=float, default=1.0e-4)
    p.add_argument("--hours", type=float, default=4.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--microphysics", default="morrison")
    p.add_argument("--n-tracers", type=int, default=9)
    p.add_argument("--scalar-advection", choices=["van_leer","weno5","weno5_hv"], default="van_leer",
                   help="monotone scalar reconstruction (weno5=less diffusive).")
    p.add_argument("--w-hyperdiff", type=float, default=0.0,
                   help="OPT-IN horizontal w-hyperdiffusion nu4 [m^4/s].")
    p.add_argument("--theta-hyperdiff", type=float, default=0.0,
                   help="OPT-IN scale-selective k4 hyperdiff on theta [m^4/s].")
    p.add_argument("--div-damping", type=float, default=0.0,
                   help="OPT-IN momentum divergence damping alpha [m^2/s].")
    p.add_argument("--dynamic", action="store_true", default=True)
    p.add_argument("--static", dest="dynamic", action="store_false")
    p.add_argument("--sgs-model", choices=["smagorinsky", "vreman"],
                   default="vreman")
    p.add_argument("--cs", type=float, default=0.18)
    p.add_argument("--sgs-buoyancy", dest="sgs_buoyancy", action="store_true",
                   default=True, help="Lilly stable-stratification SGS suppression "
                   "(REQUIRED for stratocumulus; on by default).")
    p.add_argument("--no-sgs-buoyancy", dest="sgs_buoyancy", action="store_false",
                   help="disable Lilly SGS suppression (strain-only ν_t; "
                        "over-entrains the inversion — for the controlled A/B).")
    p.add_argument("--nu-floor", type=float, default=0.0)
    p.add_argument("--time-scheme", choices=["rk3", "ab2"], default="rk3")
    p.add_argument("--micro-every", type=int, default=1)
    p.add_argument("--rad-every", type=int, default=1,
                   help="Stevens-LW recompute cadence [steps].")
    p.add_argument("--print-every", type=int, default=1000)
    p.add_argument("--record-frames", type=int, default=8)
    p.add_argument("--case-label", type=str, default=None,
                   help="Label stamped into every recorded frame, which "
                        "`legoesm.training.les_reference` checks against the "
                        "case it is asked to build. DEFAULTS TO --case; pass "
                        "it only to override. It used to default to the "
                        "literal 'dycoms' for BOTH decks, so `--case astex` "
                        "without this flag stamped a genuine 6 h ASTEX run "
                        "(domain top 1995 m against DYCOMS's 1496 m) as "
                        "'dycoms', and the reference loader refused it.")
    p.add_argument("--output", type=Path, default=Path("results/les_dycoms"))
    p.add_argument("--emit-suite-artifact", type=Path, default=None,
                   help="after the run, assemble a moist LESReferenceArtifact (the SCM "
                        "tuner's input) from the recorded prof series + RF01 radiative "
                        "forcing and write it to this path")
    args = p.parse_args()
    # The frame label follows the case unless it was given explicitly. The old
    # literal default meant the ONE field that identifies a reference was wrong
    # precisely when it mattered: `--case astex` produced 6 h of genuine ASTEX
    # stamped "dycoms", which the reference loader then refused (and should --
    # accepting the alias would let a real DYCOMS directory become the ASTEX
    # reference). Both deck names ARE the reference registry's names here, so
    # unlike run_spectral_cbl.py no mapping is needed.
    args.case_label = args.case_label or args.case
    # Resolve the case's own defaults, letting an explicit flag win. argparse
    # cannot tell "given" from "equal to the default", so the overridable
    # fields default to None and are filled in here.
    spec = _STRATOCUMULUS_CASES[args.case]
    args.case_spec = spec
    if args.case_dir is None:
        args.case_dir = resolve_sam_case_dir(spec["gsam_dir"])
    if args.Lz is None:
        args.Lz = spec["lz_m"]
    args.f_cor = spec["f_cor"]
    args.subsidence_divergence_s = spec["subsidence_divergence_s"]
    args.n_c_m3 = spec["n_c_m3"]
    args.perturb_z_m = spec["perturb_z_m"]
    return args


def build(args, dtype):
    cfg = sl.SpectralLESConfig(
        nx=args.nx, ny=args.ny, nz=args.nz, Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        z0=args.z0, dealias=True, c_s=args.cs,
        smagorinsky_dynamic=args.dynamic, sgs_model=args.sgs_model,
        time_scheme=args.time_scheme, nu_floor=args.nu_floor,
        buoyancy=True, theta_ref0=290.0, pr_sgs=1.0,
        sgs_buoyancy=args.sgs_buoyancy,
        moist=True, n_tracers=args.n_tracers, monotone_scalars=True,
        scalar_advection=args.scalar_advection,
        w_hyperdiff_coeff=args.w_hyperdiff, div_damping_coeff=args.div_damping,
        theta_hyperdiff_coeff=args.theta_hyperdiff)
    g = sl.make_grid(cfg, dtype=dtype)
    case = Path(args.case_dir)
    if not case.is_dir():
        sys.exit(
            f"[run_dycoms_les] case directory not found: {case}\n"
            f"  --case {args.case} reads the gSAM "
            f"CASES/{args.case_spec['gsam_dir']} deck. Populate the "
            f"repo-local cache:\n"
            f"    python scripts/data/fetch_les_forcing.py --only "
            f"{'ASTEX' if args.case == 'astex' else 'DYCOMSII'}\n"
            f"  or point at a checkout: LEGOESM_GSAM_ROOT=<root containing "
            f"CASES/>, or --case-dir <path/to/CASES/DYCOMS_RF01>.")
    snd = read_sam_snd(case / "snd")
    lsf = read_sam_lsf(case / "lsf")
    sfc0 = surface_at_day(read_sam_sfc(case / "sfc"), day=0.0)

    z_c = np.asarray(g.z_c)
    z_snd = np.asarray(snd.z)
    thl_prof = np.interp(z_c, z_snd, np.asarray(snd.theta))   # θ_l!
    qt_prof = np.interp(z_c, z_snd, np.asarray(snd.q_v))      # q_t [kg/kg]
    u_prof = np.interp(z_c, z_snd, np.asarray(snd.u))
    v_prof = np.interp(z_c, z_snd, np.asarray(snd.v))
    zl = np.asarray(lsf.z[0]); order = np.argsort(zl)
    ug = np.interp(z_c, zl[order], np.asarray(lsf.u_ls[0])[order])
    vg = np.interp(z_c, zl[order], np.asarray(lsf.v_ls[0])[order])

    p_sfc = float(snd.pres0) * 100.0
    # Reference column from the LIQUID-FREE part of the sounding (θ_l ≈ θ in
    # the subcloud + above-inversion air; the ≤0.5 g/kg in-cloud difference is
    # immaterial for the hydrostatic reference).
    ref = make_anelastic_reference(z_c, np.asarray(g.z_f), p_sfc,
                                   thl_prof, qt_prof, dtype=dtype)
    # Saturation-adjust the IC: (θ_l, q_t) → (θ, q_v, q_c) on the column.
    exner = jnp.asarray(ref.exner_c, dtype)
    p_c = jnp.asarray(ref.p_c, dtype)
    th_col, qv_col, qc_col = saturation_adjust(
        jnp.asarray(thl_prof, dtype), jnp.asarray(qt_prof, dtype), exner, p_c)

    ny, nx, nz = args.ny, args.nx, args.nz
    key = jax.random.PRNGKey(0)
    # Seed the theta noise below the CASE's inversion (840 m for RF01, 795 m
    # for RF02): a literal here perturbs the free troposphere of any case whose
    # inversion is lower.
    seed = (jnp.asarray(z_c) < args.perturb_z_m).astype(dtype)
    th3 = (jnp.broadcast_to(th_col, (ny, nx, nz))
           + 0.1 * jax.random.normal(key, (ny, nx, nz), dtype) * seed)
    u3 = jnp.broadcast_to(jnp.asarray(u_prof, dtype), (ny, nx, nz))
    v3 = jnp.broadcast_to(jnp.asarray(v_prof, dtype), (ny, nx, nz))
    w3 = jnp.zeros((ny, nx, nz + 1), dtype)
    u3, v3, w3 = sl.project(u3, v3, w3, dt=args.dt, g=g)
    tr = jnp.zeros((ny, nx, nz, args.n_tracers), dtype)
    tr = tr.at[..., 0].set(qv_col[None, None, :])
    tr = tr.at[..., 1].set(qc_col[None, None, :])
    # Droplet number is STORED per MASS [1/kg]; the case value is a
    # concentration [1/m^3], so divide by the reference density of the layer it
    # is seeded into.  ``Nc_0`` below stays per volume — that is a scheme
    # parameter, not a transported tracer.
    #
    # MERGE NOTE: main fixed the unit against a hardcoded RF01 constant; this
    # branch had added ASTEX, whose N_c is 100e6 against DYCOMS' 140e6. Keeping
    # main's conversion with the PER-CASE value -- the constant would silently
    # seed ASTEX with RF01's droplet number.
    _nc_per_mass = jnp.where(
        qc_col > 0.0, args.n_c_m3 / jnp.asarray(ref.rho_c, dtype), 0.0)
    tr = tr.at[..., 6].set(_nc_per_mass[None, None, :])
    st = sl.SpectralLESState(
        u=u3, v=v3, w=w3, rhs_u_prev=jnp.zeros_like(u3),
        rhs_v_prev=jnp.zeros_like(v3), rhs_w_prev=jnp.zeros_like(w3),
        theta=th3, rhs_theta_prev=jnp.zeros_like(th3),
        tracers=tr, rhs_tracers_prev=jnp.zeros_like(tr))

    rho_sfc = float(ref.rho_c[0]); exn_sfc = float(ref.exner_c[0])
    th_flux = sfc0["shf"] / (rho_sfc * constants.c_pd * exn_sfc)
    qv_flux = sfc0["lhf"] / (rho_sfc * constants.L_v)
    forc = dict(ug=jnp.asarray(ug, dtype), vg=jnp.asarray(vg, dtype),
                th_flux=th_flux, qv_flux=qv_flux, sfc=sfc0)
    return g, st, ref, forc


def make_stevens_lw(g, ref, dtype):
    """Stevens et al. (2005) RF01 parameterized LW as a θ tendency.

    The flux fit itself is shared with the SCM
    (:mod:`legoesm.atmosphere.physics.radiation.simple_lw`) so the two sides
    of an SCM-vs-LES comparison cannot drift apart. What stays here is the
    LES-specific part: the tracer-slot layout, and the conversion of the
    kernel's TEMPERATURE tendency to the θ tendency this dycore prognoses,
    which is a division by the reference Exner function.

    ``g.z_c`` is passed explicitly rather than re-derived from ``g.z_f``: the
    two agree mathematically on this uniform grid but need not agree in the
    last bit, and the difference would reach the clear-sky term.
    """
    rho = jnp.asarray(ref.rho_c, dtype)            # (nz,)
    exner = jnp.asarray(ref.exner_c, dtype)
    z_c = g.z_c; z_f = g.z_f; dz = g.dz
    cp = _SIMPLE_LW.cp_j_kg_k

    def lw_theta_tendency(tracers):
        # CLOUD liquid only — Stevens/gSAM rad_simple excludes rain from the
        # LW optical depth (codex (c); RF01 is non-drizzling anyway). Slot 0
        # is q_v, so q_t is the sum.
        q_l = tracers[..., 1]
        q_t = tracers[..., 0] + q_l
        dT_dt = simple_lw_temperature_tendency(
            q_l, q_t, rho, dz, z_f, _SIMPLE_LW, z_full=z_c)
        # θ = T/Exner, and the reference Exner is time-invariant here.
        return dT_dt / exner[None, None, :]

    return lw_theta_tendency


def main():
    args = parse_args()
    dtype = jnp.float32 if args.f32 else jnp.float64
    args.output.mkdir(parents=True, exist_ok=True)
    g, st, ref, forc = build(args, dtype)
    # RF01 droplet concentration via the SPECIFIED-Nc path: Morrison's SAM
    # default (predict_Nc=False) uses the constant Nc_0 — the tracer-slot seed
    # alone would be ignored (codex (h)). Nc_0 = 140 cm⁻³ per the RF01 spec.
    micro_cfg = (MicrophysicsConfig(
        scheme="morrison",
        morrison=MorrisonConfig(morrison_flavor="sam", Nc_0=args.n_c_m3))
        if args.microphysics == "morrison"
        else MicrophysicsConfig(scheme=args.microphysics))
    dt0 = float(args.dt)
    micro = make_les_microphysics_fn(micro_cfg, ref, g.dz,
                                     dt0 * args.micro_every)
    lw_tend = make_stevens_lw(g, ref, dtype)

    # Subsidence w_ls = −D·z on θ and q_t (slots 0 + 1); upwind in z.
    # The case's SUBSIDENCE divergence, which is not the longwave's -- see
    # _STRATOCUMULUS_CASES.
    w_ls = (-args.subsidence_divergence_s * g.z_c).astype(dtype)
    dz = g.dz

    def ddz_up(f):
        d = (f[..., 1:] - f[..., :-1]) / dz
        return jnp.concatenate([d, jnp.zeros_like(d[..., :1])], axis=-1)

    zc = g.z_c; z_sp = 0.85 * args.Lz; tau_sp = 30.0
    spc = jnp.where(zc > z_sp, 0.5 * (1.0 - jnp.cos(
        jnp.pi * (zc - z_sp) / (args.Lz - z_sp))), 0.0).astype(dtype)
    zf = g.z_f
    spf = jnp.where(zf > z_sp, 0.5 * (1.0 - jnp.cos(
        jnp.pi * (zf - z_sp) / (args.Lz - z_sp))), 0.0).astype(dtype)

    @partial(jax.jit, static_argnames=("first", "do_micro", "do_rad"))
    def step(state, dt, first=False, do_micro=True, do_rad=True):
        state, us = sl.step(state, g=g, dt=dt,
                            u_geo=(forc["ug"], forc["vg"]), f_cor=args.f_cor,
                            first=first, force=(0.0, 0.0),
                            sfc_theta_flux=forc["th_flux"],
                            sfc_qv_flux=forc["qv_flux"])
        th = state.theta
        tr = state.tracers
        th = th - dt * w_ls * ddz_up(th)
        tr = tr.at[..., 0].set(tr[..., 0] - dt * w_ls * ddz_up(tr[..., 0]))
        tr = tr.at[..., 1].set(tr[..., 1] - dt * w_ls * ddz_up(tr[..., 1]))
        if do_rad:
            th = th + dt * args.rad_every * lw_tend(tr)
        state = state._replace(theta=th, tracers=tr)
        if do_micro:
            dth, dtr, _pr = micro(state.theta, state.tracers)
            trn = state.tracers + dt * args.micro_every * dtr
            wat = trn[..., :6]
            created = jnp.sum(jnp.clip(-wat, 0.0, None))
            trn = jnp.concatenate([jnp.clip(wat, 0.0, None),
                                   jnp.clip(trn[..., 6:], 0.0, None)], axis=-1)
            state = state._replace(
                theta=state.theta + dt * args.micro_every * dth, tracers=trn)
        else:
            created = jnp.asarray(0.0, state.theta.dtype)
        rc = (dt / tau_sp) * spc
        rf = (dt / tau_sp) * spf
        u = state.u - rc * (state.u - state.u.mean((0, 1), keepdims=True))
        v = state.v - rc * (state.v - state.v.mean((0, 1), keepdims=True))
        w = state.w - rf * state.w
        u, v, w = sl.project(u, v, w, dt=dt, g=g)
        return state._replace(u=u, v=v, w=w), us, created

    T = args.hours * 3600.0
    n_steps = int(round(T / dt0))
    print(f"[{args.case.upper()} LES] {args.nx}x{args.ny}x{args.nz} dx={g.dx:.1f} "
          f"dz={g.dz:.1f} dt={dt0:.2f}s {args.time_scheme} "
          f"sgs={'LASD' if args.dynamic else args.sgs_model} "
          f"micro={micro.scheme_name} dtype={dtype.__name__}")
    d0 = _diag(st, g, ref)
    print(f"  IC: LWP={d0['lwp']:.1f} g/m² cloud cover={d0['cloud_cover']:.2f}"
          f" (sat-adjusted θ_l sounding); {n_steps} steps")

    rec = args.record_frames > 0
    zc_np = np.asarray(g.z_c)
    if rec:
        h_idx, h_z = les_record.select_heights(zc_np, args.Lz)
        frame = 0
        _last_rec_h = [None]

        def _save(t_hours):
            nonlocal frame
            les_record.record_frame(
                args.output, frame, t_hours, args.case_label, zc_np,
                np.asarray(st.u), np.asarray(st.v), np.asarray(sl.f2c(st.w)),
                _record_theta_l(st, ref), args.Lx, args.Ly, h_idx, h_z, args.z0,
                qv3=np.asarray(st.tracers[..., 0]),
                qc3=np.asarray(st.tracers[..., 1]),
                qr3=np.asarray(st.tracers[..., 2]),
                rho_z=np.asarray(ref.rho_c))
            frame += 1
            _last_rec_h[0] = t_hours

    dt = jnp.asarray(dt0, dtype)
    if rec:
        _save(0.0)
    t = 0.0; i = 0
    created_tot = 0.0
    # Trailing time-mean of the cloud metrics over the quasi-steady 2nd half
    # (t > 0.5·T): instantaneous cloud cover / LWP fluctuate, so GCSS reports
    # time-means. Sampled at the print cadence (use a smaller --print-every for
    # a denser mean).
    cc_sum = lwp_sum = 0.0; n_cavg = 0
    # 2nd-half time-mean of the parameterized RF01 LW θ-tendency [K/s] — the cloud-driven
    # radiative COOLING that is DYCOMS's turbulence engine. The moist SCM has radiation off,
    # so this profile is prescribed to it as the artifact's ``theta_adv`` (the same SCMForcing
    # channel BOMEX uses for large-scale advection): SCM and LES then see the same non-turbulent
    # θ forcing. lw_tend returns dθ/dt already (÷exner), NEGATIVE at cloud top (∂F/∂z>0 there ⇒
    # −∂F/∂z<0 = cooling), so it enters ``theta_adv`` with the correct cooling sign.
    rad_tend_sum = np.zeros(zc_np.shape[0]); n_rad = 0
    t_cavg0 = 0.5 * T
    next_rec = T / args.record_frames if rec else np.inf
    t0 = time.time()
    while t < T:
        st, us, created = step(
            st, dt, first=(i == 0),
            do_micro=((i + 1) % args.micro_every == 0),
            do_rad=((i + 1) % args.rad_every == 0))
        created_tot += float(created)
        t += dt0; i += 1
        if i % args.print_every == 0:
            mw = float(jnp.max(jnp.abs(st.w)))
            if not np.isfinite(mw) or mw > 1e3:
                print(f"[BLOWUP] step {i} max|w|={mw}"); return 1
            d = _diag(st, g, ref)
            print(f"{i:7d} {t/3600.0:5.2f}h max|w|={mw:5.2f} "
                  f"cc={d['cloud_cover']:.2f} LWP={d['lwp']:6.1f} g/m² "
                  f"zi={d['zi']:5.0f} m u*={float(us):.3f} "
                  f"clip_q={created_tot:.2e}", flush=True)
            if t >= t_cavg0:
                cc_sum += float(d["cloud_cover"]); lwp_sum += float(d["lwp"])
                n_cavg += 1
                rad_tend_sum += np.asarray(lw_tend(st.tracers)).mean((0, 1))
                n_rad += 1
        if rec and t >= next_rec and frame < args.record_frames:
            _save(t / 3600.0); next_rec += T / args.record_frames
    wall = time.time() - t0
    print(f"[DONE] wall={wall:.0f}s  {i/wall:.1f} steps/s")
    # ALWAYS record the terminal state. The initial-state frame consumes one
    # slot of --record-frames, so the in-loop cadence stops one step short of
    # t = T and the old `frame < record_frames` guard was already exhausted
    # here -- every reference silently ended one cadence step early (measured:
    # --record-frames 3 over 0.25 h gave t = 0, 0.083, 0.167 h, never 0.25).
    # Guarded on the time, not the count, so it cannot emit a duplicate frame.
    if rec and (_last_rec_h[0] is None or t / 3600.0 > _last_rec_h[0] + 1.0e-9):
        _save(t / 3600.0)
    d = _diag(st, g, ref)
    cc_avg = cc_sum / n_cavg if n_cavg else float(d["cloud_cover"])
    lwp_avg = lwp_sum / n_cavg if n_cavg else float(d["lwp"])
    np.savez(args.output / "dycoms_les_final.npz", z=zc_np,
             theta=np.asarray(st.theta).mean((0, 1)),
             qv=np.asarray(st.tracers[..., 0]).mean((0, 1)),
             qc=np.asarray(st.tracers[..., 1]).mean((0, 1)),
             cloud_cover=d["cloud_cover"], lwp=d["lwp"], zi=d["zi"],
             cloud_cover_timemean=cc_avg, lwp_timemean=lwp_avg)
    print(f"  FINAL: LWP={d['lwp']:.1f} (2nd-half mean {lwp_avg:.1f}) g/m², "
          f"cloud cover={d['cloud_cover']:.2f} "
          f"(2nd-half mean {cc_avg:.2f}), z_i={d['zi']:.0f} m")
    # Per-case reference values, not one deck's hardcoded into the line: the
    # RF01 targets do not describe ASTEX, and printing them beside an ASTEX
    # result invites the comparison.
    ref_targets = args.case_spec.get("ref_targets")
    print(f"  reference ({args.case}): "
          + (ref_targets if ref_targets else
             "none registered for this deck -- RF01's targets do NOT apply"))
    # Persist the 2nd-half-mean radiative θ-tendency [K/s] the SCM needs as theta_adv.
    rad_tend = (rad_tend_sum / n_rad if n_rad else rad_tend_sum)
    np.savez(args.output / "dycoms_rad_forcing.npz",
             z=zc_np, theta_rad_tend=rad_tend, subsidence_w=np.asarray(w_ls))
    if args.emit_suite_artifact is not None:
        _emit_suite_artifact(args, forc, rad_tend, np.asarray(w_ls),
                             args.emit_suite_artifact)
    return 0


def _resolved_f_cor(args) -> float:
    """The case's Coriolis parameter, from the run if it carries one.

    Argument resolution puts it on ``args``; a caller that builds an artifact
    straight from recorded profiles never went through that, so the case table
    is consulted only in that case.
    """
    value = getattr(args, "f_cor", None)
    if value is not None:
        return float(value)
    return float(
        _STRATOCUMULUS_CASES[_case_key_for_label(args.case_label)]["f_cor"])


def _case_key_for_label(case_label):
    """Map an artifact's case label back to its entry in the case table.

    Labels are the case key with a descriptive suffix (``dycoms_rf01_sc``),
    so match on the leading key rather than requiring an exact name; an
    unrecognised label raises rather than silently picking a case.
    """
    label = str(case_label)
    for key in _STRATOCUMULUS_CASES:
        if label == key or label.startswith(key + "_"):
            return key
    raise KeyError(
        f"case label {label!r} matches no entry in _STRATOCUMULUS_CASES "
        f"({sorted(_STRATOCUMULUS_CASES)}); its Coriolis parameter and "
        f"subsidence cannot be resolved")

def _emit_suite_artifact(args, forc, rad_tend, subsidence_w, out_path):
    """Assemble a moist stratocumulus ``LESReferenceArtifact`` from the ``prof_NNN.npz`` series.

    Mirrors ``run_bomex_les._emit_suite_artifact`` but for DYCOMS-II RF01, whose non-turbulent
    θ forcing is the parameterized RF01 LW COOLING (``rad_tend``, dθ/dt [K/s], negative at cloud
    top) rather than BOMEX's large-scale advective tendency — it is placed in the SAME
    ``theta_adv`` channel so the SCM (radiation off) still sees the LES's cloud-top radiative
    driving. Large-scale subsidence is ``−D·z`` (``subsidence_w``, +up ⇒ negative = sinking).
    There is no prescribed q_v advection in RF01 (subsidence already dries), so ``qv_adv=0``.
    ``theta`` in the moist prof IS θ_l. Signs follow the artifact convention (fluxes +up,
    subsidence_w +up); the cooling sign is verified in ``make_stevens_lw``.
    """
    from legoesm.atmosphere.les_suite.bridge import (  # noqa: E402
        LESReferenceArtifact,
        save_artifact,
    )
    prof_dir = Path(args.output) / "profiles"
    files = sorted(prof_dir.glob("prof_*.npz"))
    if not files:
        raise SystemExit(
            "[emit-suite-artifact] no prof_*.npz in "
            f"{prof_dir} — run with --record-frames > 0")
    frames = [np.load(f) for f in files]
    z = np.asarray(frames[0]["z"], np.float64)
    times = np.array([float(f["t_hours"]) * 3600.0 for f in frames], np.float64)

    def stack(key):
        return np.stack([np.asarray(f[key], np.float64) for f in frames])  # (nt, nz)

    nt = len(frames)
    wtheta = stack("wtheta")
    wqt = stack("wqt")
    sgs = "lasd" if getattr(args, "dynamic", False) else args.sgs_model
    art = LESReferenceArtifact(
        case_name=args.case_label, sgs=sgs,
        heights_m=z, times_s=times,
        theta=stack("theta"),                 # θ_l for the moist prognostic
        u=stack("u"), v=stack("v"),
        wtheta_resolved=wtheta, wtheta_sgs=np.zeros_like(wtheta),
        qt=stack("qt"),
        # Cloud water, so a consumer can tell a cloudy start from a clear one
        # without guessing from saturation; absent for a run that carried none.
        qc=(stack("qc") if "qc" in frames[0] else None),
        wqt_resolved=wqt, wqt_sgs=np.zeros_like(wqt),
        prescribe="fluxes",
        w_theta_s=np.full(nt, float(forc["th_flux"]), np.float64),
        w_qv_s=np.full(nt, float(forc["qv_flux"]), np.float64),
        # The Coriolis parameter is a property of the CASE, not of how the
        # emitter was reached: a caller that builds the artifact from recorded
        # profiles alone never went through argument resolution. Resolved
        # before the call rather than inside a lookup with a default, because
        # a default argument is evaluated whether or not it is needed -- an
        # earlier version raised on any unfamiliar case label even for a run
        # that had the value in hand.
        f_c=_resolved_f_cor(args),
        u_geo=np.asarray(forc["ug"], np.float64),
        v_geo=np.asarray(forc["vg"], np.float64),
        subsidence_w=np.asarray(subsidence_w, np.float64),
        theta_adv=np.asarray(rad_tend, np.float64),   # RF01 LW cooling (dθ/dt, <0 at cloud top)
        qv_adv=np.zeros_like(z))
    save_artifact(art, out_path)
    print(f"[emit-suite-artifact] moist DYCOMS LESReferenceArtifact "
          f"({nt} frames, {z.size} levels, sgs={sgs}) -> {out_path}")


def _diag(st, g, ref):
    qc = np.asarray(st.tracers[..., 1])
    qt = np.asarray(st.tracers[..., 0]) + qc
    rho = np.asarray(ref.rho_c)
    lwp = float((qc * rho[None, None, :]).sum(-1).mean() * g.dz) * 1e3
    cc = float(np.any(qc > 1.0e-5, axis=-1).mean())
    z_c = np.asarray(g.z_c)
    below = qt.mean((0, 1)) >= _QT_INV
    zi = float(z_c[below][-1]) if below.any() else 0.0
    return dict(lwp=lwp, cloud_cover=cc, zi=zi)


if __name__ == "__main__":
    sys.exit(main())
