#!/usr/bin/env python
"""Equilibrium spin-up of the multi-layer land model + DifferLand carbon at a
handful of climate pixels, with a physical-realism and carbon-allocation audit.

For each pixel (tropical rainforest ... arctic tundra) this driver:

  1. Builds a ``MultiLayerLandConfig`` from the pixel's USDA soil texture and
     CLM5 plant functional type, with ``carbon="differland"`` and the coupled
     Farquhar-Ball/Berry stomatal path enabled.
  2. Integrates the FULL coupled land step (soil thermal + Richards hydrology +
     snow + surface energy + carbon) to a repeating-annual-climate equilibrium
     via a nested ``lax.scan`` (inner = sub-daily steps of one year, outer =
     years), reusing the shared synthetic LMIP forcing generator.
  3. Reconstructs the model's *exact* carbon-flux breakdown each step
     (``step_carbon_differland(return_diagnostics=True)`` fed the same
     end-of-step surface temperature and ``root_zone_beta_soil`` the coupled
     step used) to accumulate annual GPP / NPP / Ra / Rh / NEE and the
     foliage/labile/root/wood allocation fluxes.
  4. Audits the result: allocation closure (Σ A == max(NPP,0)), per-pool
     input/output balance at equilibrium, NEE -> 0, and each pool / flux
     against published biome ranges.

Outputs (under ``--output``): a JSON scorecard, per-pixel pool-trajectory and
allocation PNGs, and a Markdown realism report.

Designed to run on a compute node (see the companion sbatch wrapper); it is a
multi-minute JIT-compiled integration, NOT a login-node command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.soil_thermal import SoilThermalConfig
from legoesm.land.richards import RichardsConfig
from legoesm.land.soil_texture import SOIL_TEXTURE_VG
from legoesm.land.surface_params import (
    CLM5_PFT_NAMES,
    _CLM5_PFT_TABLE_RAW,
    PARAM_NAMES,
    is_cold_deciduous,
    is_evergreen,
    is_woody,
)
from legoesm.land.carbon.config import CarbonConfig
from legoesm.land.carbon.stomata import StomataConfig
from legoesm.land.carbon.carbon_cycle import init_carbon_state
from legoesm.land.carbon.realism_ranges import LITERATURE_BIOME_RANGES
from legoesm.land.carbon.spinup import run_semi_analytic_spinup
from legoesm.land.carbon_diagnostics import reconstruct_carbon_diagnostics
from legoesm.land.multilayer_land import (
    step_multilayer_land,
    init_multilayer_land_state,
)

_U_MIN = 1.0
_SECS_PER_DAY = 86400.0
_YEAR_DAYS = 365.0
_DEG2RAD = jnp.pi / 180.0


# ---------------------------------------------------------------------------
# Climate pixel definitions
# ---------------------------------------------------------------------------

# (name, lat_deg, lon_deg, pft, texture, T_init_K, precip_rate[kg/m2/s],
#  freeze_thaw, biome_key)
PIXELS = [
    ("tropical_rainforest",  2.0,  -60.0, "broadleaf_evergreen_tropical",
     "clay_loam", 298.0, 6.0e-5, False, "tropical_forest"),
    ("tropical_savanna",    12.0,   25.0, "c4_grass",
     "sandy_loam", 300.0, 2.0e-5, False, "savanna"),
    ("temperate_deciduous", 45.0,   -85.0, "broadleaf_deciduous_temperate",
     "loam", 283.0, 2.5e-5, False, "temperate_forest"),
    ("temperate_grassland", 40.0,  -100.0, "c3_grass",
     "silt_loam", 285.0, 1.2e-5, False, "grassland"),
    ("boreal_forest",       60.0,   90.0, "needleleaf_evergreen_boreal",
     "loam", 271.0, 1.5e-5, True, "boreal_forest"),
    ("semiarid_shrubland",  33.0,   45.0, "broadleaf_deciduous_boreal_shrub",
     "sandy_loam", 296.0, 1.3e-5, False, "shrubland"),
    ("arctic_tundra",       70.0,  -150.0, "c3_arctic_grass",
     "loam", 266.0, 1.0e-5, True, "tundra"),
    # Larch: the headline death-spiral defect (needleleaf_deciduous_boreal is
    # dead 0/0 in the global build).  Cold-deciduous, so BOTH mechanisms engage.
    ("boreal_larch",        62.0,  100.0, "needleleaf_deciduous_boreal",
     "loam", 269.0, 1.3e-5, True, "boreal_forest"),
]

# Published biome realism ranges (SOC / biomass / GPP / NPP / LAI) are the
# single-source-of-truth table in ``legoesm.land.carbon.realism_ranges``,
# shared with the global-carbon-IC-map validator (scripts/validate/
# global_carbon_ic_map.py).  Aliased to ``LITERATURE`` for this harness's
# existing call sites (``_biome_carbon_init`` / ``assess_pixel``).
LITERATURE = LITERATURE_BIOME_RANGES


def _pft_row(pft: str) -> dict:
    idx = CLM5_PFT_NAMES.index(pft)
    return dict(zip(PARAM_NAMES, _CLM5_PFT_TABLE_RAW[idx]))


def _biome_carbon_init(biome: str, woody: bool, LCMA: float) -> dict:
    """Region-realistic initial carbon pools [gC/m2] for one biome.

    A UNIFORM IC across regions is unphysical (a tundra should not start with a
    tropical forest's 10 kgC/m2 wood + 10 kgC/m2 SOM).  But two rules must hold
    together:

    1. **Region-scale the pools that RETAIN their IC over the spin-up** — the
       slow ones: SOM (~centuries) from the biome's soil-carbon range, and wood
       (~decades) from its biomass range.  These carry the regional signal.
    2. **Seed the FAST, maintenance-bearing pools (foliage/root/labile)
       conservatively, below equilibrium, so the stand GROWS INTO its steady
       state.**  Seeding them near-equilibrium-from-above makes the
       biomass-proportional maintenance respiration exceed a climate-suppressed
       GPP, and the over-seeded biomass dies back through the carbon
       death-spiral's dead attractor (an over-seeded root pool sinks a
       grassland exactly as an over-seeded wood pool sinks a boreal stand).
       Root is seeded modestly and decoupled from the (tree-inclusive) biome
       biomass so a grass pixel is not given a savanna tree's belowground mass.
       These fast pools re-equilibrate in a few years regardless of the seed.

    Foliage is seeded at the biome's mean LAI (fast but sets initial GPP).
    """
    ref = LITERATURE[biome]
    biomass = 0.5 * (ref["biomass"][0] + ref["biomass"][1]) * 1000.0  # gC/m2
    soc = 0.5 * (ref["soc"][0] + ref["soc"][1]) * 1000.0             # gC/m2
    lai0 = 0.5 * (ref["lai"][0] + ref["lai"][1])
    return dict(
        C_lab_init=100.0,                       # fast NSC buffer, grows in
        C_fol_init=max(lai0 * LCMA, 1.0),       # initial LAI == biome mean
        C_root_init=300.0,                      # fast, grows to steady state
        # Slow pools carry the regional IC (below-equilibrium wood so climate-
        # limited stands are not over-seeded into the dead attractor).
        C_wood_init=(0.3 * biomass if woody else 0.0),
        C_lit_init=500.0,
        C_som_init=soc,
    )


def build_pixel_config(pft: str, texture: str, freeze_thaw: bool,
                       n_layers: int, soil_depth: float, biome: str,
                       nsc_gated_respiration: bool = False,
                       cold_deciduous_dormancy: bool = False,
                       nsc_ref_labile_frac: float = CarbonConfig().nsc_ref_labile_frac,
                       r_maint_floor_frac: float = CarbonConfig().r_maint_floor_frac,
                       ) -> MultiLayerLandConfig:
    """MultiLayerLandConfig for one pixel: texture -> hydraulics, PFT ->
    surface + photosynthesis params, DifferLand carbon + Farquhar stomata on.
    Carbon pools are seeded region-realistically (``_biome_carbon_init``).  The
    opt-in high-latitude productivity gates (default off -> byte-identical) are
    PFT-scoped: ``cold_deciduous`` is set from ``is_cold_deciduous(pft)`` so the
    dormancy gate engages only on larch / arctic-grass / boreal-shrub pixels."""
    row = _pft_row(pft)
    woody = is_woody(pft)
    carbon = CarbonConfig(
        scheme="differland", LCMA=row["LCMA"], woody=woody,
        evergreen=is_evergreen(pft),
        nsc_gated_respiration=nsc_gated_respiration,
        cold_deciduous_dormancy=cold_deciduous_dormancy,
        cold_deciduous=is_cold_deciduous(pft),
        nsc_ref_labile_frac=nsc_ref_labile_frac,
        r_maint_floor_frac=r_maint_floor_frac,
        **_biome_carbon_init(biome, woody, row["LCMA"]),
    )
    return MultiLayerLandConfig(
        albedo_land=row["albedo_veg"],
        emissivity_land=row["emissivity"],
        z0_land=row["z0"],
        root_depth=row["root_depth"],
        theta_wp=row["theta_wp"],
        theta_fc=row["theta_fc"],
        bulk_scheme="most",
        snow_albedo_feedback=True,
        soil_grid=SoilGridConfig(
            n_layers=n_layers, total_depth=soil_depth, growth_factor=1.5),
        hydraulics=SoilHydraulicsConfig(**SOIL_TEXTURE_VG[texture]),
        thermal=SoilThermalConfig(enable_freeze_thaw=freeze_thaw),
        richards=RichardsConfig(),
        carbon=carbon,
        stomata=StomataConfig(
            enabled=True, stomata_model="ball_berry",
            Vc_max25=row["Vc_max25"], g1_bb=row["g1"]),
    )


# ---------------------------------------------------------------------------
# Semi-analytic equilibrium integration (shared driver)
# ---------------------------------------------------------------------------


def run_pixel(config: MultiLayerLandConfig, lat_deg: float, lon_deg: float,
              T_init: float, precip_rate: float, n_spinup: int, dt: float,
              n_verify: int = 30):
    """Spin one pixel to a VERIFIED soil-carbon equilibrium.

    A single soil-C pool with ~270-yr turnover needs millennia to equilibrate
    by brute force.  Instead this uses the standard **semi-analytic spin-up**
    (Xia et al. 2012, GMD; cf. accelerated decomposition, Thornton & Rosenbloom
    2005; Koven et al. 2013):

    1. Spin up ``n_spinup`` years so the fast pools (labile/foliage/root/litter)
       and wood equilibrate and the mean annual carbon fluxes (NPP, allocation,
       litterfall, temperature-modified turnover) become stationary.
    2. Solve the LINEAR slow-pool steady state analytically from those mean
       fluxes — for a first-order pool ``dC/dt = I - k·C``, the turnover ``k``
       is C-independent, so ``C_eq = I / k = C_spinup · (I_annual / loss_annual)``
       — and reset wood + SOM to it.
    3. Run ``n_verify`` more years to CONFIRM the analytic equilibrium holds
       (the reported drift → 0); the returned diagnostics are this verified
       equilibrium segment.

    Returns (annual_diag over the verify segment, final_state, final_carbon).
    """
    from legoesm.land.lmip_forcing import make_synthetic_lmip_forcing

    lat_rad = float(lat_deg * _DEG2RAD)
    lon_rad = float(lon_deg * _DEG2RAD)
    lat_jnp = jnp.asarray([lat_rad])
    steps_per_year = int(round(_SECS_PER_DAY * _YEAR_DAYS / dt))

    grid = make_soil_grid(config.soil_grid)
    # Root density weights (scalar-param path, matches step_multilayer_land).
    root_frac = jnp.exp(-grid.z_node / config.root_depth)
    root_frac = root_frac / jnp.sum(root_frac)
    theta_wp = config.theta_wp
    theta_fc = config.theta_fc
    beta_min = config.beta_min

    state0 = init_multilayer_land_state(1, config, T_init=T_init)
    carbon0 = init_carbon_state((1,), config.carbon)

    def forcing_fn(doy, hour):
        return make_synthetic_lmip_forcing(
            lat_rad, lon_rad, doy, hour, precip_rate=precip_rate)

    def step_fn(state, carbon, forcing, doy):
        new_state, _response, carbon_new = step_multilayer_land(
            state, forcing, config, _U_MIN, dt,
            lat=lat_jnp, carbon_state=carbon, doy=doy)
        # Reconstruct the carbon-flux breakdown from the end-of-step soil state
        # via the shared helper (same routine run_lmip's spin-up uses); the
        # shared driver derives the model's NEE from the closed-column mass
        # balance, so the discarded TileResponse is not needed here.
        diag = reconstruct_carbon_diagnostics(
            new_state, forcing, carbon, config, root_frac, theta_wp, theta_fc,
            beta_min, lat_jnp, doy, dt, spatial=False)
        return new_state, carbon_new, diag

    # The three-phase transient -> analytic slow-pool reset -> verify loop lives
    # in the shared driver (legoesm.land.carbon.spinup); this validator and the
    # batched archetype map (global_init.equilibrate_archetypes) do not
    # re-derive it.  It returns per-verify-year annual diagnostics.
    final_state, final_carbon, annual, _reset_fluxes = run_semi_analytic_spinup(
        step_fn, state0, carbon0, forcing_fn,
        n_spinup=n_spinup, n_verify=n_verify, steps_per_year=steps_per_year,
        dt=dt, cwd_humification_eff=config.carbon.cwd_humification_eff,
        f_active_to_slow=config.carbon.f_active_to_slow,
        f_slow_to_passive=config.carbon.f_slow_to_passive)

    annual = {k: np.asarray(v).reshape(n_verify, -1).squeeze()
              for k, v in annual.items()}
    return annual, final_state, final_carbon


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

def assess_pixel(name: str, biome: str, annual: dict, final_carbon) -> dict:
    """Turn one pixel's annual arrays into an audit record."""
    last = -1
    ny = annual["gpp"].shape[0]
    # Per-year, per-step means where useful.
    nsteps = float(annual["nsteps"][last])
    gpp = float(annual["gpp"][last])
    npp = float(annual["npp"][last])
    ra = float(annual["r_auto"][last])
    rh = float(annual["r_het"][last])
    nee = float(annual["nee"][last])
    nee_model = float(annual["nee_model"][last])
    lai_max = float(annual["lai_max"][last])
    lai_mean = float(annual["lai_sum"][last] / max(nsteps, 1.0))
    alloc_resid = float(annual["alloc_resid"][last])

    pools = {k: float(annual[k][last]) / 1000.0  # kgC/m2
             for k in ("C_lab", "C_fol", "C_root", "C_wood", "C_lit")}
    # Total SOM = active + slow + passive (slow/passive inert == 0 in phase A1);
    # the downstream realism/equilibrium diagnostics use bulk SOM.
    pools["C_som"] = float(
        annual["C_som_active"][last] + annual["C_som_slow"][last]
        + annual["C_som_passive"][last]) / 1000.0
    biomass = pools["C_lab"] + pools["C_fol"] + pools["C_root"] + pools["C_wood"]
    total_c = biomass + pools["C_lit"] + pools["C_som"]

    # Allocation fractions (of positive NPP).
    a = {k: float(annual[k][last]) for k in ("a_fol", "a_lab", "a_root", "a_wood")}
    a_tot = sum(a.values())
    alloc_frac = {k: (v / a_tot if a_tot > 0 else 0.0) for k, v in a.items()}

    # Equilibrium indicators: input/output ratio per structural pool
    # (==1 at steady state).  Wood in == a_wood, wood out == wood_litter, etc.
    def _ratio(inp, out):
        inp, out = float(annual[inp][last]), float(annual[out][last])
        return inp / out if out > 1e-9 else float("nan")
    wood_ratio = _ratio("a_wood", "wood_litter")
    som_in = float(annual["lit_to_som"][last]) + float(annual["wood_to_som"][last])
    som_out = float(annual["r_het_som"][last])
    som_ratio = som_in / som_out if som_out > 1e-9 else float("nan")
    # Analytic steady-state estimates (linear pool: C_eq = C_now * in/out).
    C_wood_eq = pools["C_wood"] * wood_ratio if np.isfinite(wood_ratio) else float("nan")
    C_som_eq = pools["C_som"] * som_ratio if np.isfinite(som_ratio) else float("nan")

    # Drift over the last min(20, ny//2) years (fractional per year) for total C.
    win = max(2, min(20, ny // 2))
    tc_series = (annual["C_lab"] + annual["C_fol"] + annual["C_root"]
                 + annual["C_wood"] + annual["C_lit"]
                 + annual["C_som_active"] + annual["C_som_slow"]
                 + annual["C_som_passive"]) / 1000.0
    tc0, tc1 = float(tc_series[-win]), float(tc_series[-1])
    total_c_drift_frac_per_yr = (tc1 - tc0) / (win * max(tc1, 1e-9))

    ref = LITERATURE[biome]
    def _in(val, rng):
        return rng[0] <= val <= rng[1]
    realism = {
        "gpp": dict(value=gpp, range=ref["gpp"], ok=_in(gpp, ref["gpp"])),
        "npp": dict(value=npp, range=ref["npp"], ok=_in(npp, ref["npp"])),
        "biomass_kgC": dict(value=biomass, range=ref["biomass"], ok=_in(biomass, ref["biomass"])),
        "soc_kgC": dict(value=pools["C_som"], range=ref["soc"], ok=_in(pools["C_som"], ref["soc"])),
        "lai_max": dict(value=lai_max, range=ref["lai"], ok=_in(lai_max, ref["lai"])),
    }

    # Physical-consistency checks (hard, not literature).  These use only the
    # model's EXACT quantities (allocation residual, pool signs, GPP≥NPP), NOT
    # the reconstructed NEE (see nee_diag_rel_mismatch below).
    is_herbaceous = any(g in name for g in ("grass", "savanna", "tundra"))
    checks = {
        "allocation_closes": alloc_resid < 1e-6 * max(abs(npp), 1.0),
        "npp_le_gpp": npp <= gpp + 1e-9,
        "ra_nonneg": ra >= -1e-9,
        "rh_nonneg": rh >= -1e-9,
        "pools_nonneg": all(v >= -1e-12 for v in pools.values()),
        # Herbaceous PFTs should not build a large woody pool.
        "herbaceous_no_wood": (pools["C_wood"] < 1.0) if is_herbaceous else True,
    }
    # INFORMATIONAL (not a hard gate): the diagnostic GPP/NPP/allocation
    # breakdown is RECONSTRUCTED (end-of-step surface T + root_zone_beta_soil),
    # while the coupled model computes GPP inside its surface-energy solve
    # (surface_out.gpp, at its own iterated surface temperature).  The
    # reconstructed GPP/NPP are close, but NEE is a small residual of large
    # fluxes (GPP − Ra − Rh), so a few-percent GPP error amplifies into a large
    # relative NEE mismatch (worst for low-flux biomes).  The POOLS, SOM, and
    # NEE reported for the equilibrium are the model's EXACT values; this number
    # only flags a grossly diverged reconstruction.
    nee_diag_rel_mismatch = abs(nee - nee_model) / max(abs(nee_model), 1.0)

    return dict(
        nee_diag_rel_mismatch=nee_diag_rel_mismatch,
        name=name, biome=biome, n_years=ny,
        gpp=gpp, npp=npp, r_auto=ra, r_het=rh, nee=nee, nee_model=nee_model,
        cue=(npp / gpp if gpp > 1e-9 else float("nan")),
        lai_max=lai_max, lai_mean=lai_mean,
        pools_kgC=pools, biomass_kgC=biomass, total_c_kgC=total_c,
        alloc_frac=alloc_frac, alloc_resid=alloc_resid,
        wood_in_out_ratio=wood_ratio, som_in_out_ratio=som_ratio,
        C_wood_eq_kgC=C_wood_eq, C_som_eq_kgC=C_som_eq,
        total_c_drift_frac_per_yr=total_c_drift_frac_per_yr,
        is_herbaceous=is_herbaceous,
        realism=realism, checks=checks,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(out_dir: Path, results: list[dict], annuals: dict[str, dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Pool trajectories (one subplot per pixel).
    npix = len(results)
    fig, axes = plt.subplots(2, (npix + 1) // 2, figsize=(4 * ((npix + 1) // 2), 7))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, results):
        a = annuals[r["name"]]
        yrs = np.arange(a["gpp"].shape[0])
        for pool, lab in [("C_fol", "foliage"), ("C_root", "root"),
                          ("C_wood", "wood"), ("C_lit", "litter")]:
            ax.plot(yrs, a[pool] / 1000.0, label=lab, lw=1.3)
        # Total SOM = active + slow + passive (the passive pool dominates).
        som = a["C_som_active"] + a["C_som_slow"] + a["C_som_passive"]
        ax.plot(yrs, som / 1000.0, label="SOM", lw=1.3)
        ax.set_title(f"{r['name']}\n({r['biome']})", fontsize=8)
        ax.set_xlabel("year"); ax.set_ylabel("kgC/m²")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.legend(fontsize=6, ncol=2)
    for ax in axes[npix:]:
        ax.axis("off")
    fig.suptitle("Carbon-pool spin-up trajectories", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "pool_trajectories.png", dpi=120)
    plt.close(fig)

    # 2. GPP / NPP / NEE bars + realism ranges.
    fig, ax = plt.subplots(figsize=(11, 5))
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["gpp"] for r in results], 0.35, label="GPP")
    ax.bar(x + 0.2, [r["npp"] for r in results], 0.35, label="NPP")
    for i, r in enumerate(results):
        g = r["realism"]["gpp"]["range"]; n = r["realism"]["npp"]["range"]
        ax.plot([i - 0.2, i - 0.2], g, color="k", lw=2, alpha=0.5)
        ax.plot([i + 0.2, i + 0.2], n, color="0.4", lw=2, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("gC/m²/yr"); ax.legend()
    ax.set_title("Annual GPP / NPP vs published ranges (bars=model, lines=literature)")
    fig.tight_layout(); fig.savefig(out_dir / "gpp_npp.png", dpi=120); plt.close(fig)

    # 3. Allocation fractions (stacked).
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(names))
    for comp, color in [("a_fol", "#4c9f70"), ("a_lab", "#a7d08c"),
                        ("a_root", "#8b5a2b"), ("a_wood", "#5b3a29")]:
        vals = np.array([r["alloc_frac"][comp] for r in results])
        ax.bar(names, vals, bottom=bottom, label=comp.replace("a_", ""), color=color)
        bottom += vals
    ax.set_ylabel("fraction of NPP"); ax.legend()
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_title("NPP allocation fractions (foliage / labile / root / wood)")
    fig.tight_layout(); fig.savefig(out_dir / "allocation.png", dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(out_dir: Path, results: list[dict]):
    lines = ["# Land carbon equilibrium — realism & allocation audit", ""]
    n_fail = 0
    lines.append("## Physical-consistency checks (hard)")
    lines.append("")
    lines.append("| pixel | alloc closes | NPP≤GPP | herb no-wood | pools≥0 | diag Δ% |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        c = r["checks"]
        def _m(b):
            return "✅" if b else "❌"
        for b in c.values():
            if not b:
                n_fail += 1
        lines.append(
            f"| {r['name']} | {_m(c['allocation_closes'])} | "
            f"{_m(c['npp_le_gpp'])} | {_m(c['herbaceous_no_wood'])} | "
            f"{_m(c['pools_nonneg'])} | {r['nee_diag_rel_mismatch']*100:.0f} |")
    lines.append("")
    lines.append("## Realism vs published biome ranges")
    lines.append("")
    lines.append("| pixel | GPP | NPP | CUE | LAImax | biomass kgC | SOC kgC | drift %/yr |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        rl = r["realism"]
        def _v(key, unit=""):
            d = rl[key]; mark = "" if d["ok"] else " ⚠️"
            return f"{d['value']:.0f}{unit}{mark}"
        lines.append(
            f"| {r['name']} | {_v('gpp')} [{rl['gpp']['range'][0]}-{rl['gpp']['range'][1]}] "
            f"| {_v('npp')} | {r['cue']:.2f} | {r['realism']['lai_max']['value']:.1f} "
            f"| {r['biomass_kgC']:.1f} | {r['pools_kgC']['C_som']:.1f} "
            f"| {r['total_c_drift_frac_per_yr']*100:.2f} |")
    lines.append("")
    lines.append("## Equilibrium (input/output ratio; 1.0 = steady state)")
    lines.append("")
    lines.append("| pixel | wood in/out | C_wood_eq kgC | SOM in/out | C_som_eq kgC |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['wood_in_out_ratio']:.2f} | {r['C_wood_eq_kgC']:.1f} "
            f"| {r['som_in_out_ratio']:.2f} | {r['C_som_eq_kgC']:.1f} |")
    lines.append("")
    lines.append(f"**Total hard-check failures: {n_fail}**")
    (out_dir / "report.md").write_text("\n".join(lines))
    return n_fail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spinup-years", type=int, default=200,
                   help="Transient spin-up years before the analytic slow-pool "
                        "solve (fast pools + wood must stationarise).")
    p.add_argument("--verify-years", type=int, default=40,
                   help="Verification years after the analytic equilibrium "
                        "reset (reported drift should -> 0).")
    p.add_argument("--dt", type=float, default=3600.0)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--soil-depth", type=float, default=3.0)
    p.add_argument("--output", default="results/land_carbon_equilibrium")
    p.add_argument("--only", default=None,
                   help="Run only this pixel name (debug).")
    p.add_argument("--nsc-gated-respiration", action="store_true",
                   help="Enable NSC-gated maintenance respiration (arctic "
                        "productivity rescue Mechanism 1) for every pixel.")
    p.add_argument("--cold-deciduous-dormancy", action="store_true",
                   help="Enable cold-deciduous freeze dormancy (Mechanism 2); "
                        "PFT-scoped via is_cold_deciduous, so it only affects "
                        "larch / arctic-grass / boreal-shrub pixels.")
    p.add_argument("--nsc-ref-labile-frac", type=float,
                   default=CarbonConfig().nsc_ref_labile_frac,
                   help="NSC-gate reference: labile as a fraction of live biomass "
                        "at which R_maint is unthrottled (tune to avoid biting "
                        "healthy plants).")
    p.add_argument("--r-maint-floor-frac", type=float,
                   default=CarbonConfig().r_maint_floor_frac,
                   help="NSC-gate floor: basal R_maint fraction at full depletion.")
    args = p.parse_args(argv)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pixels = [px for px in PIXELS if args.only is None or px[0] == args.only]
    if not pixels:
        raise SystemExit(f"--only {args.only!r} matched no pixel of "
                         f"{[p[0] for p in PIXELS]}")

    results, annuals = [], {}
    for (name, lat, lon, pft, texture, T_init, precip, ft, biome) in pixels:
        print(f"[{name}] pft={pft} texture={texture} lat={lat} "
              f"T_init={T_init} precip={precip:.1e} freeze_thaw={ft}", flush=True)
        config = build_pixel_config(
            pft, texture, ft, args.n_layers, args.soil_depth, biome,
            nsc_gated_respiration=args.nsc_gated_respiration,
            cold_deciduous_dormancy=args.cold_deciduous_dormancy,
            nsc_ref_labile_frac=args.nsc_ref_labile_frac,
            r_maint_floor_frac=args.r_maint_floor_frac)
        annual, _fs, final_carbon = run_pixel(
            config, lat, lon, T_init, precip, args.spinup_years, args.dt,
            n_verify=args.verify_years)
        rec = assess_pixel(name, biome, annual, final_carbon)
        results.append(rec)
        annuals[name] = annual
        print(f"  GPP={rec['gpp']:.0f} NPP={rec['npp']:.0f} "
              f"biomass={rec['biomass_kgC']:.1f}kgC SOM={rec['pools_kgC']['C_som']:.1f}kgC "
              f"LAImax={rec['lai_max']:.1f} alloc_resid={rec['alloc_resid']:.2e} "
              f"wood%={rec['alloc_frac']['a_wood']*100:.0f} "
              f"C_som_eq={rec['C_som_eq_kgC']:.0f}kgC", flush=True)

    (out_dir / "scorecard.json").write_text(json.dumps(results, indent=2, default=float))
    try:
        make_plots(out_dir, results, annuals)
    except Exception as exc:  # plotting must never mask the science
        print(f"WARN: plotting failed: {exc}", flush=True)
    n_fail = write_report(out_dir, results)
    print(f"\nWrote {out_dir}/scorecard.json, report.md, *.png "
          f"({n_fail} hard-check failures)", flush=True)


if __name__ == "__main__":
    main()
