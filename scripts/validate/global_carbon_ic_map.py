#!/usr/bin/env python
"""Drift / realism validator for the global carbon initial-condition map.

Proves the archetype-based finidat (Task 6) starts NEAR equilibrium -- its
total-carbon re-integration drift -> 0 -- versus a cold start, and that its
per-PFT SOC / live-biomass stocks sit in published biome ranges.

Archetype-level validation (deviation from the per-cell brief)
--------------------------------------------------------------
The per-cell finidat ``global_carbon_ic.npz`` stores only per-cell pools +
lat/lon/PFT ids; it does NOT store per-cell climate or soil texture, so a
per-cell coupled config cannot be rebuilt offline without re-loading the raw
CLM5 / ERA5 / HWSD maps (not available on a compute node).  Instead this
validator works from ``archetypes.npz`` (also written by the Task-6 driver),
which carries the full :class:`ArchetypeTable` (``pft_id`` / ``soil_class`` +
the climate features) AND the mapped per-archetype equilibria ``eq_C_*``.

This is INDICATIVE QC, not a per-cell proof.  ``map_to_grid`` is LINEAR in the
IC pools, but re-integrating a MIXED cell is NOT: GPP / LAI / stomatal
conductance / respiration depend nonlinearly on the summed foliar carbon and a
SHARED soil column, so a cell's drift is NOT in general the cover-weighted mix
of its archetypes' drifts.  Small per-archetype drift is therefore a NECESSARY
condition and a strong signal that each archetype equilibrium is stationary --
but it does not by itself prove that every mixed grid cell starts at
equilibrium.  Rigorous per-cell validation (re-integrating a sample of ACTUAL
mixed grid cells) is a documented follow-up.  The archetype set is
self-contained and testable on the dry-run ``archetypes.npz``.

Each archetype is re-integrated ``n_years`` from (a) the MAPPED IC (``eq_C_*``)
and (b) a COLD IC (the pipeline's below-equilibrium seed via
``init_carbon_state``), reusing the SHARED coupled-step construction
(:func:`legoesm.land.carbon.global_init.iter_archetype_batches` /
``make_archetype_step_fn``) and the shared raw forward integrator
(:func:`legoesm.land.carbon.spinup.integrate_annual_pools`) -- no numerics are
re-derived here.

Runs on a compute node (multi-minute JIT of the coupled land step); NOT a
login-node command.  Outputs (under ``--output``, gitignored): ``report.md`` +
``report.json`` + a mapped-vs-cold drift PNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Pure-numpy module surface so ``drift_frac_per_yr`` imports without JAX; the
# heavy JAX / land-model imports are deferred into ``assess_ic_map`` (mirrors
# the Stage-A/B split in ``global_init``).

# Divide-safety floor for the drift fraction (matches ``global_init``).
_DRIFT_EPS = 1e-9
# A well-initialised finidat should drift < this fraction of its total carbon
# per year over a short re-integration; a cold start drifts more.  Secondary
# gate to the primary "mapped median < cold median" comparison.
_DRIFT_PASS_THRESHOLD = 0.05
_G_PER_KG = 1000.0
# CarbonState pool order (== CarbonState._fields; kept literal so the metric
# helpers stay JAX-free at import).  Guarded against drift by
# test_global_carbon_ic_map.test_pool_fields_matches_carbon_state.  SOM is
# resolved into active/slow/passive since phase A1.
_POOL_FIELDS = (
    "C_lab", "C_fol", "C_root", "C_wood", "C_lit",
    "C_som_active", "C_som_slow", "C_som_passive",
)

# Soil-column geometry fallbacks (only used when an OLD archetypes.npz predates
# geometry persistence; a Task-8h npz carries n_layers/soil_depth/dt).  Match the
# build_global_carbon_ic.py driver defaults.
_N_LAYERS_DEFAULT = 10
_SOIL_DEPTH_DEFAULT = 3.0
_DT_DEFAULT = 3600.0


def resolve_geometry(cli_val, stored_val, default_val, name, *, hard):
    """Reconcile a CLI geometry value with the value stored in ``archetypes.npz``.

    * ``cli_val is None`` (flag omitted) -> use the stored value, or ``default_val``
      when the npz predates geometry persistence.
    * ``stored_val is None`` (old npz) -> trust the CLI value.
    * both present and equal -> that value.
    * both present and DIFFERENT -> ``hard=True`` (soil-column geometry:
      ``n_layers`` / ``soil_depth``) raises ``SystemExit`` so a map is never
      re-integrated on a mismatched column; ``hard=False`` (``dt`` is the
      re-integration timestep, not column geometry) warns and honours the CLI.
    """
    if cli_val is None:
        return stored_val if stored_val is not None else default_val
    if stored_val is None:
        return cli_val
    if cli_val == stored_val:
        return stored_val
    if hard:
        raise SystemExit(
            f"--{name} {cli_val} conflicts with the stored map geometry "
            f"{stored_val} in archetypes.npz; the map must be re-integrated on "
            f"the SAME soil column it equilibrated on. Omit --{name} to use the "
            f"stored value.")
    print(f"WARN: --{name} {cli_val} overrides stored {stored_val} (allowed; "
          f"{name} is the re-integration timestep, not soil-column geometry).",
          flush=True)
    return cli_val


def drift_frac_per_yr(series):
    """Mean fractional drift per year of a scalar time series.

    ``(series[-1] - series[0]) / (len(series) - 1) / max(|series[-1]|, eps)``:
    the average year-on-year fractional change normalised by the final
    magnitude.  ~0 for a series starting at equilibrium; large +/- for a cold
    start climbing / collapsing toward equilibrium.  Pure numpy (no model).
    """
    s = np.asarray(series, dtype=float).ravel()
    n = s.shape[0]
    if n < 2:
        return 0.0
    denom = max(abs(float(s[-1])), _DRIFT_EPS)
    return float((s[-1] - s[0]) / (n - 1) / denom)


def _per_pft_realism(pft_id, pft_names, eq_np, ranges, pft_biome):
    """Per-PFT SOC + live-biomass realism vs published biome ranges.

    For each PFT present, the median (across its archetypes) mapped-equilibrium
    SOC and live biomass [kgC/m2] are checked against the biome bracket the PFT
    maps to (``realism_ranges.PFT_BIOME``).  ``bare_soil`` (no biome) reports
    stocks but no pass/fail.
    """
    pft_id = np.asarray(pft_id, int)
    # Total SOM = active + slow + passive (slow/passive inert == 0 in phase A1).
    soc = (eq_np["C_som_active"] + eq_np["C_som_slow"]
           + eq_np["C_som_passive"]) / _G_PER_KG
    biomass = (eq_np["C_lab"] + eq_np["C_fol"]
               + eq_np["C_root"] + eq_np["C_wood"]) / _G_PER_KG
    out: dict = {}
    for p in sorted(set(pft_id.tolist())):
        name = pft_names[p] if p < len(pft_names) else str(p)
        sel = pft_id == p
        soc_med = float(np.median(soc[sel]))
        bio_med = float(np.median(biomass[sel]))
        biome = pft_biome.get(name)
        if biome is None:
            out[name] = dict(
                biome=None, n=int(sel.sum()), soc_kgC=soc_med,
                biomass_kgC=bio_med, soc_range=None, biomass_range=None,
                soc_ok=None, biomass_ok=None, ok=None)
            continue
        sr = ranges[biome]["soc"]
        br = ranges[biome]["biomass"]
        soc_ok = bool(sr[0] <= soc_med <= sr[1])
        bio_ok = bool(br[0] <= bio_med <= br[1])
        out[name] = dict(
            biome=biome, n=int(sel.sum()), soc_kgC=soc_med, biomass_kgC=bio_med,
            soc_range=[float(sr[0]), float(sr[1])],
            biomass_range=[float(br[0]), float(br[1])],
            soc_ok=soc_ok, biomass_ok=bio_ok, ok=bool(soc_ok and bio_ok))
    return out


def assess_ic_map(archetypes_npz_path, *, n_years, dt=None, n_layers=None,
                  soil_depth=None):
    """Re-integrate every archetype from the mapped IC and a cold IC; report
    per-archetype total-carbon drift, the mapped<<cold summary, per-PFT realism
    and coverage.

    Parameters
    ----------
    archetypes_npz_path : path-like
        ``archetypes.npz`` from the Task-6 driver (the ``ArchetypeTable`` +
        ``eq_C_*`` mapped equilibria + ``pft_names`` + the persisted soil-column
        geometry ``n_layers`` / ``soil_depth`` / ``dt``).
    n_years : int
        Forward re-integration years (drift is measured over IC -> year
        ``n_years``).
    dt, n_layers, soil_depth : float / int / float, optional
        Re-integration timestep [s] and soil column (layers, depth [m]).  When
        omitted (``None``) they are READ from ``archetypes.npz`` -- the geometry
        the map equilibrated on -- so the mapped IC is re-integrated on the SAME
        column by default (falling back to the module defaults only for an old
        npz that predates geometry persistence).

    Returns
    -------
    dict
        Per-archetype mapped/cold drift, the mapped<<cold summary, per-PFT
        realism, and coverage (``n_archetypes`` / ``n_pft``).
    """
    # Deferred heavy imports (keep the metric helpers JAX-free at module load).
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm.land.carbon.config import CarbonState
    from legoesm.land.carbon.carbon_cycle import init_carbon_state
    from legoesm.land.carbon.global_init import (
        ArchetypeTable, iter_archetype_batches, make_archetype_step_fn,
    )
    from legoesm.land.carbon.realism_ranges import (
        LITERATURE_BIOME_RANGES, PFT_BIOME,
    )
    from legoesm.land.carbon.spinup import integrate_annual_pools
    from legoesm.land.multilayer_land import init_multilayer_land_state
    from legoesm.land.surface_params import CLM5_PFT_NAMES

    z = np.load(archetypes_npz_path, allow_pickle=True)
    # Reconcile the re-integration geometry with what the map equilibrated on
    # (stored in the npz): a None arg adopts the stored value; an explicit value
    # that CONFLICTS with the stored soil column (n_layers / soil_depth) is a hard
    # error -- so a map is NEVER re-integrated on a mismatched column, whether
    # this is invoked via the CLI (main) or as a direct library call.  dt is the
    # re-integration timestep (not column geometry), so an override is allowed.
    stored_nl = int(z["n_layers"]) if "n_layers" in z.files else None
    stored_sd = float(z["soil_depth"]) if "soil_depth" in z.files else None
    stored_dt = float(z["dt"]) if "dt" in z.files else None
    n_layers = int(resolve_geometry(
        n_layers, stored_nl, _N_LAYERS_DEFAULT, "n-layers", hard=True))
    soil_depth = resolve_geometry(
        soil_depth, stored_sd, _SOIL_DEPTH_DEFAULT, "soil-depth", hard=True)
    dt = resolve_geometry(dt, stored_dt, _DT_DEFAULT, "dt", hard=False)
    table = ArchetypeTable(
        pft_id=np.asarray(z["pft_id"], int),
        mat_k=np.asarray(z["mat_k"], float),
        map_yr=np.asarray(z["map_yr"], float),
        t_seasonal_amp_k=np.asarray(z["t_seasonal_amp_k"], float),
        aridity=np.asarray(z["aridity"], float),
        sw_mean_w=np.asarray(z["sw_mean_w"], float),
        soil_class=np.asarray(z["soil_class"], dtype=object),
    )
    eq_np = {p: np.asarray(z["eq_" + p], float) for p in _POOL_FIELDS}
    pft_names = ([str(s) for s in z["pft_names"]]
                 if "pft_names" in z.files else list(CLM5_PFT_NAMES))
    n_arch = int(table.pft_id.shape[0])

    # Physics switches the archetypes were spun with (canopy + opt-in carbon
    # flags).  Pre-v7 archetype files carry none: those built before
    # 2026-08-20 were spun on the SimpleSEB default surface scheme, later ones
    # on the two-leaf canopy, so no default reproduces both -- the builder's
    # current defaults are used and the ambiguity is printed loudly (codex:
    # provenance-less files cannot be re-integrated on a provably identical
    # model).
    physics = {k: (z[k].item() if k in z.files else d) for k, d in
               (("stomatal_model", "ball_berry"),
                ("nsc_gated_respiration", False),
                ("cold_deciduous_dormancy", False),
                ("leaf_c_resorption_frac", 0.0))}
    missing = [k for k in physics if k not in z.files]
    if missing:
        print("[global_carbon_ic_map] WARNING: archetypes.npz lacks physics "
              f"provenance for {missing} (pre-v7 build); re-integrating on the "
              f"builder defaults {physics} -- drift numbers are NOT a "
              "verification of the model that produced these pools if it "
              "was built before 2026-08-20 (SimpleSEB default then).")
    else:
        print(f"[global_carbon_ic_map] physics switches from archetypes.npz: {physics}")
    batches = iter_archetype_batches(
        table, n_layers=n_layers, soil_depth=soil_depth, dt=dt, **physics)

    mapped_drift = np.full(n_arch, np.nan)
    cold_drift = np.full(n_arch, np.nan)
    mapped_tc0 = np.full(n_arch, np.nan)
    mapped_tc1 = np.full(n_arch, np.nan)
    cold_tc0 = np.full(n_arch, np.nan)
    cold_tc1 = np.full(n_arch, np.nan)

    for batch in batches:
        g_idx = batch.g_idx
        ncol_g = int(g_idx.shape[0])
        # Pass the batch's perennial-frost index so the drift-validator step
        # applies the SAME permafrost/anaerobic SOM protection the equilibrium
        # IC was built with (else the mapped IC would spuriously "drift" as an
        # unprotected step decomposed its protected permafrost SOC).
        step_fn = make_archetype_step_fn(
            batch.config, batch.land_params, dt=dt,
            soil_frozen_fraction=batch.soil_frozen_fraction)
        state0 = init_multilayer_land_state(
            ncol_g, batch.config, T_init=batch.t_init)
        # (a) MAPPED IC: this group's archetype equilibria from the finidat.
        carbon_mapped = CarbonState(
            **{p: jnp.asarray(eq_np[p][g_idx]) for p in _POOL_FIELDS})
        # (b) COLD IC: the pipeline's below-equilibrium seed (init_carbon_state
        # on the batch carbon config; herbaceous groups get no wood).  This is
        # the control -- what a naive cold start (no equilibration) drifts like.
        carbon_cold = init_carbon_state((ncol_g,), batch.config.carbon)

        ann_mapped = integrate_annual_pools(
            step_fn, state0, carbon_mapped, batch.forcing_fn,
            n_years=n_years, steps_per_year=batch.steps_per_year, dt=dt)
        ann_cold = integrate_annual_pools(
            step_fn, state0, carbon_cold, batch.forcing_fn,
            n_years=n_years, steps_per_year=batch.steps_per_year, dt=dt)
        # Total-carbon series (n_years+1, ncol_g).
        tc_mapped = np.asarray(sum(ann_mapped[p] for p in _POOL_FIELDS))
        tc_cold = np.asarray(sum(ann_cold[p] for p in _POOL_FIELDS))
        for col, a in enumerate(g_idx.tolist()):
            mapped_drift[a] = drift_frac_per_yr(tc_mapped[:, col])
            cold_drift[a] = drift_frac_per_yr(tc_cold[:, col])
            mapped_tc0[a] = float(tc_mapped[0, col])
            mapped_tc1[a] = float(tc_mapped[-1, col])
            cold_tc0[a] = float(tc_cold[0, col])
            cold_tc1[a] = float(tc_cold[-1, col])

    mapped_med = float(np.median(np.abs(mapped_drift)))
    cold_med = float(np.median(np.abs(cold_drift)))
    mapped_much_less = bool(mapped_med < cold_med)
    passed = bool(mapped_much_less and mapped_med < _DRIFT_PASS_THRESHOLD)

    realism = _per_pft_realism(
        table.pft_id, pft_names, eq_np, LITERATURE_BIOME_RANGES, PFT_BIOME)

    pft_id_list = [int(x) for x in table.pft_id.tolist()]
    return dict(
        archetypes_path=str(archetypes_npz_path),
        n_archetypes=n_arch,
        n_pft=int(len(set(pft_id_list))),
        n_years=int(n_years), dt=float(dt),
        n_layers=int(n_layers), soil_depth=float(soil_depth),
        drift_pass_threshold=_DRIFT_PASS_THRESHOLD,
        per_archetype=dict(
            pft_id=pft_id_list,
            pft_name=[pft_names[p] if p < len(pft_names) else str(p)
                      for p in pft_id_list],
            soil_class=[str(s) for s in table.soil_class.tolist()],
            mapped_drift_frac_per_yr=[float(x) for x in mapped_drift],
            cold_drift_frac_per_yr=[float(x) for x in cold_drift],
            mapped_total_c0_gC=[float(x) for x in mapped_tc0],
            mapped_total_c1_gC=[float(x) for x in mapped_tc1],
            cold_total_c0_gC=[float(x) for x in cold_tc0],
            cold_total_c1_gC=[float(x) for x in cold_tc1],
        ),
        mapped_median_abs_drift=mapped_med,
        cold_median_abs_drift=cold_med,
        mapped_much_less_than_cold=mapped_much_less,
        passed=passed,
        per_pft_realism=realism,
    )


def make_figure(out_dir: Path, result: dict):
    """Grouped bar chart: |total-C drift| per archetype, cold vs mapped IC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pa = result["per_archetype"]
    md = np.abs(np.asarray(pa["mapped_drift_frac_per_yr"], float)) * 100.0
    cd = np.abs(np.asarray(pa["cold_drift_frac_per_yr"], float)) * 100.0
    x = np.arange(len(md))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(md) + 3.0), 5.0))
    ax.bar(x - w / 2, cd, w, label="cold start", color="#c1543a")
    ax.bar(x + w / 2, md, w, label="mapped IC", color="#3a7cc1")
    ax.axhline(result["drift_pass_threshold"] * 100.0, color="k", ls="--",
               lw=1.0, alpha=0.6,
               label=f"threshold {result['drift_pass_threshold'] * 100:.0f}%/yr")
    ax.set_xlabel("archetype")
    ax.set_ylabel("|total-C drift| [%/yr]")
    ax.set_title(
        f"Mapped-IC vs cold-start re-integration drift "
        f"({result['n_years']} yr)\nmapped median "
        f"{result['mapped_median_abs_drift'] * 100:.2f} vs cold "
        f"{result['cold_median_abs_drift'] * 100:.2f} %/yr")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{i}\n{n}" for i, n in enumerate(pa["pft_name"])],
        fontsize=6, rotation=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mapped_vs_cold_drift.png", dpi=120)
    plt.close(fig)


def write_report(out_dir: Path, result: dict):
    pa = result["per_archetype"]
    L = ["# Global carbon IC map -- drift / realism validation", ""]
    L.append(f"Archetypes file: `{result['archetypes_path']}`  ")
    L.append(f"Re-integration: {result['n_years']} yr, dt={result['dt']:.0f}s, "
             f"{result['n_layers']} soil layers, {result['soil_depth']:.1f} m depth  ")
    L.append(f"Coverage: {result['n_archetypes']} archetypes across "
             f"{result['n_pft']} PFTs")
    L.append("")
    verdict = "PASS" if result["passed"] else "FAIL"
    L.append(f"## Verdict: **{verdict}**")
    L.append("")
    L.append(f"- mapped-IC median |drift| = "
             f"**{result['mapped_median_abs_drift'] * 100:.3f} %/yr**")
    L.append(f"- cold-start median |drift| = "
             f"**{result['cold_median_abs_drift'] * 100:.3f} %/yr**")
    L.append(f"- mapped < cold: {result['mapped_much_less_than_cold']}; "
             f"mapped < threshold "
             f"({result['drift_pass_threshold'] * 100:.0f} %/yr): "
             f"{result['mapped_median_abs_drift'] < result['drift_pass_threshold']}")
    L.append("")
    L.append("## Per-archetype total-carbon drift (mapped IC vs cold start)")
    L.append("")
    L.append("| # | PFT | soil | mapped %/yr | cold %/yr | "
             "mapped total 0->1 kgC | cold total 0->1 kgC |")
    L.append("|---|---|---|---|---|---|---|")
    for i in range(result["n_archetypes"]):
        L.append(
            f"| {i} | {pa['pft_name'][i]} | {pa['soil_class'][i]} | "
            f"{pa['mapped_drift_frac_per_yr'][i] * 100:.2f} | "
            f"{pa['cold_drift_frac_per_yr'][i] * 100:.2f} | "
            f"{pa['mapped_total_c0_gC'][i] / 1000:.1f}->"
            f"{pa['mapped_total_c1_gC'][i] / 1000:.1f} | "
            f"{pa['cold_total_c0_gC'][i] / 1000:.1f}->"
            f"{pa['cold_total_c1_gC'][i] / 1000:.1f} |")
    L.append("")
    L.append("## Per-PFT realism (mapped equilibrium vs published biome ranges)")
    L.append("")
    L.append("| PFT | biome | n | SOC kgC/m2 (range) | "
             "biomass kgC/m2 (range) | ok |")
    L.append("|---|---|---|---|---|---|")
    for name, r in result["per_pft_realism"].items():
        if r["biome"] is None:
            L.append(f"| {name} | -- | {r['n']} | {r['soc_kgC']:.1f} | "
                     f"{r['biomass_kgC']:.1f} | n/a |")
            continue
        sflag = "" if r["soc_ok"] else " (!)"
        bflag = "" if r["biomass_ok"] else " (!)"
        L.append(
            f"| {name} | {r['biome']} | {r['n']} | "
            f"{r['soc_kgC']:.1f} ({r['soc_range'][0]:g}-{r['soc_range'][1]:g})"
            f"{sflag} | "
            f"{r['biomass_kgC']:.1f} "
            f"({r['biomass_range'][0]:g}-{r['biomass_range'][1]:g}){bflag} | "
            f"{'ok' if r['ok'] else '(!)'} |")
    L.append("")
    L.append("_Deviation from the per-cell brief: validated at the ARCHETYPE "
             "level (the finidat stores no per-cell climate/soil to rebuild "
             "per-cell configs offline). `map_to_grid` is LINEAR in the IC "
             "pools, but re-integrating a MIXED cell is nonlinear (GPP / LAI / "
             "stomata / respiration depend nonlinearly on the summed foliar "
             "carbon and a shared soil column), so small per-archetype drift is "
             "INDICATIVE QC -- a necessary condition and strong signal that each "
             "archetype equilibrium is stationary, NOT a per-cell proof. "
             "Rigorous per-cell validation (a sample of actual mixed grid cells) "
             "is a documented follow-up._")
    (out_dir / "report.md").write_text("\n".join(L) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archetypes",
                   default="results/global_carbon_ic/archetypes.npz",
                   help="archetypes.npz written by the Task-6 driver.")
    p.add_argument("--years", type=int, default=3,
                   help="Forward re-integration years for the drift metric.")
    # Geometry flags default to None -> use the geometry stored in archetypes.npz
    # (the column the map equilibrated on); an explicit --n-layers/--soil-depth
    # that CONFLICTS with the stored column is a hard error (resolve_geometry).
    p.add_argument("--dt", type=float, default=None,
                   help="Re-integration timestep [s] (default: stored build dt).")
    p.add_argument("--n-layers", type=int, default=None,
                   help="Soil layers (default: stored build value; conflict=error).")
    p.add_argument("--soil-depth", type=float, default=None,
                   help="Soil depth [m] (default: stored build value; conflict=error).")
    p.add_argument("--output", default="results/global_carbon_ic_validation")
    args = p.parse_args(argv)

    arch_path = Path(args.archetypes)
    if not arch_path.exists():
        raise SystemExit(
            f"--archetypes {arch_path} does not exist "
            f"(run scripts/data/build_global_carbon_ic.py first)")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Geometry (n_layers/soil_depth/dt) defaults to what the map equilibrated on;
    # assess_ic_map reads it from the npz and hard-errors on a conflicting
    # n_layers/soil_depth override (a None flag => adopt the stored value).
    result = assess_ic_map(
        arch_path, n_years=args.years, dt=args.dt,
        n_layers=args.n_layers, soil_depth=args.soil_depth)

    (out_dir / "report.json").write_text(
        json.dumps(result, indent=2, default=float))
    try:
        make_figure(out_dir, result)
    except Exception as exc:  # plotting must never mask the science
        print(f"WARN: figure failed: {exc}", flush=True)
    write_report(out_dir, result)

    verdict = "PASS" if result["passed"] else "FAIL"
    print(
        f"[global_carbon_ic_map] {verdict}: mapped median |drift| "
        f"{result['mapped_median_abs_drift'] * 100:.3f} %/yr vs cold "
        f"{result['cold_median_abs_drift'] * 100:.3f} %/yr "
        f"(threshold {result['drift_pass_threshold'] * 100:.0f} %/yr); "
        f"wrote {out_dir}/report.md, report.json, mapped_vs_cold_drift.png",
        flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
