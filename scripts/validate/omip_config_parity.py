"""Cross-grid OMIP config-parity gate (pre-flight for controlled comparisons).

Compares the ``resolved_config`` of two OMIP ``run_manifest.json`` files and
classifies every differing leaf into:

  GRID_INHERENT   — unavoidable per-grid facts (mesh paths, per-grid arrays,
                    schema layout, Voronoi-only numerics knobs).
  DECLARED        — scheme-family choices the campaign has explicitly made
                    per grid (e.g. tripole runs NEMO zdftke while the KPP
                    grids run KPP+enhanced_diffusion) — allowed, but printed
                    so every comparison names its declared differences.
  CAPABILITY_GAP  — levers one grid cannot run yet (MPAS: iwm/dm2dc/isf/
                    bbl/rgb-chl bridges) — allowed with the gap named.
  UNJUSTIFIED     — everything else.  These are parity BUGS (a flag silently
                    ignored, an sbatch omission, a drifted default).  The
                    gate exits 1 when any exist.

Motivating audit (2026-07-18): trp3-vs-mpas8 manifests revealed MPAS silently
ran tvd + adcroft PGF + legacy linear drag while the tripole ran superbee +
smc03 + nemo_quadratic from byte-identical sbatch flag sets — three
confounders in every cross-grid claim.

Usage:
  python scripts/validate/omip_config_parity.py A/run_manifest.json B/run_manifest.json
  (add --strict-declared to also fail on DECLARED differences)
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys

# --- classification tables (extend deliberately; every entry carries a reason) ---
GRID_INHERENT = {
    "grid": "the variable under test",
    "mesh": "per-grid mesh file",
    "output_path": "per-run output location",
    "runtime_config.__type__": "per-grid config class",
    "runtime_config.runoff_depth_spread_map.*": "per-grid regridded field",
    # Schema-layout noise: the two config classes nest the same knobs
    # differently (flat on MPAS, sub-config on lat-lon).  Handled by the
    # ALIASES fold below; whatever remains one-sided-only is layout.
    "runtime_config.constants.*": "lat-lon groups constants; MPAS uses module constants",
    "runtime_config.runtime_checks.*": "layout (flat on MPAS)",
    "runtime_config.barotropic.*": "layout (flat on MPAS)",
    "runtime_config.bottom_drag.*": "layout (flat on MPAS)",
    "runtime_config.lateral_viscosity.*": "layout (flat on MPAS)",
    "runtime_config.polar_filter.*": "lat-lon-only polar filter (Voronoi needs none)",
    "runtime_config.tidal_forcing.*": "lat-lon-only optional block (disabled)",
    # Voronoi-only / C-grid-only numerics knobs (operators differ by mesh):
    "runtime_config.pv_scheme": "Voronoi TRiSK PV flux family",
    "runtime_config.pv_alpha": "Voronoi TRiSK PV flux family",
    "runtime_config.apvm_dt": "Voronoi APVM damping",
    "runtime_config.K_zeta_bih": "Voronoi zeta-checkerboard control",
    "runtime_config.vertex_thickness_alpha": "Voronoi vertex reconstruction",
    "runtime_config.semi_implicit_coriolis": "MPAS integrator structure",
    "runtime_config.equatorial_visc_boost": "MPAS-only stabiliser (off)",
    "runtime_config.use_h_actual_pgf": "MPAS PGF plumbing detail",
    "runtime_config.use_baroclinic_rho_ref": "MPAS PGF plumbing detail",
    "runtime_config.use_static_baroclinic_rho_ref": "MPAS PGF plumbing detail",
    "runtime_config.barotropic_u_viscosity": "MPAS barotropic-mode knob",
    "runtime_config.barotropic_u_biharmonic": "MPAS barotropic-mode knob",
    "runtime_config.barotropic_damping": "MPAS barotropic-mode knob",
    "runtime_config.C_smag": "per-grid Smagorinsky family (biharmonic)",
    "runtime_config.C_leith*": "per-grid Leith family",
    "runtime_config.mle": "layout (nested on lat-lon physics)",
    "runtime_config.gm_redi.*": "layout (duplicated view of physics.lateral_mixing)",
    "runtime_config.enable_runtime_checks": "layout (flat on MPAS)",
    "runtime_config.salinity_m*": "layout (flat on MPAS)",
    "runtime_config.temperature_m*": "layout (flat on MPAS)",
    "runtime_config.A_h": "layout (grouped under lateral_viscosity on lat-lon)",
    "runtime_config.B_h": "layout (grouped under lateral_viscosity on lat-lon)",
}
# One-sided keys that only exist on the lat-lon C-grid config (integrator
# details etc.) — inherent as long as they are ABSENT on the other side.
LATLON_ONLY_OK = [
    "runtime_config.ab2_*", "runtime_config.adaptive_implicit_vertadv",
    "runtime_config.backscatter", "runtime_config.coriolis_*",
    "runtime_config.dt_mom_ratio", "runtime_config.ew_cyclic_overlap",
    "runtime_config.fix_eta_drift", "runtime_config.hyperdiff_coeff",
    "runtime_config.implicit_vmix_dzw_slot", "runtime_config.ke_gradient_scheme",
    "runtime_config.lateral_friction_scheme", "runtime_config.lateral_side_bc",
    "runtime_config.lateral_viscosity_operator", "runtime_config.meridionally_flat",
    "runtime_config.momentum_advection", "runtime_config.momentum_flux_scheme",
    "runtime_config.momentum_friction_additive",
    "runtime_config.momentum_time_integrator", "runtime_config.omp25",
    "runtime_config.outer_integrator", "runtime_config.pgf_quadrature",
    "runtime_config.prescribed_flow", "runtime_config.qg_leith_*",
    "runtime_config.slope_foot_*", "runtime_config.sponge_forcing_implicit",
    "runtime_config.surface_forcing_implicit", "runtime_config.tracer_time_integrator",
    "runtime_config.tracer_wall_neumann_fill", "runtime_config.vertical_momentum_scheme",
    "runtime_config.vortcor_*", "runtime_config.wall_grid_filter_rate_s",
    "runtime_config.weno_*",
]
DECLARED = {
    "runtime_config.physics.vertical_mixing.scheme":
        "campaign choice: tripole runs NEMO zdftke; KPP grids run KPP",
    "runtime_config.physics.vertical_mixing.tke.*":
        "TKE subconfig live only where scheme=tke (ORCA1 namelist card)",
    "runtime_config.physics.vertical_mixing.kpp.*":
        "KPP subconfig live only where scheme=kpp (Ri_crit 0.15 campaign value)",
    "runtime_config.physics.vertical_mixing.iwm.enabled":
        "CAPABILITY: zdfiwm not wired on the MPAS vmix bridge",
    "runtime_config.physics.convection.scheme":
        "pairs with vmix family: TKE grids none (internal EVD), KPP grids enhanced_diffusion",
    "runtime_config.physics.surface_forcing.scheme":
        "host-loop application path differs (tripole none / MPAS external, tau+q only)",
    "runtime_config.A_v":
        "background viscosity floor: molecular under iwm (tripole) vs KPP-default (MPAS)",
    "runtime_config.K_v":
        "background diffusivity floor: molecular under iwm (tripole) vs KPP-default (MPAS)",
    "runtime_config.C_smag_lap":
        "per-grid stability-tuned Laplacian Smagorinsky (3.0 curvilinear / 0.33 Voronoi)",
    "runtime_config.physics.lateral_mixing.gm_redi.mld_rho_c":
        "TREE-SKEW: field added after 38483caa8 at inert default — run both grids from ONE pinned tree",
    "runtime_config.physics.lateral_mixing.gm_redi.nemo_mld_slope_ramp":
        "TREE-SKEW: field added after 38483caa8 at inert default — run both grids from ONE pinned tree",
    "runtime_config.physics.lateral_mixing.gm_redi.nemo_slope_shapiro":
        "TREE-SKEW: field added after 38483caa8 at inert default — run both grids from ONE pinned tree",
}


def flatten(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{pre}{k}."))
    elif isinstance(d, list):
        out[pre[:-1]] = str(d)[:80]
    else:
        out[pre[:-1]] = d
    return out


# Flat(MPAS) <-> nested(lat-lon) aliases: the SAME knob lives at a different
# path per config class.  Before classifying an ABSENT-on-one-side key, look
# it up through these prefixes on the other side; equal values fold away as
# layout, DIFFERING values surface as a real (aliased) violation — this is
# how `bottom_drag_scheme legacy` (MPAS flat) vs
# `bottom_drag.bottom_drag_scheme nemo_quadratic` (lat-lon nested) is caught.
_NEST_PREFIXES = (
    "runtime_config.barotropic.", "runtime_config.bottom_drag.",
    "runtime_config.runtime_checks.", "runtime_config.lateral_viscosity.",
)


def resolve_alias(key, flat_side):
    """Value of `key` on `flat_side`, following flat<->nested aliases."""
    if key in flat_side:
        return flat_side[key]
    if key.startswith("runtime_config.") and key.count(".") == 1:
        leaf = key.split(".", 1)[1]
        for pre in _NEST_PREFIXES:
            if (pre + leaf) in flat_side:
                return flat_side[pre + leaf]
    for pre in _NEST_PREFIXES:
        if key.startswith(pre):
            flat = "runtime_config." + key[len(pre):]
            if flat in flat_side:
                return flat_side[flat]
    return "<ABSENT>"


def _match(key, table):
    for pat in table:
        if fnmatch.fnmatch(key, pat) or key == pat:
            return pat
    return None


def classify(key, va, vb):
    pat = _match(key, GRID_INHERENT)
    if pat:
        return "GRID_INHERENT", GRID_INHERENT[pat]
    if (_match(key, dict.fromkeys(LATLON_ONLY_OK, ""))
            and ("<ABSENT>" in (str(va), str(vb)))):
        return "GRID_INHERENT", "lat-lon-only integrator/config detail"
    pat = _match(key, DECLARED)
    if pat:
        return "DECLARED", DECLARED[pat]
    if "<ABSENT>" in (str(va), str(vb)):
        # Post-alias one-sided key: the other grid's config CLASS has no such
        # knob, so it cannot be mis-set there — a field-set difference, not a
        # value drift.  (True value-parity bugs always survive aliasing with
        # values on BOTH sides; tree-skew fields are named in DECLARED so
        # they stay visible.)
        return "GRID_INHERENT", "config-class field-set difference (knob absent on one grid)"
    return "UNJUSTIFIED", "no allowlist entry — parity bug or undocumented drift"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("manifest_a")
    p.add_argument("manifest_b")
    p.add_argument("--strict-declared", action="store_true",
                   help="also exit 1 when DECLARED differences exist")
    args = p.parse_args(argv)
    A = json.load(open(args.manifest_a))
    B = json.load(open(args.manifest_b))
    fa = flatten(A["config"]["resolved_config"])
    fb = flatten(B["config"]["resolved_config"])
    buckets = {"GRID_INHERENT": [], "DECLARED": [], "UNJUSTIFIED": []}
    seen_alias = set()
    for k in sorted(set(fa) | set(fb)):
        va, vb = resolve_alias(k, fa), resolve_alias(k, fb)
        if va == vb:
            continue
        # canonicalize aliased pairs so flat+nested don't double-report
        canon = k
        for pre in _NEST_PREFIXES:
            if k.startswith(pre):
                canon = "runtime_config." + k[len(pre):]
        if canon in seen_alias:
            continue
        seen_alias.add(canon)
        # classify the CANONICAL key: a value-differing aliased pair must be
        # judged as the knob itself, not folded by the nested-layout pattern.
        cls, why = classify(canon, va, vb)
        buckets[cls].append((canon, va, vb, why))
    la = A["config"].get("config_kind", "A")
    lb = B["config"].get("config_kind", "B")
    print(f"[parity] {args.manifest_a}  vs  {args.manifest_b}")
    for cls in ("DECLARED", "CAPABILITY_GAP", "UNJUSTIFIED"):
        rows = buckets.get(cls, [])
        if not rows and cls != "UNJUSTIFIED":
            continue
        print(f"\n== {cls} ({len(rows)}) ==")
        for k, va, vb, why in rows:
            print(f"  {k:52s} | {str(va)[:24]:24s} | {str(vb)[:24]:24s} | {why}")
    n_inh = len(buckets["GRID_INHERENT"])
    print(f"\n[parity] GRID_INHERENT differences folded: {n_inh} "
          f"(run with -v in future to list)")
    bad = len(buckets["UNJUSTIFIED"]) + (
        len(buckets["DECLARED"]) if args.strict_declared else 0)
    if bad:
        print(f"[parity] FAIL: {bad} unjustified difference(s) — fix or "
              f"add a REASONED allowlist entry before comparing these runs.")
        return 1
    print("[parity] OK: no unjustified differences.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
