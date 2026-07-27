"""Format a calibrated multilayer-land JSON into the clm_surface_map.py _TUNED_*_MULTILAYER
constant block (per-PFT tuples + snow/glacier scalars) for a reviewed paste.

Run: PYTHONPATH=. python scripts/experiment/format_land_bake.py results/land_tuned_fullgrid.json
"""
import json
import sys


def _tuple(name, vals, per_line=7, ind=len(""), prec=4):
    body = ", ".join(f"{v:.{prec}f}" for v in vals)
    return f"{name} = ({body})"


def main():
    cal = json.load(open(sys.argv[1]))
    g = lambda k: list(cal[k])
    wp = g("pft_wp"); fcgap = g("pft_fcgap"); fc = [w + d for w, d in zip(wp, fcgap)]
    print("# --- paste into packages/land/legoesm/land/clm_surface_map.py ---\n")
    print(_tuple("_TUNED_PFT_ALBEDO_MULTILAYER", g("pft_alb")))
    print(_tuple("_TUNED_PFT_EMISSIVITY_MULTILAYER", g("pft_emis")))
    print(_tuple("TUNED_PFT_ROOT_DEPTH_MULTILAYER", g("pft_root"), prec=3))
    print(_tuple("_TUNED_PFT_Z0_MULTILAYER", g("pft_z0")))
    print(_tuple("_TUNED_PFT_CH_MULTILAYER", g("pft_ch"), prec=6))
    print(_tuple("_TUNED_PFT_KSCALE_MULTILAYER", g("pft_kscale")))
    print(_tuple("_TUNED_PFT_CSCALE_MULTILAYER", g("pft_cscale")))
    print(_tuple("TUNED_PFT_WP_MULTILAYER", wp))
    print(_tuple("TUNED_PFT_FC_MULTILAYER", fc))
    print(f"TUNED_GLACIER_ALBEDO_MULTILAYER = {cal['glac_alb']:.4f}")
    print(f"TUNED_SNOW_ALBEDO_MAX_MULTILAYER = {cal['snow_max']:.4f}")
    print(f"TUNED_SNOW_ALBEDO_MIN_MULTILAYER = {cal['snow_min']:.4f}")
    print(f"TUNED_SNOW_DCRIT_MULTILAYER = {cal['snow_dcrit']:.4f}       # kg/m2 for full snow cover")
    print(f"TUNED_SNOW_TAU_DAYS_MULTILAYER = {cal['snow_tau_days']:.4f}   # snow-albedo age e-folding [days]")
    if "soil_dry_boost" in cal:
        print(f"TUNED_SOIL_DRY_BOOST_MULTILAYER = {cal['soil_dry_boost']:.4f}   # CLM dry-soil albedo brightening")
    print(f"TUNED_GLACIER_CBOOST_MULTILAYER = {cal['th_glacier_cboost']:.4f}")


if __name__ == "__main__":
    main()
