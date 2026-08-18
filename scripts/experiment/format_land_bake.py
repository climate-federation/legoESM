"""Format a calibrated land JSON into the clm_surface_map.py tuned-constant block
(per-PFT tuples + snow/glacier scalars) for a reviewed paste.

Run: PYTHONPATH=. python scripts/experiment/format_land_bake.py results/land_tuned_fullgrid.json
     PYTHONPATH=. python scripts/experiment/format_land_bake.py results/land/land_tuned_dual_slab.json --tier slab
"""
import argparse
import json


def _tuple(name, vals, per_line=7, ind=len(""), prec=4):
    body = ", ".join(f"{v:.{prec}f}" for v in vals)
    return f"{name} = ({body})"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path")
    ap.add_argument("--tier", choices=["multilayer", "slab"], default="multilayer")
    args = ap.parse_args()
    cal = json.load(open(args.json_path))
    g = lambda k: list(cal[k])
    if args.tier == "slab":
        # slab family: albedo/emissivity/W_max + glacier/snow scalars (the slab's
        # trainable set; root_depth is NOT slab-trainable — no gradient path)
        print("# --- paste into packages/land/legoesm/land/clm_surface_map.py (SLAB) ---\n")
        print(_tuple("_TUNED_PFT_ALBEDO", g("pft_alb")))
        print(_tuple("_TUNED_PFT_EMISSIVITY", g("pft_emis")))
        print(_tuple("_TUNED_PFT_WMAX", g("pft_wmax"), prec=1))
        print(f"TUNED_GLACIER_ALBEDO = {cal['glac_alb']:.4f}")
        print(f"TUNED_SNOW_ALBEDO_MAX = {cal['snow_max']:.4f}")
        print(f"TUNED_CH = {cal['ch']:.6f}")
        return
    wp = g("pft_wp"); fcgap = g("pft_fcgap"); fc = [w + d for w, d in zip(wp, fcgap)]
    print("# --- paste into packages/land/legoesm/land/clm_surface_map.py ---\n")
    print(_tuple("_TUNED_PFT_ALBEDO_MULTILAYER", g("pft_alb")))
    print(_tuple("_TUNED_PFT_EMISSIVITY_MULTILAYER", g("pft_emis")))
    print(_tuple("_TUNED_PFT_ROOT_DEPTH_MULTILAYER", g("pft_root"), prec=3))
    print(_tuple("_TUNED_PFT_Z0_MULTILAYER", g("pft_z0")))
    print(_tuple("_TUNED_PFT_CH_MULTILAYER", g("pft_ch"), prec=6))
    print(_tuple("_TUNED_PFT_KSCALE_MULTILAYER", g("pft_kscale")))
    print(_tuple("_TUNED_PFT_CSCALE_MULTILAYER", g("pft_cscale")))
    print(_tuple("_TUNED_PFT_WP_MULTILAYER", wp))
    print(_tuple("_TUNED_PFT_FC_MULTILAYER", fc))
    print(f"TUNED_GLACIER_ALBEDO_MULTILAYER = {cal['glac_alb']:.4f}")
    print(f"TUNED_SNOW_ALBEDO_MAX_MULTILAYER = {cal['snow_max']:.4f}")
    print(f"TUNED_SNOW_ALBEDO_MIN_MULTILAYER = {cal['snow_min']:.4f}")
    print(f"TUNED_SNOW_DCRIT_MULTILAYER = {cal['snow_dcrit']:.4f}       # kg/m2 for full snow cover")
    print(f"TUNED_SNOW_TAU_DAYS_MULTILAYER = {cal['snow_tau_days']:.4f}   # snow-albedo age e-folding [days]")
    if "soil_dry_boost" in cal:
        print(f"TUNED_SOIL_DRY_BOOST_MULTILAYER = {cal['soil_dry_boost']:.4f}   # CLM dry-soil albedo brightening")
    print(f"TUNED_GLACIER_CBOOST_MULTILAYER = {cal['th_glacier_cboost']:.4f}")
    if "soil_alb_scale" in cal:
        print(f"TUNED_SOIL_ALB_SCALE_MULTILAYER = {cal['soil_alb_scale']:.4f}")
    # Farquhar canopy conductance (dual-target LE calibration, 2026-08): effective-
    # conductance values — do NOT reuse as photosynthetic capacity in the carbon cycle
    # without an identifiability check (Vc_max25/LCMA may be collinear on monthly LE).
    for k, name, prec in (("pft_vcmax", "_TUNED_PFT_VCMAX_MULTILAYER", 2),
                          ("pft_g1", "_TUNED_PFT_G1_MULTILAYER", 3),
                          ("pft_lcma", "_TUNED_PFT_LCMA_MULTILAYER", 2)):
        if k in cal:
            print(_tuple(name, g(k), prec=prec))


if __name__ == "__main__":
    main()
