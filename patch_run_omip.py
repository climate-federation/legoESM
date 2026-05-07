"""
Patch for legoESM/scripts/run_omip.py
======================================
Add --biogeo CLI flag and BGC initialization for cubed-sphere path.

Apply on Levante:
    python patch_run_omip.py
"""
from pathlib import Path

RUN_FILE = Path(
    "/work/bd1083/b309178/diffESM/legoESM/scripts/run_omip.py"
)

src = RUN_FILE.read_text()

# ── 1. Add --biogeo and --pco2-atm to parse_args() ──────────────────
OLD_PARSE = '    p.add_argument("--diag-every", type=int, default=None,\n                   help="Diagnostic interval in steps (default: ~1 day)")\n    return p.parse_args()'

NEW_PARSE = '    p.add_argument("--diag-every", type=int, default=None,\n                   help="Diagnostic interval in steps (default: ~1 day)")\n    p.add_argument("--biogeo", type=str, default="none",\n                   choices=["none", "abiotic", "npzd"],\n                   help="BGC scheme: none (default), abiotic, or npzd")\n    p.add_argument("--pco2-atm", type=float, default=400.0,\n                   help="Atmospheric pCO2 [uatm] for air-sea CO2 exchange")\n    return p.parse_args()'

if "--biogeo" not in src:
    src = src.replace(OLD_PARSE, NEW_PARSE, 1)

# ── 2. Add BGC init in run_omip_single() after state init ───────────
OLD_SETUP = "    setup_time = time.time() - t_setup\n    print(f\"  Setup: {setup_time:.1f}s\")"

NEW_SETUP = """    # ── BGC initialization (cubed-sphere only) ──────────────────────
    if getattr(args, "biogeo", "none") != "none":
        if grid_type == "cubed_sphere":
            from legoesm.ocean.biogeochemistry import BiogeoConfig, init_biogeo_state
            from legoesm.ocean.dynamics.ocean_model import OceanModel
            biogeo_cfg = BiogeoConfig(
                scheme=args.biogeo,
                pCO2_atm=getattr(args, "pco2_atm", 400.0),
                wind_speed=7.0,
            )
            biogeo_state = init_biogeo_state(
                state.T.data.shape,
                z_coord.z_full_ref,
                biogeo_cfg,
            )
            state = state._replace(biogeo=biogeo_state)
            config = config._replace(biogeo=biogeo_cfg)
            model = OceanModel(grid, z_coord, config)
            print(f"  BGC: scheme={args.biogeo}"
                  f"  pCO2={getattr(args, 'pco2_atm', 400.0):.0f} uatm")
        else:
            print(f"  WARNING: --biogeo ignored for {grid_type} "
                  f"(cubed-sphere only for now)")

    setup_time = time.time() - t_setup
    print(f"  Setup: {setup_time:.1f}s")"""

if "BGC initialization" not in src:
    src = src.replace(OLD_SETUP, NEW_SETUP, 1)

# ── 3. Add DIC/Phyto to _extract_scalars diagnostics ────────────────
OLD_RETURN = "    return {\"SST\": sst, \"SSS\": sss, \"SSH\": ssh, \"max_speed\": max_u if grid_type != \"spectral\" else 0.0}"

NEW_RETURN = """    result = {"SST": sst, "SSS": sss, "SSH": ssh,
              "max_speed": max_u if grid_type != "spectral" else 0.0}

    # BGC surface diagnostics
    if (grid_type not in ("spectral", "mpas")
            and hasattr(state, "biogeo") and state.biogeo is not None):
        import jax.numpy as jnp
        biogeo = state.biogeo
        wet = mask > 0.5
        DIC_surf  = float(np.mean(np.asarray(biogeo.DIC)[..., 0][wet]))
        result["DIC_surf"] = DIC_surf
        if biogeo.Phyto is not None:
            Phyto_surf = float(np.mean(np.asarray(biogeo.Phyto)[..., 0][wet]))
            result["Phyto_surf"] = Phyto_surf

    return result"""

if "DIC_surf" not in src:
    src = src.replace(OLD_RETURN, NEW_RETURN, 1)

RUN_FILE.write_text(src)
print(f"Patched: {RUN_FILE}")
assert "--biogeo" in RUN_FILE.read_text()
assert "BGC initialization" in RUN_FILE.read_text()
print("OK: BGC integration added to run_omip.py")
