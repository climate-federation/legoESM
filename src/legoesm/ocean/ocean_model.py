"""
Patch for legoesm/ocean/dynamics/ocean_model.py
================================================
Add BGC step after conservation fixer in OceanModel.step().

Apply on Levante:
    python patch_ocean_model.py
"""
from pathlib import Path

MODEL_FILE = Path(
    "/work/bd1083/b309178/diffESM/legoESM/src/legoesm/ocean/dynamics/ocean_model.py"
)

src = MODEL_FILE.read_text()

# ── 1. Add biogeo_cfg to __init__ ───────────────────────────────────
OLD_INIT_END = """        if discretization in _LEGACY_DISCRETIZATION_MAP:
            import warnings
            canonical = _LEGACY_DISCRETIZATION_MAP[discretization]
            warnings.warn(
                f"Ocean discretization {discretization!r} is deprecated; \""""

# Find end of __init__ by looking for the last line before step()
# We add _biogeo_cfg just before step() — find a unique anchor in __init__
OLD_INIT_ANCHOR = "        self.config = config\n"
NEW_INIT_ANCHOR = """        self.config = config
        # BGC config extracted from OceanConfig (None = disabled)
        self._biogeo_cfg = getattr(config, "biogeo", None)
"""
if "_biogeo_cfg" not in src:
    src = src.replace(OLD_INIT_ANCHOR, NEW_INIT_ANCHOR, 1)

# ── 2. Add BGC step after conservation fixer in step() ──────────────
OLD_STEP_RETURN = """        return cast_pytree(state_new, None, "storage")

    def step_checked"""

NEW_STEP_RETURN = """        # --- 6. BGC source/sink step (cubed-sphere only) ---
        if self._biogeo_cfg is not None and state_new.biogeo is not None:
            from legoesm.ocean.biogeochemistry.carbon_cycle import (
                step_ocean_biogeochemistry,
            )
            new_biogeo, _co2_diag = step_ocean_biogeochemistry(
                state=state_new.biogeo,
                T_degC=state_new.T.data,
                S_psu=state_new.S.data,
                dz_ref=self.z_coord.dz_ref,
                z_full_ref=self.z_coord.z_full_ref,
                ocean_mask=state_new.land_mask.data,
                dt=dt,
                cfg=self._biogeo_cfg,
            )
            state_new = state_new._replace(biogeo=new_biogeo)

        return cast_pytree(state_new, None, "storage")

    def step_checked"""

if "BGC source/sink step" not in src:
    src = src.replace(OLD_STEP_RETURN, NEW_STEP_RETURN, 1)

MODEL_FILE.write_text(src)
print(f"Patched: {MODEL_FILE}")
assert "BGC source/sink step" in MODEL_FILE.read_text()
assert "_biogeo_cfg" in MODEL_FILE.read_text()
print("OK: BGC step added to OceanModel.step()")
