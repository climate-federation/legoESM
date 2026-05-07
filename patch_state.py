"""
Patch for legoesm/ocean/state.py
=================================
Add `biogeo` field to OceanState and `biogeo` field to OceanConfig.

Apply on Levante:
    python patch_state.py
"""
from pathlib import Path

STATE_FILE = Path(
    "/work/bd1083/b309178/diffESM/legoESM/src/legoesm/ocean/state.py"
)

src = STATE_FILE.read_text()

# ── 1. Add import for TYPE_CHECKING at top (after existing imports) ──
if "OceanBiogeoState" not in src:
    # Add import after the last 'from' or 'import' line near the top
    old_import = "from __future__ import annotations"
    new_import = """from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from legoesm.ocean.biogeochemistry.config import (
        BiogeoConfig, OceanBiogeoState,
    )"""
    src = src.replace(old_import, new_import, 1)

# ── 2. Add biogeo field to OceanState ───────────────────────────────
old_ocean_state_end = """    u: Field
    v: Field
    T: Field
    S: Field
    eta: Field
    H_bathy: Field
    land_mask: Field"""

new_ocean_state_end = """    u: Field
    v: Field
    T: Field
    S: Field
    eta: Field
    H_bathy: Field
    land_mask: Field
    biogeo: "OceanBiogeoState | None" = None  # BGC tracers; None = disabled"""

src = src.replace(old_ocean_state_end, new_ocean_state_end, 1)

# ── 3. Add biogeo field to OceanConfig ──────────────────────────────
# Find last field in OceanConfig and append biogeo
# OceanConfig ends with several float/bool fields; find a unique one
OLD_CONFIG_TAIL = "    ab2_epsilon: float = 0.1  # AB2 stabilization (MITgcm ABepsBar)"
NEW_CONFIG_TAIL = """    ab2_epsilon: float = 0.1  # AB2 stabilization (MITgcm ABepsBar)
    biogeo: "BiogeoConfig | None" = None  # None = BGC disabled"""

if "biogeo" not in src:
    src = src.replace(OLD_CONFIG_TAIL, NEW_CONFIG_TAIL, 1)

STATE_FILE.write_text(src)
print(f"Patched: {STATE_FILE}")

# Verify
assert "biogeo" in STATE_FILE.read_text(), "FAILED: biogeo not found in state.py"
print("OK: biogeo field added to OceanState and OceanConfig")
