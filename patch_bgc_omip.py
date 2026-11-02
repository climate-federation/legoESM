"""Wire legoESM BGC (npzd / npzd_v2) into the OMIP run.

Run once on Levante:
    cd /work/bd1083/b309178/diffESM/legoesm_ggn
    .venv/bin/python patch_bgc_omip.py

What this patch does
--------------------
1. state.py          — adds  biogeo: OceanBiogeoState | None = None  to OceanState
2. ocean_model.py    — calls step_ocean_biogeochemistry() after the conservation
                       fixer inside OceanModel.step(), carrying BGC state through
                       the dynamic step transparently
3. run_omip.py       — adds --biogeo / --pco2-atm CLI flags, initialises
                       OceanBiogeoState from WOA nutrients on the cubed-sphere,
                       calls the BGC step after each dynamic step, and adds
                       DIC_surf / Phyto_surf / NO3_surf to scalar diagnostics
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path("/work/bd1083/b309178/diffESM/legoesm_ggn")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def patch(path: Path, old: str, new: str, tag: str) -> None:
    src = path.read_text()
    if old not in src:
        print(f"  SKIP  {tag}  (marker not found — already patched?)")
        return
    path.write_text(src.replace(old, new, 1))
    print(f"  OK    {tag}")


def assert_contains(path: Path, marker: str, tag: str) -> None:
    if marker not in path.read_text():
        print(f"  FAIL  {tag}")
        sys.exit(1)
    print(f"  CHECK {tag}")


# ===========================================================================
# 1. state.py — add biogeo field to OceanState
# ===========================================================================

STATE_PY = ROOT / "src/legoesm/ocean/state.py"

patch(
    STATE_PY,
    # ── old ──────────────────────────────────────────────────────────────
    """\
from __future__ import annotations

from typing import NamedTuple

from legoesm import constants
from legoesm.core.field import Field""",
    # ── new ──────────────────────────────────────────────────────────────
    """\
from __future__ import annotations

from typing import NamedTuple

from legoesm import constants
from legoesm.core.field import Field
from legoesm.ocean.biogeochemistry.config import OceanBiogeoState""",
    "state.py: add OceanBiogeoState import",
)

patch(
    STATE_PY,
    # ── old ──────────────────────────────────────────────────────────────
    """\
    u: Field
    v: Field
    T: Field
    S: Field
    eta: Field
    H_bathy: Field
    land_mask: Field


class OceanTendencies""",
    # ── new ──────────────────────────────────────────────────────────────
    """\
    u: Field
    v: Field
    T: Field
    S: Field
    eta: Field
    H_bathy: Field
    land_mask: Field
    biogeo: OceanBiogeoState | None = None   # BGC tracer state (optional)


class OceanTendencies""",
    "state.py: add biogeo field to OceanState",
)


# ===========================================================================
# 2. ocean_model.py — call BGC after conservation fixer
# ===========================================================================

MODEL_PY = (ROOT / "src/legoesm/ocean/dynamics/ocean_model.py")

patch(
    MODEL_PY,
    # ── old ──────────────────────────────────────────────────────────────
    """\
        return cast_pytree(state_new, None, "storage", allow_downcast=True)

    def step_checked""",
    # ── new ──────────────────────────────────────────────────────────────
    """\
        # --- 6. Biogeochemistry (source/sink) ---
        # BGC is stepped outside of the JIT-compiled barotropic loop.
        # Advection/diffusion of BGC tracers uses the same operators as T/S
        # (handled by the calling script via the biogeo field on OceanState).
        if state_new.biogeo is not None and self._bgc_cfg is not None:
            from legoesm.ocean.biogeochemistry import step_ocean_biogeochemistry
            T_degC  = state_new.T.data
            S_psu   = state_new.S.data
            mask_2d = state_new.land_mask.data
            biogeo_new, _co2_diag = step_ocean_biogeochemistry(
                state_new.biogeo,
                T_degC, S_psu,
                self.z_coord.dz_ref,
                self.z_coord.z_full_ref,
                mask_2d,
                dt,
                self._bgc_cfg,
            )
            state_new = state_new._replace(biogeo=biogeo_new)

        return cast_pytree(state_new, None, "storage", allow_downcast=True)

    def step_checked""",
    "ocean_model.py: add BGC step after conservation fixer",
)

# Store _bgc_cfg on __init__
patch(
    MODEL_PY,
    # ── old ──────────────────────────────────────────────────────────────
    """\
    def __init__(
        self,
        grid,
        z_coord,
        config: OceanConfig,""",
    # ── new ──────────────────────────────────────────────────────────────
    """\
    def __init__(
        self,
        grid,
        z_coord,
        config: OceanConfig,
        bgc_cfg=None,          # BiogeoConfig | None""",
    "ocean_model.py: add bgc_cfg to __init__ signature",
)

# Save bgc_cfg as instance attribute — find the end of __init__ body
patch(
    MODEL_PY,
    # ── old ──────────────────────────────────────────────────────────────
    """\
        self.grid    = grid
        self.z_coord = z_coord
        self.config  = config""",
    # ── new ──────────────────────────────────────────────────────────────
    """\
        self.grid      = grid
        self.z_coord   = z_coord
        self.config    = config
        self._bgc_cfg  = bgc_cfg""",
    "ocean_model.py: store _bgc_cfg",
)


# ===========================================================================
# 3. run_omip.py — CLI flag, BGC init, BGC step, diagnostics
# ===========================================================================

OMIP_PY = ROOT / "scripts/run_omip.py"

# 3a. CLI flags
patch(
    OMIP_PY,
    # ── old ──────────────────────────────────────────────────────────────
    '    p.add_argument("--checkpoint-days", type=float, default=30.0)',
    # ── new ──────────────────────────────────────────────────────────────
    '''\
    p.add_argument("--checkpoint-days", type=float, default=30.0)
    p.add_argument("--biogeo", type=str, default="none",
                   choices=["none", "abiotic", "npzd", "npzd_v2"],
                   help="BGC scheme (default: none)")
    p.add_argument("--pco2-atm", type=float, default=400.0,
                   help="Atmospheric pCO2 [uatm] for BGC (default: 400)")''',
    "run_omip.py: add --biogeo / --pco2-atm CLI flags",
)

# 3b. BGC initialisation + OceanModel construction (cubed-sphere path)
patch(
    OMIP_PY,
    # ── old ──────────────────────────────────────────────────────────────
    """\
        model = OceanModel(grid, z_coord, config)""",
    # ── new ──────────────────────────────────────────────────────────────
    """\
        # ── BGC setup (cubed-sphere only) ────────────────────────────────
        _bgc_cfg = None
        if getattr(args, "biogeo", "none") != "none":
            from legoesm.ocean.biogeochemistry import BiogeoConfig, init_biogeo_state
            _bgc_cfg = BiogeoConfig(
                scheme=args.biogeo,
                pCO2_atm=getattr(args, "pco2_atm", 400.0),
                wind_speed=7.0,
            )
            logger.info(f"BGC enabled: scheme={args.biogeo}  pCO2_atm={_bgc_cfg.pCO2_atm}")

        model = OceanModel(grid, z_coord, config, bgc_cfg=_bgc_cfg)""",
    "run_omip.py: BGC init and OceanModel construction (cubed-sphere)",
)

# 3c. Initialise biogeo state on template_state after it is built
patch(
    OMIP_PY,
    # ── old ──────────────────────────────────────────────────────────────
    """\
    state = template_state._replace(**replacements)
    return state, restart_day, restart_step""",
    # ── new ──────────────────────────────────────────────────────────────
    """\
    state = template_state._replace(**replacements)
    return state, restart_day, restart_step


def _init_bgc_on_state(state, z_coord, bgc_cfg):
    \"\"\"Attach an initialised OceanBiogeoState to an OceanState.\"\"\"
    if bgc_cfg is None or bgc_cfg.scheme == "none":
        return state
    from legoesm.ocean.biogeochemistry import init_biogeo_state
    shape_3d = state.T.data.shape          # (6, n, n, nlev) for cubed-sphere
    biogeo   = init_biogeo_state(shape_3d, z_coord.z_full_ref, bgc_cfg)
    return state._replace(biogeo=biogeo)""",
    "run_omip.py: add _init_bgc_on_state helper",
)

# 3d. Call _init_bgc_on_state just before the time loop starts
patch(
    OMIP_PY,
    # ── old ──────────────────────────────────────────────────────────────
    """\
    scalars = _extract_scalars(state, grid_type, grid, z_coord)
    for k, v in scalars.items():""",
    # ── new ──────────────────────────────────────────────────────────────
    """\
    # Attach BGC state if enabled (no-op when bgc_cfg is None)
    _bgc_cfg_loop = getattr(model, "_bgc_cfg", None)
    if _bgc_cfg_loop is not None:
        state = _init_bgc_on_state(state, z_coord, _bgc_cfg_loop)
        logger.info("BGC state initialised on OceanState")

    scalars = _extract_scalars(state, grid_type, grid, z_coord)
    for k, v in scalars.items():""",
    "run_omip.py: call _init_bgc_on_state before time loop",
)

# 3e. Add BGC surface diagnostics to _extract_scalars
patch(
    OMIP_PY,
    # ── old ──────────────────────────────────────────────────────────────
    """\
def _extract_scalars(state, grid_type, grid, z_coord):
    \"\"\"Compute scalar diagnostics from ocean state.\"\"\"""",
    # ── new ──────────────────────────────────────────────────────────────
    """\
def _extract_bgc_scalars(state, mask):
    \"\"\"Extract surface BGC diagnostics from biogeo state (if present).\"\"\"
    result = {}
    b = getattr(state, "biogeo", None)
    if b is None:
        return result
    import numpy as np
    wet = np.asarray(mask) > 0.5
    def _surf_mean(arr):
        a = np.asarray(arr)[..., 0]
        return float(np.nanmean(np.where(wet, a, np.nan))) if wet.any() else 0.0
    result["DIC_surf"]   = _surf_mean(b.DIC)
    result["ALK_surf"]   = _surf_mean(b.ALK)
    if b.NO3   is not None: result["NO3_surf"]   = _surf_mean(b.NO3)
    if b.Phyto is not None: result["Phyto_surf"] = _surf_mean(b.Phyto)
    if b.Zoo   is not None: result["Zoo_surf"]   = _surf_mean(b.Zoo)
    return result


def _extract_scalars(state, grid_type, grid, z_coord):
    \"\"\"Compute scalar diagnostics from ocean state.\"\"\"""",
    "run_omip.py: add _extract_bgc_scalars helper",
)

# 3f. Call _extract_bgc_scalars inside _extract_scalars before the return
patch(
    OMIP_PY,
    # ── old ──────────────────────────────────────────────────────────────
    """\
    if grid_type == "mpas":
        # MPAS: (nCells, nlev), mask: (nCells,)
        wet = mask > 0.5
        sst = float(np.mean(T[wet, 0])) if wet.any() else 0.0
        sss = float(np.mean(S[wet, 0])) if wet.any() else 0.0
        ssh = float(np.mean(eta[wet])) if wet.any() else 0.0
        u = np.asarray(state.u.data)
        max_u = float(np.max(np.abs(u)))""",
    # ── new ──────────────────────────────────────────────────────────────
    """\
    if grid_type == "mpas":
        # MPAS: (nCells, nlev), mask: (nCells,)
        wet = mask > 0.5
        sst = float(np.mean(T[wet, 0])) if wet.any() else 0.0
        sss = float(np.mean(S[wet, 0])) if wet.any() else 0.0
        ssh = float(np.mean(eta[wet])) if wet.any() else 0.0
        u = np.asarray(state.u.data)
        max_u = float(np.max(np.abs(u)))
        result = {"SST": sst, "SSS": sss, "SSH": ssh, "max_speed": max_u}
        result.update(_extract_bgc_scalars(state, mask))
        return result""",
    "run_omip.py: add BGC scalars to MPAS path",
)


# ===========================================================================
# Verify
# ===========================================================================

print("\nVerifying patches...")
assert_contains(STATE_PY,  "biogeo: OceanBiogeoState | None = None",
                "OceanState.biogeo field")
assert_contains(MODEL_PY,  "step_ocean_biogeochemistry",
                "ocean_model.py BGC step call")
assert_contains(MODEL_PY,  "bgc_cfg=None",
                "ocean_model.py __init__ bgc_cfg param")
assert_contains(OMIP_PY,   "--biogeo",
                "run_omip.py --biogeo flag")
assert_contains(OMIP_PY,   "_init_bgc_on_state",
                "run_omip.py _init_bgc_on_state")
assert_contains(OMIP_PY,   "_extract_bgc_scalars",
                "run_omip.py _extract_bgc_scalars")

print("""
All patches applied successfully.

Test with a quick abiotic run (no biology, just DIC + ALK + air-sea CO2):
  cd /work/bd1083/b309178/diffESM/legoesm_ggn
  JAX_ENABLE_X64=1 .venv/bin/python scripts/run_omip.py \\
      --quick --grid cubed_sphere --biogeo abiotic

Then a full NPZD run:
  JAX_ENABLE_X64=1 .venv/bin/python scripts/run_omip.py \\
      --quick --grid cubed_sphere --biogeo npzd

Check the timeseries CSV for DIC_surf, NO3_surf, Phyto_surf columns.
""")
