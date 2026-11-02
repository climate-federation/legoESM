"""Keep MARBL tracers finite on land, so a NaN cannot spread into the ocean.
Backups: *.bak_landnan"""
import sys
from pathlib import Path

R = Path("/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/src/legoesm")
OC = Path(sys.argv[1]) if len(sys.argv) > 2 else R / "ocean/biogeochemistry/marbl/omip_coupling.py"
MD = Path(sys.argv[2]) if len(sys.argv) > 2 else R / "ocean/dynamics/ocean_model_latlon_cgrid.py"

OC_EDITS = [
    ('        dep = {k: f[k][m] for k in ("dust_flux", "iron_flux", "nox_flux", "nhy_flux")}\n',
     '        dep = {k: f[k][m] * _M["mask"] for k in ("dust_flux", "iron_flux", "nox_flux", "nhy_flux")}\n'),
    ('    bgc_new, diag = _M["step"](state.biogeo, state.T.data, state.S.data, h_k, forcing)\n',
     '    bgc_new, diag = _M["step"](state.biogeo, state.T.data, state.S.data, h_k, forcing)\n'
     '    _wet = _M["mask"] > 0.5                       # land columns: tracers and diagnostics = 0\n'
     '    bgc_new = jax.tree_util.tree_map(lambda x: jnp.where(_wet[..., None], x, 0.0), bgc_new)\n'
     '    diag = jax.tree_util.tree_map(lambda x: jnp.where(_wet, x, 0.0), diag)\n'),
]
MD_EDITS = [
    ('    def apply(f):\n        tx = cx * (f - jnp.roll(f, 1, axis=1))\n',
     '    def apply(f):\n        f = jnp.where(a > 0, f, 0.0)              # inactive cells never enter the fluxes\n'
     '        tx = cx * (f - jnp.roll(f, 1, axis=1))\n'),
]
plan = []
for path, edits in ((OC, OC_EDITS), (MD, MD_EDITS)):
    s = path.read_text()
    bad = [old[:50] for old, new in edits if s.count(old) != 1]
    if "land columns: tracers" in s or "inactive cells never enter" in s:
        bad.append("already patched")
    if bad:
        sys.exit(f"{path}: not patched, nothing changed: {bad}")
    for old, new in edits:
        s = s.replace(old, new)
    plan.append((path, s))
for path, s in plan:
    path.with_name(path.name + ".bak_landnan").write_text(path.read_text())
    path.write_text(s)
    print(f"patched {path} (backup {path.name}.bak_landnan)")
