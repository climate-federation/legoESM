"""Replace the leaky masked-Laplacian diffusion of MARBL tracers by a conservative
flux-form diffusion (zero flux through coasts / sea floor, face thickness min(h)).
Backup: ocean_model_latlon_cgrid.py.bak_fluxdiff"""
import sys
from pathlib import Path

P = Path("/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py")
if len(sys.argv) > 1:
    P = Path(sys.argv[1])

OLD = '''    K_h = float(config.K_h or 0.0)

    def _lap(f):
        return jax.vmap(lambda f2, m2: laplacian_cgrid(f2, grid, mask=m2),
                        in_axes=(-1, -1), out_axes=-1)(f, active_3d)

    def _one(tr):
        dtype = tr.dtype
        if K_h > 0.0:
            tr = tr + dt * K_h * _lap(tr) * active_3d
'''
NEW = '''    K_h = float(config.K_h or 0.0)
    _diff = _marbl_flux_form_diffusion(grid, h_k_old, active_3d, K_h, dt) if K_h > 0.0 else None

    def _one(tr):
        dtype = tr.dtype
        if _diff is not None:
            tr = tr + _diff(tr).astype(dtype)
'''
MARK = "def _transport_marbl_tracers("
HELPER = '''def _marbl_flux_form_diffusion(grid, h, act, K_h, dt, radius=6.371229e6):
    """Conservative horizontal diffusion for MARBL tracers on the lat-lon C-grid.

    Flux form with face thickness min(h_left, h_right) and zero flux across faces
    touching an inactive (land / below-bottom) cell, so sum(area * h * f) is
    conserved exactly.  Returns f -> dt * df/dt (same shape as f)."""
    lat = jnp.asarray(grid.lat).reshape(-1)
    lon = jnp.asarray(grid.lon).reshape(-1)
    dlam = jnp.abs(lon[1] - lon[0])
    dphi = jnp.abs(lat[1] - lat[0])
    cos_c = jnp.cos(lat)[:, None, None]
    cos_f = jnp.cos(0.5 * (lat[1:] + lat[:-1]))[:, None, None]
    a = (act > 0.5).astype(h.dtype)
    hx = jnp.minimum(h, jnp.roll(h, 1, axis=1)) * a * jnp.roll(a, 1, axis=1)
    hy = jnp.minimum(h[1:], h[:-1]) * a[1:] * a[:-1]
    cx = K_h * hx * dphi / (cos_c * dlam)
    cy = K_h * hy * cos_f * dlam / dphi
    vol = radius ** 2 * cos_c * dlam * dphi * jnp.maximum(h, 1e-10)

    def apply(f):
        tx = cx * (f - jnp.roll(f, 1, axis=1))
        ty = cy * (f[1:] - f[:-1])
        div = jnp.roll(tx, -1, axis=1) - tx
        div = div.at[:-1].add(ty).at[1:].add(-ty)
        return jnp.where(a > 0, dt * div / vol, 0.0)

    return apply


'''
s = P.read_text()
bad = [n for pat, n in ((OLD, "old Laplacian block"), (MARK, "def _transport_marbl_tracers(")) if s.count(pat) != 1]
if "_marbl_flux_form_diffusion" in s:
    bad.append("already patched")
if bad:
    sys.exit(f"not patched, nothing changed: {bad}")
P.with_name(P.name + ".bak_fluxdiff").write_text(s)
P.write_text(s.replace(OLD, NEW).replace(MARK, HELPER + MARK))
print(f"patched {P} (backup {P.name}.bak_fluxdiff)")
