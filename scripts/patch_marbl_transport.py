"""Transport MARBL tracers in the lat-lon C-grid ocean model, and fix BGC restarts.

Edits (all-or-nothing; backups written as *.bak_marbl):

1. src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py
   a. new module-level helpers ``_is_marbl_tracers`` and ``_transport_marbl_tracers``
      (inserted before ``def _forward_backward_coriolis_3d(``)
   b. in ``_step_impl``: when ``state.biogeo`` is a ``MarblTracers``, apply the
      same horizontal diffusion (K_h) and the same flux-form advection (same mass
      fluxes, layer thicknesses, scheme and integrator) as T and S
   c. in ``_apply_implicit_vertical_mixing``: the same backward-Euler K_v solve
      for every MARBL tracer
   Older BGC schemes (OceanBiogeoState) are untouched: still not transported.

2. scripts/run_omip.py
   restart restore rebuilds ``type(state.biogeo)`` instead of always
   ``OceanBiogeoState`` (so MarblTracers restarts load).

Usage (from the repo root):  python scripts/patch_marbl_transport.py
"""
import sys
from pathlib import Path

MODEL = Path("src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py")
OMIP = Path("scripts/run_omip.py")

HELPERS = '''def _is_marbl_tracers(bgc) -> bool:
    """True when ``bgc`` is the 32-tracer MARBL state (transported by the model)."""
    if bgc is None:
        return False
    from legoesm.ocean.biogeochemistry.marbl.config import MarblTracers
    return isinstance(bgc, MarblTracers)


def _transport_marbl_tracers(bgc, config, grid, mass_flux_u, mass_flux_v, w_baro,
                             h_k_old, h_k_new, h_u_old, h_v_old, active_3d, dt):
    """Move MARBL tracers like T and S: K_h Laplacian diffusion, then the same
    flux-form advection (scheme, integrator, mass fluxes, layer thicknesses).
    Vertical mixing is applied in ``_apply_implicit_vertical_mixing``."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import laplacian_cgrid

    if config.tracer_advection == "som":
        raise NotImplementedError("MARBL tracers: tracer_advection='som' is not supported")
    if config.tracer_time_integrator == "ab2":
        raise NotImplementedError("MARBL tracers: use tracer_time_integrator 'euler' or 'rk3'")
    if config.gm_redi is not None:
        raise NotImplementedError("MARBL tracers: GM/Redi transport is not implemented")
    if not config.implicit_vertical_mixing:
        raise NotImplementedError("MARBL tracers need implicit_vertical_mixing=True")

    K_h = float(config.K_h or 0.0)

    def _lap(f):
        return jax.vmap(lambda f2, m2: laplacian_cgrid(f2, grid, mask=m2),
                        in_axes=(-1, -1), out_axes=-1)(f, active_3d)

    def _one(tr):
        dtype = tr.dtype
        if K_h > 0.0:
            tr = tr + dt * K_h * _lap(tr) * active_3d
        if config.tracer_time_integrator == "rk3":
            tr_new = _ssp_rk3_tracer_step(
                tr, config.tracer_advection,
                mass_flux_u, mass_flux_v, w_baro,
                h_k_old, h_k_new, h_u_old, h_v_old,
                grid, dt, active_3d,
            )
        else:
            div_hut, vert_flux_div = _compute_advection_flux_div(
                tr, config.tracer_advection,
                mass_flux_u, mass_flux_v, w_baro,
                h_k_old, h_u_old, h_v_old, grid, dt,
            )
            tr_new = (h_k_old * tr - dt * (div_hut + vert_flux_div)) / jnp.maximum(h_k_new, 1e-10)
            tr_new = jnp.where(active_3d > 0.5, tr_new, tr)
        return tr_new.astype(dtype)

    return jax.tree_util.tree_map(_one, bgc)


'''

STEP_MARKER = "        # Include vertical velocity diagnostic in state\n"
STEP_INSERT = '''        # MARBL tracers: same horizontal diffusion and advection as T and S
        # (older BGC schemes in state.biogeo are not transported, as before).
        if _is_marbl_tracers(state.biogeo):
            state_new = state_new._replace(biogeo=_transport_marbl_tracers(
                state.biogeo, self.config, self.grid,
                mass_flux_u, mass_flux_v, w_baro,
                h_k_old, h_k_new, h_u_old, h_v_old, active_3d, dt,
            ))

'''

IVM_OLD = '''        return state._replace(
            u=state.u.replace(data=u_new),
            v=state.v.replace(data=v_new),
            T=state.T.replace(data=T_new),
            S=state.S.replace(data=S_new),
        )
'''
IVM_NEW = '''        state = state._replace(
            u=state.u.replace(data=u_new),
            v=state.v.replace(data=v_new),
            T=state.T.replace(data=T_new),
            S=state.S.replace(data=S_new),
        )
        if _is_marbl_tracers(state.biogeo):
            # Same backward-Euler K_v solve for every MARBL tracer.
            def _mix(f):
                f_new = implicit_vertical_diffusion_ocean(
                    f, K_v_cell.astype(f.dtype), dz_cell.astype(f.dtype),
                    dz_half_cell.astype(f.dtype), dt,
                )
                return jnp.where(mask_3d > 0.5, f_new, f)
            state = state._replace(biogeo=jax.tree_util.tree_map(_mix, state.biogeo))
        return state
'''

RESTART_OLD = "biogeo=OceanBiogeoState(**kwargs)"
RESTART_NEW = "biogeo=type(bgc)(**kwargs)"


def main():
    for p in (MODEL, OMIP):
        if not p.exists():
            sys.exit(f"{p} not found -- run from the repo root; nothing changed")
    m = MODEL.read_text()
    o = OMIP.read_text()

    checks = [
        (m, "def _forward_backward_coriolis_3d(\n", "model: def _forward_backward_coriolis_3d("),
        (m, STEP_MARKER, "model: '# Include vertical velocity diagnostic in state'"),
        (m, IVM_OLD, "model: return block of _apply_implicit_vertical_mixing"),
        (o, RESTART_OLD, "run_omip: biogeo=OceanBiogeoState(**kwargs)"),
    ]
    bad = [name for text, pat, name in checks if text.count(pat) != 1]
    if "_transport_marbl_tracers" in m:
        bad.append("model already contains _transport_marbl_tracers (patched before?)")
    if bad:
        sys.exit("not patched, nothing changed:\n  " + "\n  ".join(bad))

    m = m.replace("def _forward_backward_coriolis_3d(\n",
                  HELPERS + "def _forward_backward_coriolis_3d(\n")
    m = m.replace(STEP_MARKER, STEP_INSERT + STEP_MARKER)
    m = m.replace(IVM_OLD, IVM_NEW)
    o = o.replace(RESTART_OLD, RESTART_NEW)

    MODEL.with_name(MODEL.name + ".bak_marbl").write_text(MODEL.read_text())
    OMIP.with_name(OMIP.name + ".bak_marbl").write_text(OMIP.read_text())
    MODEL.write_text(m)
    OMIP.write_text(o)
    print(f"patched {MODEL} and {OMIP} (backups: *.bak_marbl)")


if __name__ == "__main__":
    main()
