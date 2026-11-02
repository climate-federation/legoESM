"""GM/Redi transport for the MARBL tracers, identical to the one applied to T and S.

Until now _transport_marbl_tracers raised NotImplementedError when gm_redi was configured, so
MARBL could only run with --no-gm-redi.  This patch

1. gm_redi_latlon_cgrid.py: adds gm_redi_tendencies_many_latlon(Q, T, S, eta, H_bathy, grid,
   z_coord, cfg, ...): the same preamble as gm_redi_tracer_tendency_latlon (jacobian, EOS with
   2 iterations, slopes, Visbeck kappa_GM), computed ONCE from T and S, then the same per-tracer
   tendency (triads; or centered + surface complement) for every tracer of the stack Q
   (n_tracers, n_lat, n_lon, nlev) via jax.lax.map.
2. ocean_model_latlon_cgrid.py:
   - in the step, right before the T/S GM/Redi block, stores the arguments of the T/S call
     (T_mid, S_mid, eta, H_bathy, masks, eos) in _gm_ctx_marbl;
   - passes gm_ctx=_gm_ctx_marbl to _transport_marbl_tracers;
   - _transport_marbl_tracers applies Q += dt * GM/Redi(Q) on active cells before diffusion and
     advection (T/S: Euler step before advection), instead of raising NotImplementedError.
   MARBL_NO_GM=1 switches the MARBL GM/Redi off (tests).

Backups: <file>.bak_marblgm.  Usage:
  python patch_marbl_gm_redi.py /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/src/legoesm
"""
import re
import shutil
import sys
from pathlib import Path

NEW_FUNC = '''


# ----------------------------------------------------------------------------- [patch_marbl_gm_redi]
def gm_redi_tendencies_many_latlon(
    Q, T, S, eta, H_bathy, grid, z_coord, cfg, *,
    eos="wright", eos_linear=None, mask=None, u_mask=None, v_mask=None, f_coriolis=None,
):
    """GM/Redi tendencies for a stack of passive tracers Q (n, n_lat, n_lon, nlev).

    Density, slopes and kappa_GM come from T and S exactly as in gm_redi_tracer_tendency_latlon,
    and every tracer gets the same per-tracer operator as T and S (so q = T gives dT_dt)."""
    import jax as _jax
    if mask is None:
        mask = jnp.ones(T.shape[:2], dtype=T.dtype)
    if u_mask is None:
        u_mask = jnp.ones((T.shape[0], T.shape[1] + 1), dtype=T.dtype)
    if v_mask is None:
        v_mask = jnp.ones((T.shape[0] + 1, T.shape[1]), dtype=T.dtype)
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
    eos_fn = make_eos_fn(eos, eos_linear)
    fill_fn = lambda field: _neumann_fill_cgrid(field, mask)  # noqa: E731
    rho, _rho_prime, _p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask, fill_fn, eos_fn, z_coord.dz_ref, _RHO_0, constants.g, n_iter=2,
    )
    S_x, S_y, _taper = compute_isopycnal_slopes_latlon_cgrid(rho, mask, z_coord, jacobian, grid, cfg)
    if cfg.visbeck.enabled:
        if f_coriolis is None:
            f_coriolis = 2.0 * constants.Omega * jnp.sin(grid.lat[:, jnp.newaxis])
            f_coriolis = jnp.broadcast_to(f_coriolis, mask.shape)
        kappa_GM = compute_visbeck_kappa_gm(rho, S_x, S_y, z_coord, jacobian, f_coriolis, cfg.visbeck)
    else:
        kappa_GM = cfg.kappa_GM
    scheme = getattr(cfg, "slope_scheme", "triads")
    if scheme not in ("triads", "centered"):
        raise ValueError(f"Unknown GMRediConfig.slope_scheme={scheme!r}")

    def _one(q):
        if scheme == "triads":
            return gm_redi_tracer_tendency_triads_latlon_cgrid(
                q, rho, mask, u_mask, v_mask, z_coord, jacobian, grid, kappa_GM, cfg.kappa_Redi, cfg.S_max)
        d = gm_redi_tracer_tendency_latlon_cgrid(
            q, S_x, S_y, mask, u_mask, v_mask, z_coord, jacobian, grid, kappa_GM, cfg.kappa_Redi)
        if cfg.surface_complement:
            complement = jnp.where(jnp.abs(z_coord.z_full_ref) < cfg.surface_complement_depth, 1.0, 0.0)
            q_filled = _neumann_fill_cgrid(q, mask)
            dq_dx_u = gradient_x_cgrid(q_filled, grid) * u_mask[:, :, jnp.newaxis]
            dq_dy_v = gradient_y_cgrid(q_filled, grid) * v_mask[:, :, jnp.newaxis]
            F_x = cfg.kappa_Redi * complement * dq_dx_u
            F_y = cfg.kappa_Redi * complement * dq_dy_v
            d = d + divergence_cgrid(F_x, F_y, grid) * mask[:, :, jnp.newaxis]
        return d

    return _jax.lax.map(_one, Q)
'''

CTX = '''{ind}# MARBL tracers get GM/Redi with the same arguments as T/S  [patch_marbl_gm_redi]
{ind}_gm_ctx_marbl = None
{ind}if self.config.gm_redi is not None:
{ind}    _gm_ctx_marbl = dict(
{ind}        T=T_mid, S=S_mid, eta=state_new.eta.data, H_bathy=state_new.H_bathy.data,
{ind}        grid=self.grid, z_coord=self.z_coord, cfg=self.config.gm_redi,
{ind}        eos=self.config.eos, eos_linear=self.config.eos_linear,
{ind}        mask=state.land_mask.data, u_mask=state.u_mask.data, v_mask=state.v_mask.data,
{ind})
'''

APPLY = '''    leaves, treedef = jax.tree_util.tree_flatten(bgc)
    _stack = jnp.stack(leaves)
    if (config.gm_redi is not None and gm_ctx is not None
            and __import__("os").environ.get("MARBL_NO_GM", "0") != "1"):   # [patch_marbl_gm_redi]
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import gm_redi_tendencies_many_latlon
        _gm = gm_redi_tendencies_many_latlon(_stack, **gm_ctx)
        _stack = (_stack + dt * _gm * (active_3d > 0.5)).astype(_stack.dtype)
    out = jax.lax.map(_one, _stack)
'''


def patch_gm(p):
    s = p.read_text()
    if "patch_marbl_gm_redi" in s:
        return "already patched"
    for name in ("def gm_redi_tracer_tendency_latlon(", "def gm_redi_tracer_tendency_triads_latlon_cgrid(",
                 "def gm_redi_tracer_tendency_latlon_cgrid(", "compute_ocean_jacobian", "make_eos_fn",
                 "iterate_eos_and_pressure_anomaly", "compute_visbeck_kappa_gm", "_RHO_0", "gradient_x_cgrid",
                 "divergence_cgrid", "_neumann_fill_cgrid", "constants"):
        assert name in s, f"{name} not found in {p}"
    out = s.rstrip("\n") + NEW_FUNC
    compile(out, str(p), "exec")
    shutil.copy(p, str(p) + ".bak_marblgm")
    p.write_text(out)
    return "added gm_redi_tendencies_many_latlon"


def patch_model(p):
    s = p.read_text()
    if "patch_marbl_gm_redi" in s:
        return "already patched"
    # 1. signature
    sig_old = "h_k_old, h_k_new, h_u_old, h_v_old, active_3d, dt):"
    i_def = s.index("def _transport_marbl_tracers(")
    i_sig = s.index(sig_old, i_def)
    s = s[:i_sig] + "h_k_old, h_k_new, h_u_old, h_v_old, active_3d, dt, gm_ctx=None):" + s[i_sig + len(sig_old):]
    # 2. no more NotImplementedError when gm_ctx is given
    old_raise = re.compile(r'    if config\.gm_redi is not None:\n\s+raise NotImplementedError\("MARBL tracers: GM/Redi transport is not implemented"\)\n')
    assert len(old_raise.findall(s)) == 1, "GM/Redi NotImplementedError not found exactly once"
    s = old_raise.sub('    if config.gm_redi is not None and gm_ctx is None:\n'
                      '        raise NotImplementedError("MARBL tracers: GM/Redi needs gm_ctx [patch_marbl_gm_redi]")\n', s)
    # 3. apply the tendency before the per-tracer transport
    old_map = ("    leaves, treedef = jax.tree_util.tree_flatten(bgc)\n"
               "    out = jax.lax.map(_one, jnp.stack(leaves))\n")
    i_def = s.index("def _transport_marbl_tracers(")
    i_end = s.find("\ndef ", i_def + 10)
    j = s.find(old_map, i_def, i_end)
    assert j > 0, "tree_flatten / lax.map lines of _transport_marbl_tracers not found"
    s = s[:j] + APPLY + s[j + len(old_map):]
    # 4. context right before the T/S GM/Redi block
    m = re.search(r"^([ \t]*)# GM/Redi isopycnal mixing \(if configured\)\n", s, re.M)
    assert m, "'# GM/Redi isopycnal mixing (if configured)' not found"
    s = s[:m.start()] + CTX.format(ind=m.group(1)) + s[m.start():]
    i_ctx = m.start()
    # 5. pass gm_ctx at the call (after the context, inside the same method)
    calls = [mm.start() for mm in re.finditer(r"_transport_marbl_tracers\(", s) if not s[max(0, mm.start() - 4):mm.start()].endswith("def ")]
    assert len(calls) == 1, f"expected one call of _transport_marbl_tracers, found {len(calls)}"
    c = calls[0]
    assert c > i_ctx, "the MARBL transport call comes before the T/S GM/Redi block"
    assert "\n    def " not in s[i_ctx:c] or s[i_ctx:c].count("\n    def ") == 0, "call and GM/Redi block in different methods"
    k = c + len("_transport_marbl_tracers(")
    depth = 1
    while depth:
        depth += {"(": 1, ")": -1}.get(s[k], 0)
        k += 1
    close = k - 1
    before = s[:close].rstrip()
    sep = "" if before.endswith(",") else ","
    s = before + sep + " gm_ctx=_gm_ctx_marbl" + s[close:]
    compile(s, str(p), "exec")
    shutil.copy(p, str(p) + ".bak_marblgm")
    p.write_text(s)
    return "GM/Redi for MARBL tracers wired into the step and _transport_marbl_tracers"


if __name__ == "__main__":
    src = Path(sys.argv[1])
    g = src / "ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py"
    m = src / "ocean/dynamics/ocean_model_latlon_cgrid.py"
    print(f"{g}: {patch_gm(g)}")
    print(f"{m}: {patch_model(m)}")
