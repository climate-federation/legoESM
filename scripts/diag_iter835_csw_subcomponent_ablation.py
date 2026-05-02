"""Iter-835 diagnostic: sub-component ablation inside `_c_sw`.

iter-831 showed FB `_c_sw` is the h-growth driver (disabling it
drops h_max 10765→2980 on C24 W2 12h).  iter-832 ruled out sync
over-application (only 6% contribution).  iter-835 drills one level
deeper: inside `_c_sw`, which specific sub-step drives the h-growth?

Sub-components ablated (one at a time):

  (A) mass_flux     : zero the mass-flux divergence → h_star = h.
                     p_grad_c then computes gradient of the (pre-
                     transport) h field, which still varies spatially
                     for W2 (sin²(lat) structure, NOT constant); the
                     pair desynchronises from the actual transported h,
                     mis-timing the stabilising pressure-gradient
                     correction.  Isolates whether the first-order
                     upwind h_star construction is a driver.
  (B) ke_gradient   : zero the KE contribution (dke_x = dke_y = 0).
                     uc_new = uc + fy1*vort_x (vort flux only).
  (C) vort_flux     : zero the vort-flux contribution
                     (fy1*vort_x = fx1*vort_y = 0).  uc_new = uc + dke_x
                     (KE gradient only).
  (D) p_grad_c      : zero p_grad_c's output (dp_x = dp_y = 0) so the
                     pressure-gradient correction is removed.  This
                     leaves c_sw's uc_new/vc_new untouched before
                     feeding d_sw_native.

Each ablation is MONKEY-PATCHED by wrapping the corresponding
module-level function.  No production code is changed.  The
patches are reverted in a `finally` clause.

Expected interpretation cues:
- If (A) drops h_max near 2980: first-order upwind mass transport
  is the dominant h-growth driver.
- If (B) drops h_max near 2980: KE gradient at cube corners is the
  driver (candidate: Fortran c_sw KE upwind blend missing in our
  DUOGRID path).
- If (C) drops h_max near 2980: vorticity-flux construction is the
  driver.
- If (D) drops h_max near 2980: pressure gradient (c_sw's h_star→
  p_grad_c coupling) is the driver (would support the iter-831
  hypothesis that the `c_sw h_star` → `p_grad_c` chain is the
  feedback path).
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
for _stale in ("JAX_PLATFORM_NAME", "JAX_DISABLE_JIT", "JAX_DEBUG_NANS"):
    os.environ.pop(_stale, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.core import fv3_sw_core as fv3_mod
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3FBShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


# Capture originals at script-top time.  For correctness, the process
# must enter this script with pristine fv3_sw_core bindings — i.e. no
# prior monkey-patch has installed a stub at `fv3_mod._c_sw`.  This is
# the intended use (standalone CLI invocation).  The `finally` block
# restores from this snapshot at script exit; if another script patches
# _c_sw before this one is invoked in the same session, the
# restoration would restore the patched stub, not the pristine
# production function (Codex iter-835b minor caveat).
orig_c_sw = fv3_mod._c_sw
orig_p_grad_c = fv3_mod._p_grad_c


def _run_fb(n, ablation, hours=12.0, dt=300.0):
    """Run FB DUOGRID W2 with a specific sub-component ablation.

    ablation ∈ {'baseline', 'mass_flux', 'ke_gradient', 'vort_flux',
                'p_grad_c'}
    """
    n_steps = int(round(hours * 3600 / dt))

    # -------- monkey-patch helpers --------
    if ablation == 'baseline':
        fv3_mod._c_sw = orig_c_sw
        fv3_mod._p_grad_c = orig_p_grad_c
    elif ablation == 'mass_flux':
        # Run orig _c_sw then overwrite h_star with h (no mass transport)
        def _patched_c_sw(h, u_d, v_d, h_s, cdgrid, dt, g):
            h_star, uc_new, vc_new, ua, va = orig_c_sw(
                h, u_d, v_d, h_s, cdgrid, dt, g)
            return h, uc_new, vc_new, ua, va   # h_star <- h
        fv3_mod._c_sw = _patched_c_sw
        fv3_mod._p_grad_c = orig_p_grad_c
    elif ablation == 'ke_gradient':
        # Replicate orig _c_sw but zero the dke_x, dke_y contribution.
        from legoesm.core.fv3_sw_core import (
            _d2a2c_vect, _ke_upwind, _corner_vorticity, _vorticity_flux,
            _pad_halo_auto)
        def _patched_c_sw(h, u_d, v_d, h_s, cdgrid, dt, g):
            n_ = cdgrid.n
            dt2 = 0.5 * dt
            dg = cdgrid.base.duogrid
            use_duogrid = dg is not None and dg.ng >= 2

            ua, va, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)
            from legoesm.grids.halo import pad_halo
            dy = cdgrid.dy_edge_x
            dx = cdgrid.dx_edge_y
            sg = cdgrid.sin_sg
            grid = cdgrid.base
            _offs = None if use_duogrid else grid.halo_interp_offsets
            _dg = dg if use_duogrid else None

            sin_east = sg[:, :, :, 2]
            sin_west = sg[:, :, :, 0]
            se_pad = pad_halo(sin_east, interp_offsets=_offs, duogrid=_dg)
            sw_pad = pad_halo(sin_west, interp_offsets=_offs, duogrid=_dg)
            sin_upwind_x = jnp.where(ut > 0, se_pad[:, :n_+1, 1:-1],
                                              sw_pad[:, 1:n_+2, 1:-1])
            ut_scaled = dt2 * ut * dy * sin_upwind_x

            sin_north = sg[:, :, :, 3]
            sin_south = sg[:, :, :, 1]
            sn_pad = pad_halo(sin_north, interp_offsets=_offs, duogrid=_dg)
            ss_pad = pad_halo(sin_south, interp_offsets=_offs, duogrid=_dg)
            sin_upwind_y = jnp.where(vt > 0, sn_pad[:, 1:-1, :n_+1],
                                              ss_pad[:, 1:-1, 1:n_+2])
            vt_scaled = dt2 * vt * dx * sin_upwind_y

            h_pad = _pad_halo_auto(h, cdgrid)
            h_left = h_pad[:, :-1, 1:-1]
            h_right = h_pad[:, 1:, 1:-1]
            fx = jnp.where(ut_scaled > 0, h_left, h_right) * ut_scaled

            h_bot = h_pad[:, 1:-1, :-1]
            h_top = h_pad[:, 1:-1, 1:]
            fy = jnp.where(vt_scaled > 0, h_bot, h_top) * vt_scaled

            if use_duogrid:
                from legoesm.grids.halo import synchronize_cgrid_fluxes
                fx, fy = synchronize_cgrid_fluxes(fx, fy, n_)

            rarea = 1.0 / cdgrid.base.area
            h_star = h + (fx[:, :-1, :] - fx[:, 1:, :]
                          + fy[:, :, :-1] - fy[:, :, 1:]) * rarea

            vort_abs = _corner_vorticity(uc, vc, cdgrid, use_duogrid)
            fy1, vort_x, fx1, vort_y = _vorticity_flux(
                v_d, u_d, uc, vc, vort_abs, cdgrid, use_duogrid)
            fy1 = dt2 * fy1
            fx1 = dt2 * fx1

            # ABLATION: zero dke_x, dke_y
            uc_new = uc + fy1 * vort_x
            vc_new = vc - fx1 * vort_y
            return h_star, uc_new, vc_new, ua, va
        fv3_mod._c_sw = _patched_c_sw
        fv3_mod._p_grad_c = orig_p_grad_c
    elif ablation == 'vort_flux':
        # Replicate orig _c_sw but zero the fy1*vort_x, fx1*vort_y.
        from legoesm.core.fv3_sw_core import (
            _d2a2c_vect, _ke_upwind, _corner_vorticity, _vorticity_flux,
            _pad_halo_auto)
        def _patched_c_sw(h, u_d, v_d, h_s, cdgrid, dt, g):
            n_ = cdgrid.n
            dt2 = 0.5 * dt
            dg = cdgrid.base.duogrid
            use_duogrid = dg is not None and dg.ng >= 2

            ua, va, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)
            from legoesm.grids.halo import pad_halo
            dy = cdgrid.dy_edge_x
            dx = cdgrid.dx_edge_y
            sg = cdgrid.sin_sg
            grid = cdgrid.base
            _offs = None if use_duogrid else grid.halo_interp_offsets
            _dg = dg if use_duogrid else None

            sin_east = sg[:, :, :, 2]
            sin_west = sg[:, :, :, 0]
            se_pad = pad_halo(sin_east, interp_offsets=_offs, duogrid=_dg)
            sw_pad = pad_halo(sin_west, interp_offsets=_offs, duogrid=_dg)
            sin_upwind_x = jnp.where(ut > 0, se_pad[:, :n_+1, 1:-1],
                                              sw_pad[:, 1:n_+2, 1:-1])
            ut_scaled = dt2 * ut * dy * sin_upwind_x

            sin_north = sg[:, :, :, 3]
            sin_south = sg[:, :, :, 1]
            sn_pad = pad_halo(sin_north, interp_offsets=_offs, duogrid=_dg)
            ss_pad = pad_halo(sin_south, interp_offsets=_offs, duogrid=_dg)
            sin_upwind_y = jnp.where(vt > 0, sn_pad[:, 1:-1, :n_+1],
                                              ss_pad[:, 1:-1, 1:n_+2])
            vt_scaled = dt2 * vt * dx * sin_upwind_y

            h_pad = _pad_halo_auto(h, cdgrid)
            h_left = h_pad[:, :-1, 1:-1]
            h_right = h_pad[:, 1:, 1:-1]
            fx = jnp.where(ut_scaled > 0, h_left, h_right) * ut_scaled

            h_bot = h_pad[:, 1:-1, :-1]
            h_top = h_pad[:, 1:-1, 1:]
            fy = jnp.where(vt_scaled > 0, h_bot, h_top) * vt_scaled

            if use_duogrid:
                from legoesm.grids.halo import synchronize_cgrid_fluxes
                fx, fy = synchronize_cgrid_fluxes(fx, fy, n_)

            rarea = 1.0 / cdgrid.base.area
            h_star = h + (fx[:, :-1, :] - fx[:, 1:, :]
                          + fy[:, :, :-1] - fy[:, :, 1:]) * rarea

            ke_u, ke_v = _ke_upwind(uc, vc, ua, va, u_d, v_d, cdgrid,
                                    use_duogrid)
            ke_total = dt2 * 0.5 * (ua * ke_u + va * ke_v)
            ke_pad = _pad_halo_auto(ke_total, cdgrid)
            dke_x = cdgrid.rdxc * (ke_pad[:, :-1, 1:-1]
                                    - ke_pad[:, 1:, 1:-1])
            dke_y = cdgrid.rdyc * (ke_pad[:, 1:-1, :-1]
                                    - ke_pad[:, 1:-1, 1:])

            # ABLATION: zero vort-flux; keep KE gradient
            uc_new = uc + dke_x
            vc_new = vc + dke_y
            return h_star, uc_new, vc_new, ua, va
        fv3_mod._c_sw = _patched_c_sw
        fv3_mod._p_grad_c = orig_p_grad_c
    elif ablation == 'p_grad_c':
        def _patched_p_grad_c(h_star, h_s, cdgrid, dt2, g):
            sh_x = (h_star.shape[0], h_star.shape[1] + 1, h_star.shape[2])
            sh_y = (h_star.shape[0], h_star.shape[1], h_star.shape[2] + 1)
            return jnp.zeros(sh_x), jnp.zeros(sh_y)
        fv3_mod._c_sw = orig_c_sw
        fv3_mod._p_grad_c = _patched_p_grad_c
    else:
        raise ValueError(f"unknown ablation {ablation!r}")

    # -------- actual run --------
    # Codex iter-835b correction: disable the conservation fixer so
    # h_max is reported pre-correction.  The fixer adds a SCALAR
    # correction (mass_target - mass_new)/total_area to every cell,
    # which preserves h_max − h_min ordering but shifts absolute
    # h_max across ablations.  Observational runs should report raw
    # pre-fixer state to avoid conflating the fixer with the
    # ablation signal.
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0, div_damp=0.0, boundary_fix=True,
        damp_v=0.0, nord_v=0, d4_bg=0.16, nord=1, fix_mass=False)
    model = FV3FBShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    crashed_step = None
    for step in range(n_steps):
        state = model.step(state, dt)
        if not np.all(np.isfinite(np.asarray(state.h))):
            crashed_step = step
            break

    if crashed_step is not None:
        return {'crashed_at_step': crashed_step,
                'crashed_at_hour': crashed_step * dt / 3600.0}

    h_np = np.asarray(state.h)
    return {
        'h_max': float(np.max(np.abs(h_np))),
        'h_min': float(np.min(h_np)),
    }


n = 24
hours = 12.0

print(f"Iter-835 FB DUOGRID C{n} W2 {hours}h sub-component ablation "
      f"inside _c_sw + p_grad_c")
print()
print(f"{'ablation':>14}  {'status':>22}  {'h_min':>8}  {'h_max':>8}")
print("-" * 60)

try:
    for ablation in ('baseline', 'mass_flux', 'ke_gradient', 'vort_flux',
                     'p_grad_c'):
        r = _run_fb(n, ablation)
        if 'crashed_at_step' in r:
            status = (f"CRASH step {r['crashed_at_step']} "
                      f"({r['crashed_at_hour']:.1f}h)")
            h_min = h_max = '—'
        else:
            status = 'ok'
            h_min = f"{r['h_min']:.0f}"
            h_max = f"{r['h_max']:.0f}"
        print(f"{ablation:>14}  {status:>22}  {h_min:>8}  {h_max:>8}")
finally:
    fv3_mod._c_sw = orig_c_sw
    fv3_mod._p_grad_c = orig_p_grad_c

print()
print("Interpretation cues (observational only):")
print("- mass_flux drop → first-order upwind h_star is the driver")
print("  (the h_star field propagates into p_grad_c → uc_new/vc_new).")
print("- ke_gradient drop → KE upwind blend at cube corners is the")
print("  driver (candidate: Fortran c_sw non-duogrid sin_sg blend that")
print("  Python's DUOGRID path does not apply).")
print("- vort_flux drop → corner vorticity / sina_u,v construction is")
print("  the driver (candidate: our sina_u from sin_sg sub-grid).")
print("- p_grad_c drop → pressure gradient correction (h_star coupling)")
print("  is the driver.  Combined with mass_flux: the pair localises")
print("  whether the h_star field itself OR the p_grad_c operator is")
print("  the feedback.")
