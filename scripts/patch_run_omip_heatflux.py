#!/usr/bin/env python
"""Patch scripts/run_omip.py so JRA55-do bulk heat fluxes actually reach T."""
import re
import shutil
import sys

PATH = "/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/scripts/run_omip.py"
src = open(PATH).read()

if "_apply_surface_heat_flux" in src:
    sys.exit("run_omip.py already patched (found _apply_surface_heat_flux). Nothing to do.")

shutil.copy(PATH, PATH + ".bak_heatflux")
print("backup:", PATH + ".bak_heatflux")


def replace(old, new, label=""):
    global src
    n = src.count(old)
    if n < 1:
        open(PATH, "w").write(open(PATH + ".bak_heatflux").read())
        sys.exit(f"[{label}] anchor not found -> run_omip.py restored from backup.\nAnchor:\n{old}")
    src = src.replace(old, new)
    print(f"[{label}] replaced {n} occurrence(s)")


# 1. physics preset choice
replace('choices=["full", "minimal", "none"])',
        'choices=["full", "full_bulk", "minimal", "none"])',
        label="physics choices")

# 2. full_bulk disables the physics-level restoring
replace('        lateral_mixing=LateralMixingConfig(scheme="gm_redi"),\n'
        '        surface_forcing=SurfaceForcingConfig(scheme="restoring"),',
        '        lateral_mixing=LateralMixingConfig(scheme="gm_redi"),\n'
        '        # full_bulk: surface fluxes come from the JRA55-do bulk path only\n'
        '        surface_forcing=SurfaceForcingConfig(\n'
        '            scheme="none" if preset == "full_bulk" else "restoring"),',
        label="full_bulk preset")

# 3. penetrating SW uses absorbed (albedo-corrected) SW
replace('    sf = OceanSurfaceForcing(\n'
        '        sw_down=atm.sw_down,',
        '    sf = OceanSurfaceForcing(\n'
        '        sw_down=sw_net,  # absorbed SW for penetration (was gross sw_down)',
        label="sw_net for penetration")

# 4. apply non-solar heat flux to the top layer after model.step
replace('    # Post-step closure-domain operations.\n'
        '    if jra55_state.get("enable_sss_restoring", False):',
        '    # Non-solar surface heat flux into the top layer. q_net only reaches KPP\n'
        '    # (buoyancy / non-local transport), which adds no net heat, so LW/SH/LH\n'
        '    # must be applied explicitly. SW enters via shortwave penetration.\n'
        '    q_ns = atm.lw_down - tile_resp.lw_up - tile_resp.shflx - tile_resp.lhflx\n'
        '    state = _apply_surface_heat_flux(state, jra55_state, q_ns, dt)\n'
        '\n'
        '    # Post-step closure-domain operations.\n'
        '    if jra55_state.get("enable_sss_restoring", False):',
        label="apply q_ns after step")

# 5. helper function, inserted before _apply_sss_restoring
helper = '''def _apply_surface_heat_flux(state, jra55_state, q_ns, dt):
    """Add a surface heat flux [W/m^2, + into ocean] to the top layer.

    Post-step operator split, same top-layer thickness and land mask as
    _apply_sss_restoring:  T_top += dt * q_ns / (rho_0 * c_p * dz_top)
    """
    from legoesm import constants as _c
    rho0 = float(getattr(_c, "rho_0", 1025.0))
    cp = float(getattr(_c, "c_sw", getattr(_c, "c_p_sw", 3992.0)))
    T = state.T.data
    mask = jnp.asarray(state.land_mask.data, dtype=T.dtype)
    fac = jnp.asarray(dt / (rho0 * cp * max(jra55_state["dz_top"], 1e-6)), dtype=T.dtype)
    dT = fac * jnp.asarray(q_ns, dtype=T.dtype) * mask
    T_new = T.at[..., 0].add(dT)
    return state._replace(T=state.T.replace(data=T_new))


def _apply_sss_restoring(state, jra55_state, dt):'''
replace('def _apply_sss_restoring(state, jra55_state, dt):', helper, label="helper")

open(PATH, "w").write(src)
print("\nPatched:", PATH)

print("\n--- review these lines ---")
for i, line in enumerate(src.splitlines(), 1):
    if re.search(r'q_net\s*=\s*\(|sw_down=atm\.sw_down|== "full"|!= "full"|in \("full"|"dz_top"', line):
        print(f"{i:5d}: {line.rstrip()}")
