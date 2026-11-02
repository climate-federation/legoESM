#!/usr/bin/env python
"""Second patch: apply the heat-flux fix to the scan-block (GPU-interp) path of run_omip.py."""
import shutil, sys

PATH = "/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/scripts/run_omip.py"
src = open(PATH).read()
if "q_ns_blk" in src:
    sys.exit("Already patched (found q_ns_blk).")
shutil.copy(PATH, PATH + ".bak_heatflux2")
print("backup:", PATH + ".bak_heatflux2")

# 1. penetrating SW = absorbed SW in all remaining paths
n = src.count("sw_down=atm.sw_down")
src = src.replace("sw_down=atm.sw_down", "sw_down=sw_net")
print(f"[sw_net] replaced {n} occurrence(s) (expect 2)")

# 2. non-solar heat flux after the step, GPU scan block
anchor = ("                new_state = model._step_impl(\n"
          "                    state_in, dt, freshwater=fw,\n"
          "                    surface_forcing=sf, sponge=sponge_k,\n"
          "                )\n")
add = anchor + (
    "\n"
    "                # Non-solar heat flux into the top layer (q_net only reaches KPP).\n"
    "                q_ns_blk = atm.lw_down - tile.lw_up - tile.shflx - tile.lhflx\n"
    "                _Tb = new_state.T.data\n"
    "                _mb = jnp.asarray(new_state.land_mask.data, dtype=_Tb.dtype)\n"
    "                _Tb_top = _Tb[..., 0] + jnp.asarray(_heat_fac, dtype=_Tb.dtype) * (\n"
    "                    jnp.asarray(q_ns_blk, dtype=_Tb.dtype) * _mb)\n"
    "                new_state = new_state._replace(\n"
    "                    T=new_state.T.replace(data=_Tb.at[..., 0].set(_Tb_top)))\n"
)
n = src.count(anchor)
if n < 1:
    shutil.copy(PATH + ".bak_heatflux2", PATH)
    sys.exit("[q_ns] step anchor not found -> restored from backup")
src = src.replace(anchor, add)
print(f"[q_ns] inserted after {n} scan-block step call(s)")

# 3. static heat factor, defined next to the other static block settings
anchor2 = "    _maxvel_3d = model.config.maxvel_barotropic\n"
add2 = (
    "    from legoesm import constants as _hc\n"
    "    _heat_fac = dt / (float(getattr(_hc, \"rho_0\", 1025.0))\n"
    "                      * float(getattr(_hc, \"c_sw\", getattr(_hc, \"c_p_sw\", 3992.0)))\n"
    "                      * max(float(jra55_state[\"dz_top\"]), 1e-6))\n"
) + anchor2
n2 = src.count(anchor2)
if n2 < 1:
    shutil.copy(PATH + ".bak_heatflux2", PATH)
    sys.exit("[heat_fac] anchor not found -> restored from backup")
src = src.replace(anchor2, add2)
print(f"[heat_fac] defined in {n2} place(s)")

open(PATH, "w").write(src)
print("Patched:", PATH)
