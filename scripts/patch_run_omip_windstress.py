import shutil, sys
PATH = "/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/scripts/run_omip.py"
src = open(PATH).read()
if "_mom_fac" in src:
    sys.exit("Already patched (found _mom_fac).")
shutil.copy(PATH, PATH + ".bak_windstress")
print("backup:", PATH + ".bak_windstress")

a1 = "    _heat_fac = dt / ("
n1 = src.count(a1)
if n1 < 1: sys.exit("[mom_fac] anchor not found")
src = src.replace(a1, "    _mom_fac = dt / (float(getattr(_hc, \"rho_0\", 1025.0))\n"
                      "                     * max(float(jra55_state[\"dz_top\"]), 1e-6))\n" + a1)
print(f"[mom_fac] defined in {n1} place(s)")

a2 = ("                new_state = new_state._replace(\n"
      "                    T=new_state.T.replace(data=_Tb.at[..., 0].set(_Tb_top)))\n")
add2 = a2 + (
    "\n"
    "                # JRA55 wind stress into the top layer (previously only used for KPP u*).\n"
    "                _U = new_state.u.data\n"
    "                _V = new_state.v.data\n"
    "                _lm = jnp.asarray(new_state.land_mask.data, dtype=_U.dtype)   # 1 = ocean\n"
    "                _um = _lm * jnp.roll(_lm, 1, axis=1)\n"
    "                _um = jnp.concatenate([_um, _um[:, :1]], axis=1)               # (n_lat, n_lon+1)\n"
    "                _vm = jnp.concatenate([jnp.zeros_like(_lm[:1]), _lm[:-1] * _lm[1:],\n"
    "                                       jnp.zeros_like(_lm[:1])], axis=0)        # (n_lat+1, n_lon)\n"
    "                _tx = jnp.asarray(tile.tau_x * ramp, dtype=_U.dtype)\n"
    "                _ty = jnp.asarray(tile.tau_y * ramp, dtype=_V.dtype)\n"
    "                _tx_u = 0.5 * (jnp.roll(_tx, 1, axis=1) + _tx)\n"
    "                _tx_u = jnp.concatenate([_tx_u, _tx_u[:, :1]], axis=1)\n"
    "                _ty_v = jnp.concatenate([jnp.zeros_like(_ty[:1]),\n"
    "                                         0.5 * (_ty[:-1, :] + _ty[1:, :]),\n"
    "                                         jnp.zeros_like(_ty[:1])], axis=0)\n"
    "                _U = _U.at[..., 0].add(jnp.asarray(_mom_fac, _U.dtype) * _tx_u * _um)\n"
    "                _V = _V.at[..., 0].add(jnp.asarray(_mom_fac, _V.dtype) * _ty_v * _vm.astype(_V.dtype))\n"
    "                new_state = new_state._replace(u=new_state.u.replace(data=_U),\n"
    "                                               v=new_state.v.replace(data=_V))\n"
)
n2 = src.count(a2)
if n2 != 1:
    shutil.copy(PATH + ".bak_windstress", PATH)
    sys.exit(f"[wind] anchor count {n2} != 1 -> restored from backup")
src = src.replace(a2, add2)
print("[wind] inserted in GPU scan step")
open(PATH, "w").write(src)
print("Patched:", PATH)
