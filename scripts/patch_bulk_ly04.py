"""Add an OMIP-2 / CORE-II Large & Yeager (2004) bulk-flux option to legoESM's ocean tile.

The offline budget (flux_budget_offline.py) showed that legoESM's own large_yeager path puts
~11 W m-2 more heat into the ocean than the CORE-II / OMIP-2 formulation (sensible +5.6, latent
+2.7, albedo 0.06 vs 0.066 +1.1, emissivity 0.97 vs 1 +1.7), which drives the thermocline warming.

With the environment variable LEGOESM_BULK set, ocean_tile_response (coupler.py) replaces
  LEGOESM_BULK=ly04      sensible + latent heat flux, albedo (0.066), upward LW (emissivity 1)
  LEGOESM_BULK=ly04_tau  the same plus the wind stress (rho_a C_D |U| u)
by the CORE-II formulation (Large & Yeager 2004/2009; wind at 10 m, T and q at 2 m shifted to
10 m; rho_a = 1.22; q_sat = 0.98 * 640380 / rho_a * exp(-5107.4 / SST)).  Evaporation and the
freshwater flux are recomputed from the new latent heat flux.  Without the variable nothing changes.
It is applied wherever ocean_tile_response is called (single-step and scan-block paths).

Usage: python patch_bulk_ly04.py /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/src/legoesm/coupler/coupler.py
"""
import re
import shutil
import sys

FUNC = '''

# ----------------------------------------------------------------------------- [patch_bulk_ly04]
def _ly04_omip_fluxes(u, v, t2, q2, sst, zt=2.0, n_iter=5):
    """CORE-II / OMIP-2 Large & Yeager (2004) bulk fluxes.  Wind at 10 m; t2 [K], q2 [kg/kg] at zt
    (shifted to 10 m); sst [K].  Returns (shflx, lhflx) positive upward and (tau_x, tau_y)."""
    import jax.numpy as _jnp
    kappa, grav = 0.4, 9.81
    rho_a, cp_a, L_v = 1.22, 1000.5, 2.5e6

    def psi_m(z):
        x = _jnp.sqrt(_jnp.sqrt(_jnp.maximum(1.0 - 16.0 * z, 1e-12)))
        un = 2.0 * _jnp.log((1 + x) / 2) + _jnp.log((1 + x * x) / 2) - 2.0 * _jnp.arctan(x) + _jnp.pi / 2
        return _jnp.where(z < 0, un, -5.0 * z)

    def psi_h(z):
        x = _jnp.sqrt(_jnp.sqrt(_jnp.maximum(1.0 - 16.0 * z, 1e-12)))
        return _jnp.where(z < 0, 2.0 * _jnp.log((1 + x * x) / 2), -5.0 * z)

    def cdn(W):
        return (2.7 / W + 0.142 + 0.0764 * W) / 1e3

    U = _jnp.maximum(_jnp.sqrt(u * u + v * v), 0.5)
    qs = 0.98 * 640380.0 / rho_a * _jnp.exp(-5107.4 / sst)
    th = t2 + 0.0098 * zt
    th10, q10 = th, q2
    cd_n = cdn(U)
    rt = _jnp.sqrt(cd_n)
    stab = (th - sst > 0).astype(sst.dtype)
    cd, ch, ce = cd_n, (18.0 * stab + 32.7 * (1 - stab)) * rt / 1e3, 34.6 * rt / 1e3
    for _ in range(n_iter):
        cd_rt = _jnp.sqrt(cd)
        ustar = cd_rt * U
        tstar = ch / cd_rt * (th10 - sst)
        qstar = ce / cd_rt * (q10 - qs)
        tv = th10 * (1 + 0.608 * q10)
        bstar = grav * (tstar / tv + qstar / (q10 + 1 / 0.608))
        z10 = _jnp.clip(kappa * bstar * 10.0 / ustar ** 2, -10.0, 10.0)
        ztt = _jnp.clip(kappa * bstar * zt / ustar ** 2, -10.0, 10.0)
        shift = _jnp.log(zt / 10.0) - psi_h(ztt) + psi_h(z10)
        th10 = th - tstar / kappa * shift
        q10 = q2 - qstar / kappa * shift
        U10N = _jnp.maximum(U / (1 + rt * (-psi_m(z10)) / kappa), 0.5)
        cd_n = cdn(U10N)
        rt = _jnp.sqrt(cd_n)
        stab = (z10 > 0).astype(sst.dtype)
        ch_n, ce_n = (18.0 * stab + 32.7 * (1 - stab)) * rt / 1e3, 34.6 * rt / 1e3
        cd = (rt / (1 + rt * (-psi_m(z10)) / kappa)) ** 2
        xx = -psi_h(z10) / kappa
        ch = ch_n / (1 + ch_n * xx / rt) * _jnp.sqrt(cd / cd_n)
        ce = ce_n / (1 + ce_n * xx / rt) * _jnp.sqrt(cd / cd_n)
    shflx = -rho_a * cp_a * ch * U * (th10 - sst)
    lhflx = -rho_a * L_v * ce * U * (q10 - qs)
    tau_x = rho_a * cd * U * u
    tau_y = rho_a * cd * U * v
    return shflx, lhflx, tau_x, tau_y
'''

HOOK = '''{ind}# OMIP-2 / CORE-II bulk fluxes (LEGOESM_BULK=ly04 | ly04_tau)  [patch_bulk_ly04]
{ind}_bulk_mode = __import__("os").environ.get("LEGOESM_BULK", "")
{ind}if _bulk_mode in ("ly04", "ly04_tau"):
{ind}    _sh, _lh, _tx, _ty = _ly04_omip_fluxes(
{ind}        forcing.u_lowest - ocean_u, forcing.v_lowest - ocean_v,
{ind}        forcing.T_lowest, forcing.q_lowest, ocean_sst)
{ind}    shflx, lhflx = _sh.astype(ocean_sst.dtype), _lh.astype(ocean_sst.dtype)
{ind}    if _bulk_mode == "ly04_tau":
{ind}        tau_x, tau_y = _tx.astype(ocean_sst.dtype), _ty.astype(ocean_sst.dtype)
{ind}    alpha_ocean = jnp.full(shape, 0.066, dtype=ocean_sst.dtype)
{ind}    lw_up = constants.sigma_sb * ocean_sst ** 4
{ind}    evap_rate = lhflx / constants.L_v
{ind}    freshwater_flux = forcing.precip_total - evap_rate
'''

p = sys.argv[1]
s = open(p).read()
if "patch_bulk_ly04" in s:
    sys.exit("already patched")
i0 = s.index("def ocean_tile_response(")
m = re.compile(r"^(\s*)return TileResponse\(", re.M).search(s, i0)
assert m, "return TileResponse( not found after def ocean_tile_response"
nxt = s.find("\ndef ", i0 + 10)
assert m.start() < nxt, "return TileResponse( is not inside ocean_tile_response"
body = s[i0:m.start()]
for name in ("shflx", "lhflx", "tau_x", "alpha_ocean", "lw_up", "evap_rate", "freshwater_flux", "shape"):
    assert re.search(rf"\b{name}\b", body), f"{name} not defined in ocean_tile_response before the return"
ind = m.group(1).lstrip("\n")
emiss_line = re.search(r"emissivity=jnp\.full\(shape, config\.ocean_emissivity", s[m.start():nxt])
out = s[:m.start()] + "\n" + HOOK.format(ind=ind) + s[m.start():]
out = out.rstrip("\n") + "\n" + FUNC
compile(out, p, "exec")
shutil.copy(p, p + ".bak_bulkly04")
open(p, "w").write(out)
print(f"patched {p}: LEGOESM_BULK=ly04|ly04_tau hook before the return of ocean_tile_response; backup {p}.bak_bulkly04")
if emiss_line:
    print("  note: TileResponse.emissivity still reports config.ocean_emissivity (diagnostic only; lw_up uses 1)")
