"""Compile the air-sea flux call of the single-step JRA55 path once (jax.jit).
Backup: run_omip.py.bak_jittile"""
import sys
from pathlib import Path

P = Path("/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/scripts/run_omip.py")
if len(sys.argv) > 1:
    P = Path(sys.argv[1])

CALL_OLD = '''    tile_resp = ocean_tile_response(
        atm, sst_K, u_o, v_o, jra55_state["coupler_cfg"],
    )
'''
CALL_NEW = '''    tile_resp = _jit_tile_response(
        atm, sst_K, u_o, v_o, jra55_state["coupler_cfg"],
    )
'''
DEF_MARK = "def _jra55_step("
HELPER = '''_TILE_RESPONSE_JIT = {}


def _jit_tile_response(atm, sst_K, u_o, v_o, coupler_cfg):
    """ocean_tile_response compiled once per coupler config (single-step path)."""
    import jax as _jax
    from legoesm.coupler.coupler import ocean_tile_response
    key = id(coupler_cfg)
    f = _TILE_RESPONSE_JIT.get(key)
    if f is None:
        f = _jax.jit(lambda a, s, u, v: ocean_tile_response(a, s, u, v, coupler_cfg))
        _TILE_RESPONSE_JIT[key] = f
    return f(atm, sst_K, u_o, v_o)


'''

s = P.read_text()
bad = [n for pat, n in ((CALL_OLD, "single-step ocean_tile_response call"), (DEF_MARK, "def _jra55_step("))
       if s.count(pat) != 1]
if "_jit_tile_response" in s:
    bad.append("already patched")
if bad:
    sys.exit(f"not patched, nothing changed: {bad}")
P.with_name(P.name + ".bak_jittile").write_text(s)
s = s.replace(CALL_OLD, CALL_NEW).replace(DEF_MARK, HELPER + DEF_MARK)
P.write_text(s)
print(f"patched {P} (backup {P.name}.bak_jittile)")
