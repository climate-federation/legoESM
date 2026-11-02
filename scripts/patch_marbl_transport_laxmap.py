"""Compile the MARBL tracer transport once instead of 32 times (tree_map -> lax.map).
Backup: ocean_model_latlon_cgrid.py.bak_laxmap"""
import sys
from pathlib import Path

MODEL = Path("/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py")
if len(sys.argv) > 1:
    MODEL = Path(sys.argv[1])

ADV_OLD = "    return jax.tree_util.tree_map(_one, bgc)\n"
ADV_NEW = """    # One compiled copy of the transport, looped over the stacked tracers.
    leaves, treedef = jax.tree_util.tree_flatten(bgc)
    out = jax.lax.map(_one, jnp.stack(leaves))
    return jax.tree_util.tree_unflatten(treedef, [out[i] for i in range(len(leaves))])
"""
MIX_OLD = "            state = state._replace(biogeo=jax.tree_util.tree_map(_mix, state.biogeo))\n"
MIX_NEW = """            _leaves, _treedef = jax.tree_util.tree_flatten(state.biogeo)
            _mixed = jax.lax.map(_mix, jnp.stack(_leaves))
            state = state._replace(biogeo=jax.tree_util.tree_unflatten(
                _treedef, [_mixed[i] for i in range(len(_leaves))]))
"""


def main():
    if not MODEL.exists():
        sys.exit(f"{MODEL} not found; nothing changed")
    s = MODEL.read_text()
    bad = [n for o, n in ((ADV_OLD, "transport tree_map"), (MIX_OLD, "vertical-mixing tree_map"))
           if s.count(o) != 1]
    if bad:
        sys.exit(f"not patched, nothing changed; not found exactly once: {bad}")
    MODEL.with_name(MODEL.name + ".bak_laxmap").write_text(s)
    MODEL.write_text(s.replace(ADV_OLD, ADV_NEW).replace(MIX_OLD, MIX_NEW))
    print(f"patched {MODEL} (backup {MODEL.name}.bak_laxmap)")


if __name__ == "__main__":
    main()
