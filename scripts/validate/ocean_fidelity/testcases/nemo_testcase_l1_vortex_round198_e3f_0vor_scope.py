#!/usr/bin/env python3
"""Which certified cards does ``e3f_0vor`` move, and which does it not.

Round 198 lands NEMO's frozen barotropic vorticity thickness in the SHARED
split-explicit coefficient builder.  ``dyn_vor_init`` allocates ``e3f_0vor``
for every curl-point scheme in one ``SELECT CASE`` arm
(``VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:890``) and
``dyn_cor_2D_init`` divides ``ff_f`` by it in EVERY branch -- EEN at
``VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:960``, ENE/MIX at
``VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1016``, ENS at
``VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1046`` -- so the
statement is shared, not card-scoped.  Whether a given card's COEFFICIENTS
move is then a property of its coastline and its branch, and this probe
measures it instead of asserting it.

For each card it reports, at the card's own frozen geometry:

* ``n_vertices_e3f_0vor_differs`` -- where the masked four-T-cell mean over
  four (``dynvor.f90:897``, ``nn_e3f_typ = 0``) differs from the plain
  reference thickness the card used to hand the builder;
* ``coefficients_bit_identical`` -- whether the eight frozen ``ffu``/``ffv``
  coefficients the barotropic loop actually consumes change at all.

The second answer is the one that matters, and it is NOT implied by the
first: in the ENE branch each ``e3f_0vor`` index is paired with the v-mask of
the same F row (``dynspg_ts.f90:1016-1019``), so on a rectangular closed box
every vertex whose thickness moves is multiplied by a zero mask.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CARDS = ("GYRE-zco", "VORTEX-zco", "VORTEX_VEC-zco")


def _scoped_main(function):
    """Keep the allow-dirty escape from outliving this driver's own call."""
    from legoesm.ocean.fidelity.provenance import scoped_allow_dirty

    return scoped_allow_dirty(function)


def run(cards=CARDS, *, allow_dirty: bool = False) -> dict:
    import numpy as np
    import jax.numpy as jnp
    from legoesm.core.precision import get_policy
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as bt
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from legoesm.ocean.fidelity.provenance import (
        allow_dirty_stamps,
        worktree_stamp,
    )
    from legoesm.ocean.vertical import nemo_e3f_0vor_from_tmask

    out = {}
    for name in cards:
        card = build_nemo_testcase_card(name)
        recipe = card.recipe
        z_coord = recipe.z_coord
        raw = getattr(z_coord, "nemo_een_barotropic", None)
        if raw is None:
            raise SystemExit(f"{name}: card carries no literal barotropic "
                             "operands; this probe would prove nothing")
        # The production call site resolves the literal builder's branch from
        # the card's own barotropic Coriolis selection, exactly as here.
        bt_cor = getattr(recipe.model_config.barotropic,
                         "barotropic_coriolis", "avg")
        scheme = ("ene" if bt_cor.startswith("ene")
                  and not bt_cor.startswith("een") else "een")
        dtype = get_policy().control
        eta = jnp.zeros(np.asarray(raw.ff_f).shape, dtype=dtype)
        fill = jnp.asarray(raw.e3f_0, dtype=dtype)
        e3f_0vor = nemo_e3f_0vor_from_tmask(
            jnp.asarray(z_coord.nemo_e3t_0, dtype=dtype), z_coord.is_active,
            fill, nn_e3f_typ=0, grid=recipe.grid)
        differs = int(np.count_nonzero(
            np.asarray(e3f_0vor) != np.asarray(fill)))

        def coefficients(use_vor: bool):
            original = bt.nemo_e3f_0vor_from_tmask
            if not use_vor:
                bt.nemo_e3f_0vor_from_tmask = (
                    lambda e3t_0, tmask, dry_vertex_fill, **kw: dry_vertex_fill)
            try:
                return bt._nemo_literal_een_coefficients(
                    eta, z_coord, dtype, scheme=scheme, grid=recipe.grid)
            finally:
                bt.nemo_e3f_0vor_from_tmask = original

        new = coefficients(True)
        old = coefficients(False)
        keys = sorted(new)
        same = {k: bool(np.array_equal(np.asarray(new[k]),
                                       np.asarray(old[k]))) for k in keys}
        worst = max(
            float(np.max(np.abs(np.asarray(new[k]) - np.asarray(old[k]))))
            for k in keys)
        out[name] = {
            "barotropic_coriolis": bt_cor,
            "branch": scheme,
            "n_vertices_e3f_0vor_differs": differs,
            "n_vertices_total": int(np.asarray(fill).size),
            "coefficients_bit_identical": all(same.values()),
            "coefficient_bit_identity": same,
            "max_abs_coefficient_change": worst,
        }
    allow_dirty_stamps(allow_dirty)
    stamp = worktree_stamp()
    return {
        "format": "nemo-testcase-l1-round198-e3f-0vor-scope-v1",
        "worktree": stamp,
        "cards": out,
    }


@_scoped_main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    result = run(allow_dirty=args.allow_dirty)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    moved = [n for n, r in result["cards"].items()
             if not r["coefficients_bit_identical"]]
    unmoved = [n for n, r in result["cards"].items()
               if r["coefficients_bit_identical"]]
    if not moved:
        raise SystemExit(
            "VACUOUS: no card's coefficients move, so this probe cannot "
            "distinguish the statement from a no-op")
    print(f"E3F_0VOR_SCOPE: coefficients move on {moved}; "
          f"bit-identical on {unmoved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
