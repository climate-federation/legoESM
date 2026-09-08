#!/usr/bin/env python
"""#1226 EVD N2 mode flip -- CARD ISOLATION.

Proves, by CONSTRUCTION rather than by assertion, that selecting
``convection_n2_mode="nemo_bn2"`` on the ``nemo_dino_kamm`` /
``nemo_dino_kamm_mlf`` cards moves the EVD trigger on those two cards ONLY,
and moves nothing on any other DINO recipe.

Method (one variable: the working-tree edit).  The PRE-EDIT ``dino.py`` is
loaded from ``git show HEAD:...`` as a second module, so both card sets are
live in one process.  For every recipe in ``DINO_RECIPES`` we:

  (1) diff the assembled ``(LatLonCGridOceanConfig, OceanPhysicsConfig)``
      pytrees field by field -- catches ANY collateral card edit, not just the
      one field we meant to touch;
  (2) build a real DINO state and evaluate the assembled convective
      ``(K_v, A_v)`` through the production helper
      ``k_profiles._enhanced_diffusion_K`` (the same call the implicit solve
      makes), and compare with ``np.array_equal`` -- BITWISE, not a tolerance.

A card is EXPECTED-MOVED (the two kamm cards) or EXPECTED-IDENTICAL (all the
rest); anything else is a hard failure.  Also checks the other two N^2
consumers on the kamm card (TKE ``tke_n2_mode``, GM/Redi ``gm_redi_slope_n2``)
are SEPARATE fields and are untouched by the flip.

fp64 (skill Rule 1c) -- dtypes printed.  Run under ``run_fp64.py``.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy

set_policy(PrecisionPolicy.fp64())

DINO_REL = "packages/ocean/legoesm/ocean/experiments/dino.py"
MOVED = {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
N_LON = 12          # small but real: full bowl, seam wall, DINO stratification


def _load_head_module():
    """Import the PRE-EDIT dino.py (git HEAD) as a separate module."""
    src = subprocess.run(["git", "show", f"HEAD:{DINO_REL}"],
                         capture_output=True, text=True, check=True).stdout
    f = tempfile.NamedTemporaryFile("w", suffix="_dino_head.py", delete=False)
    f.write(src)
    f.close()
    spec = importlib.util.spec_from_file_location("dino_head", f.name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dino_head"] = mod
    spec.loader.exec_module(mod)
    return mod


def _conv_KA(mod, recipe):
    """Assembled convective (K_v, A_v) for one recipe, production helper."""
    from legoesm.ocean.physics.vertical_mixing import k_profiles
    from legoesm.ocean.eos import make_eos_fn

    cfg = mod.dino_config_for_recipe(recipe)
    grid = mod.dino_lat_lon_grid(cfg, n_lon=N_LON)
    z_coord = mod.dino_lat_lon_vertical(grid, cfg)
    state = mod.dino_lat_lon_state(grid, z_coord, cfg)
    mc, ph = mod.dino_lat_lon_model_config(grid, cfg)
    eos_fn = make_eos_fn(cfg.eos)
    K, A = k_profiles._enhanced_diffusion_K(
        state, z_coord, ph.convection, eos_fn=eos_fn,
        before_tracers=(state.T.data, state.S.data),
    )
    return (np.asarray(K), np.asarray(A), ph.convection.enhanced_diffusion,
            mc, ph, z_coord, state)


def _pytree_diff(a, b, label):
    """Field-by-field diff of two config pytrees -> list of differing paths."""
    la, ta = jax.tree_util.tree_flatten_with_path(a)
    lb, tb = jax.tree_util.tree_flatten_with_path(b)
    out = []
    if ta != tb:
        out.append(f"{label}: TREEDEF differs")
        return out
    for (pa, va), (_, vb) in zip(la, lb):
        key = jax.tree_util.keystr(pa)
        try:
            same = bool(np.array_equal(np.asarray(va), np.asarray(vb)))
        except Exception:
            same = va == vb
        if not same:
            out.append(f"{label}{key}: {va!r} -> {vb!r}")
    return out


def main() -> int:
    print(f"[policy] control dtype = {get_policy().control}")
    head = _load_head_module()
    import legoesm.ocean.experiments.dino as work

    recipes = sorted(set(head.DINO_RECIPES) | set(work.DINO_RECIPES))
    print(f"[cards] {len(recipes)} recipes: {recipes}\n")

    failures = []
    for r in recipes:
        jax.clear_caches()
        Ko, Ao, edo, mco, pho, zco, sto = _conv_KA(head, r)
        jax.clear_caches()
        Kn, An, edn, mcn, phn, _, _ = _conv_KA(work, r)

        if r == recipes[0]:
            print("[dtype] K", Ko.dtype, "A", Ao.dtype,
                  "T", np.asarray(sto.T.data).dtype,
                  "dz_ref", np.asarray(zco.dz_ref).dtype)

        k_same = np.array_equal(Ko, Kn)
        a_same = np.array_equal(Ao, An)
        cfg_diffs = (_pytree_diff(mco, mcn, "model")
                     + _pytree_diff(pho, phn, "physics"))
        expect_moved = r in MOVED
        moved = (not k_same) or (not a_same) or bool(cfg_diffs)

        tag = "MOVED " if moved else "IDENT "
        verdict = "OK" if moved == expect_moved else "*** FAIL ***"
        nfire_o = int((Ko > edo.K_bg * 1.5).sum())
        nfire_n = int((Kn > edn.K_bg * 1.5).sum())
        print(f"{tag}{verdict:14s} {r:22s} n2_mode {edo.n2_mode!r} -> "
              f"{edn.n2_mode!r}   K bitwise-equal={k_same} A={a_same}   "
              f"EVD interfaces {nfire_o} -> {nfire_n}")
        for d in cfg_diffs:
            print(f"        config diff: {d}")
        if moved != expect_moved:
            failures.append(r)

    # The other two N^2 consumers must be SEPARATE fields, untouched.
    print("\n[separate consumers on nemo_dino_kamm_mlf]")
    co = head.dino_config_for_recipe("nemo_dino_kamm_mlf")
    cn = work.dino_config_for_recipe("nemo_dino_kamm_mlf")
    for f in ("convection_n2_mode", "tke_n2_mode", "gm_redi_slope_n2",
              "convection_n2_threshold", "convection_two_level_trigger"):
        o, n = getattr(co, f), getattr(cn, f)
        mark = "CHANGED" if o != n else "same   "
        print(f"  {mark}  {f:32s} {o!r} -> {n!r}")
        if f != "convection_n2_mode" and o != n:
            failures.append(f"collateral field {f}")

    print("\nRESULT:", "PASS" if not failures else f"FAIL {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
