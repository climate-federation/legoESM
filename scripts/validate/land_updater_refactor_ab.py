"""A/B the pytree updater against a reference implementation from git.

GLM's decisive check: the refactor claims to be behaviour-preserving, so the
old closure and the new precompute+apply must agree BIT-FOR-BIT on the same
inputs.  Main's version is loaded from git (not from the working tree), so
this compares the two implementations, not two calls of one.
"""
import subprocess, sys, types, pathlib
import numpy as np, jax, jax.numpy as jnp

ref_src = subprocess.run(
    ["git", "show", "cf/main:packages/land/legoesm/land/boundary_data/step_updater.py"],
    capture_output=True, text=True, check=True).stdout
ref = types.ModuleType("step_updater_main")
ref.__file__ = "cf/main:step_updater.py"
exec(compile(ref_src, ref.__file__, "exec"), ref.__dict__)

sys.path.insert(0, "tests/land/boundary_data")
from test_step_updater_transient import _forest_then_grass          # noqa: E402
from legoesm.land.boundary_data.step_updater import (               # noqa: E402
    make_step_land_params_updater, precompute_canopy_updater, apply_canopy_updater)
from legoesm.land.canopy import CanopyConfig                        # noqa: E402

ROOTS = {"root_depth": np.array([0.5] * 17), "theta_wp": np.array([0.1] * 17),
         "theta_fc": np.array([0.3] * 17)}
gsd, cfg, theta = _forest_then_grass(), CanopyConfig(), jnp.asarray([0.28])
bad = 0
for roots in (None, ROOTS):
    old = ref.make_step_land_params_updater(gsd, cfg, pft_root_params=roots)
    new = make_step_land_params_updater(gsd, cfg, pft_root_params=roots)
    pre = precompute_canopy_updater(gsd, pft_root_params=roots)
    for doy, year in ((15.0, 1850.0), (200.0, 1925.0), (355.0, 2000.0)):
        a = (theta, jnp.asarray(doy), jnp.asarray(year))
        for tag, got in (("wrapper", new(*a)), ("split", apply_canopy_updater(pre, *a))):
            want = old(*a)
            lw, lg = jax.tree.leaves(want), jax.tree.leaves(got)
            if len(lw) != len(lg):
                print(f"LEAF COUNT roots={roots is not None} {tag} {len(lw)} vs {len(lg)}"); bad += 1; continue
            diffs = [i for i, (x, y) in enumerate(zip(lw, lg))
                     if not np.array_equal(np.asarray(x), np.asarray(y))]
            print(f"roots={'yes' if roots else 'no ':3} doy={doy:5.0f} year={year:6.0f} "
                  f"{tag:7s} leaves={len(lw)} differing={len(diffs)}")
            bad += len(diffs)
print("\nBIT-IDENTICAL TO MAIN" if bad == 0 else f"\nDIFFERENCES: {bad}")
sys.exit(1 if bad else 0)
