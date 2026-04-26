"""Iter-901 smoke test: verify the iter-901 broad-evaluation diagnostic
script is importable + structurally correct, and that its top-level
import does NOT trigger the heavy W5 + ocean-rest trajectories.

Iter-901c (Codex iter-901b stop-time fix): the original iter-901b
smoke test called `_load_script_module()` (which uses
`spec.loader.exec_module(mod)`) inside its helper-checks test.
Pre-iter-901c the diag script's main body executed at top-level so
the helper test was a 60-second trajectory run masquerading as a
smoke check.  iter-901c moves the diag script's main body under
`if __name__ == "__main__"`, then strengthens the smoke test to
actively assert the import is FAST (< 5 s) — converting the
"silently-slow smoke test" hazard into a tripwire that fires if the
guard is removed.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import importlib.util
import pathlib
import time

import jax
jax.config.update("jax_enable_x64", True)


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPT_PATH = (_REPO_ROOT / "scripts"
                / "diag_iter901_broad_eval_fortran_faithful_left.py")


def _load_script_module(name="_iter901_diag"):
    """Import the iter-901 diag script as a module.  Top-level should
    be guarded by `if __name__ == \"__main__\"` so this DOESN'T trigger
    the W5 + ocean-rest trajectories."""
    spec = importlib.util.spec_from_file_location(name, str(_SCRIPT_PATH))
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_iter901_script_exists():
    assert _SCRIPT_PATH.exists(), (
        f"iter-901 diag script missing at {_SCRIPT_PATH}")


def test_iter901_script_imports_fast_without_running_trajectories():
    """The diag script must be importable in < 5 s on a JAX-cold cache.

    iter-901c discriminating sentinel: pre-iter-901c the import took
    ~60 s because the top-level statements executed `run_w5(False);
    run_w5(True); run_ocean_rest(False); run_ocean_rest(True)` —
    each W5 run is a 288-step C36 integration.  Post-iter-901c the
    main body is under `if __name__ == "__main__"` guard, so
    importing only sets up helper functions and skips the
    trajectories.  Threshold of 5 s leaves ample margin for JAX
    boot and module-loading overhead while firing immediately if
    the guard is removed.
    """
    t0 = time.perf_counter()
    mod = _load_script_module()
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, (
        f"iter-901 diag script import took {elapsed:.2f} s — exceeds "
        f"the 5 s ceiling.  The most likely cause is that the main "
        f"body is no longer guarded by `if __name__ == \"__main__\"`, "
        f"so `_load_script_module` triggered the W5 + ocean-rest "
        f"trajectories on import.  Restore the guard.")
    # Sanity: the helpers are defined.
    assert hasattr(mod, "main"), (
        "diag script must expose a `main()` function (iter-901c "
        "convention).")
    assert hasattr(mod, "_div_damp_cube")
    assert hasattr(mod, "_build_model")
    assert hasattr(mod, "run_w5")
    assert hasattr(mod, "run_ocean_rest")


def test_iter901_script_helpers_callable():
    """`_div_damp_cube` and `_build_model` are pure helpers; check they
    have correct signatures and produce expected types without running
    the W5 / ocean-rest trajectory."""
    mod = _load_script_module()

    val = mod._div_damp_cube(48)
    assert val == 1.5e7, (
        f"_div_damp_cube(48) should return 1.5e7, got {val}")
    val36 = mod._div_damp_cube(36)
    expected36 = 1.5e7 * (48.0 / 36.0) ** 2
    assert abs(val36 - expected36) < 1e-9, (
        f"_div_damp_cube(36) deviation: {val36} vs {expected36}")

    from legoesm.grids.cubed_sphere import create_cubed_sphere
    grid = create_cubed_sphere(n=12, use_duogrid=False)

    model_iter892 = mod._build_model(grid, 12, faithful=False)
    assert (model_iter892.config.fortran_faithful_ppm_left is False), (
        "_build_model(faithful=False) must produce a config with "
        "fortran_faithful_ppm_left=False")
    assert (model_iter892.config.apply_fortran_xppm_boundary is True), (
        "_build_model must set apply_fortran_xppm_boundary=True per "
        "iter-893 production matrix alignment")

    model_iter900 = mod._build_model(grid, 12, faithful=True)
    assert (model_iter900.config.fortran_faithful_ppm_left is True), (
        "_build_model(faithful=True) must produce a config with "
        "fortran_faithful_ppm_left=True")
