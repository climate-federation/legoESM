"""#1715: the pipeline lane must bridge the per-mass N_c carry to per-volume.

Two radiation entries feed the M2005 PSD.  The physics_fn entry
(`radiation/integration.py::_extract_tracer_columns`) multiplies the per-mass
carry by moist air density; the pipeline entry
(`physics_pipeline.py::compute_radiation_core`) passed it RAW, so a prognostic
droplet number reached the liquid effective radius a factor rho too small —
inert in production (the specified-Nc+CCN path overrides a dead-zeros carry),
live the moment ``predict_Nc`` feeds it.

The production entry point is too heavy to instantiate in a unit test, so this
pins the contract at two layers:

* an AST test on the SYMBOL THAT RUNS (``compute_radiation_core``), shown
  non-vacuous by construction: it fails on the exact pre-fix form
  (``flatten_3d(N_c)`` reaching ``n_cloud_col`` with no density factor);
* a physics-level measurement of what the missing bridge DOES, through the
  same ``compute_cloud_properties`` the site calls.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PIPE = (_ROOT / "packages" / "coupler" / "legoesm" / "driver"
         / "physics_pipeline.py")


def _radiation_core_fn():
    tree = ast.parse(_PIPE.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "compute_radiation_core"):
            return node
    raise AssertionError("compute_radiation_core not found — if it was "
                         "renamed, this test must follow it")


def test_pipeline_nc_assignment_carries_a_density_factor():
    """The ``n_cloud_col`` built from the N_c carry must multiply by rho.

    Fails on the pre-fix form: an assignment whose value contains
    ``flatten_3d`` applied to ``N_c`` with no multiplication anywhere in the
    expression. The aerosol-CCN override below the site assigns
    ``n_cloud_col`` too (already per-volume, no flatten_3d(N_c)) and is
    deliberately not matched.
    """
    fn = _radiation_core_fn()
    hits = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        tgt = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "n_cloud_col" not in tgt:
            continue
        src = ast.unparse(node.value)
        if "N_c" in src and "flatten_3d" in src:
            hits.append(src)
    assert hits, ("no n_cloud_col assignment builds from the N_c carry — "
                  "the site moved; update this test to follow it")
    for src in hits:
        # codex LOW: any rho-named variable would pass a bare substring
        # check. Require the specific bridge variable this site binds from
        # compute_rho two lines above, so an unrelated density cannot
        # satisfy the pin.
        assert "_rho_nc" in src and "*" in src, (
            f"the N_c carry reaches n_cloud_col with no density factor "
            f"(#1715 regression): {src}")


def test_missing_bridge_biases_r_eff_by_rho_to_the_third():
    """What the raw pass-through DOES: r_eff too large aloft by ~rho^(-1/3).

    Same in-cloud state, two columns at rho = 1.0 and rho ~ 0.55 (500 hPa,
    250 K): with the number correctly bridged, r_eff is (near) height-
    independent for identical in-cloud q_c and per-volume N_c; fed per-mass
    as if per-volume, the aloft radius inflates by ~(1/rho)^(1/3) ≈ 22%.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        compute_cloud_properties)
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics._shared import compute_rho

    cfg = CloudConfig(scheme="sundqvist", rh_crit=0.85, q_c_diagnostic=5.0e-6)
    shape = (1, 2)
    T = jnp.asarray([[288.0, 250.0]])
    p = jnp.asarray([[1.0e5, 5.0e4]])
    q_v = jnp.asarray([[8.0e-3, 1.0e-3]])
    dp = jnp.full(shape, 2.0e3)
    q_c = jnp.full(shape, 3.0e-4)
    rho = compute_rho(T, p, q_v)                       # (1, 2), [1.16, 0.70]
    n_permass = jnp.full(shape, 1.0e8)                 # #/kg

    def reff(n_volume):
        out = compute_cloud_properties(
            T=T, p_full=p, q_v=q_v, dp=dp, config=cfg,
            q_cloud=q_c, q_ice=jnp.zeros(shape), n_cloud=n_volume,
            cloud_fraction_override=jnp.ones(shape))
        return np.asarray(out.r_eff_liq)[0]

    bridged = reff(n_permass * rho)                    # correct
    raw = reff(n_permass)                              # the #1715 defect
    # Surface (rho > 1): raw number too LARGE -> radius too SMALL there;
    # aloft (rho < 1): too small -> radius too LARGE. Ratio pins the law.
    ratio = raw / bridged
    expect = np.asarray(rho[0]) ** (1.0 / 3.0)         # r ~ (q/N)^(1/3)
    np.testing.assert_allclose(ratio, expect, rtol=2e-2)
    # Non-vacuity: the two levels' densities genuinely differ.
    assert float(rho[0, 0]) / float(rho[0, 1]) > 1.4


def test_rho_helper_floors_temperature_itself():
    """The pipeline hands ``compute_rho`` an UNFLOORED temperature.

    That was a merge decision, argued on both sides from a mechanism that does
    not exist: an unfloored T = 0 was said to give rho = inf, so that a padded
    column would multiply 0 by inf and poison the run with a NaN. The helper
    clips the virtual temperature to 1 K internally, so no such value is
    reachable, and flooring at the call site is redundant rather than
    protective. Pinning it here because the claim outlived two reviews.
    """
    import jax.numpy as jnp
    from legoesm.atmosphere.physics._shared import compute_rho

    degenerate_T = jnp.array([0.0, 0.0, 0.5, 1.0, 1.0e-30])
    p = jnp.array([0.0, 5.0e4, 1.0e5, 0.0, 1.0e5])
    q_v = jnp.array([0.0, 0.0, 0.02, 0.0, 0.0])
    rho = compute_rho(degenerate_T, p, q_v)
    assert bool(jnp.all(jnp.isfinite(rho))), f"non-finite density: {rho}"

    # And over real columns the floored and unfloored calls agree exactly, so
    # the merge choice moved no number a simulation can produce.
    T = jnp.linspace(150.0, 320.0, 512)
    p_real = jnp.linspace(1.0e3, 1.05e5, 512)
    q_real = jnp.linspace(0.0, 0.03, 512)
    floored = compute_rho(jnp.maximum(T, 1.0), p_real, q_real)
    unfloored = compute_rho(T, p_real, q_real)
    np.testing.assert_array_equal(np.asarray(floored), np.asarray(unfloored))
