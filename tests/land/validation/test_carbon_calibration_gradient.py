"""Stage-B B1 gradient smoke: prove ``jax.grad`` flows from the DIFFERENTIABLE
archetype SOC forward map back to the SOM ``CarbonConfig`` parameters.

The crux of Phase B1 -- it retires the differentiability risk: the tunable SOM
parameters, spliced in as TRACED leaves by ``equilibrate_archetypes_traced``, must
genuinely move the per-archetype equilibrium SOC AND let the gradient flow
end-to-end through the coupled ``lax.scan`` spin-up + the analytic slow-pool
reset.  We assert a FINITE, NON-ZERO gradient for at least ``som_freeze_floor``
and ``tor_som_active``.

Compute-node scale (JIT-compiles the coupled land+carbon step and its
reverse-mode; a short spin-up with ``jax.checkpoint`` per year) -- run via the
sbatch/srun wrapper, NOT the login node.  ``JAX_ENABLE_X64=1``.
"""

from __future__ import annotations

import numpy as np
import pytest


# The SOM fields the Stage-B v1 calibration targets (traced through the spin-up).
_SOM_FIELDS = (
    "tor_som_active", "tor_som_slow", "tor_som_passive",
    "f_active_to_slow", "f_slow_to_passive", "som_freeze_floor",
    "Q10_het_exp", "cwd_humification_eff",
)

_G_PER_KG = 1000.0  # gC/m2 -> kgC/m2

# Short training spin-up: enough for the fast pools + analytic slow reset to make
# the SOM parameters bite, cheap enough for a reverse-mode compute-node smoke.
_N_SPINUP, _N_VERIFY = 12, 2
_DT, _N_LAYERS, _SOIL_DEPTH = 7200.0, 6, 2.0


def _tiny_cold_warm_table():
    """A WARM tropical + a COLD boreal woody-evergreen archetype on one soil.

    Both PFTs are woody + evergreen, so they share ONE (is_woody, is_evergreen,
    soil_class) group -> a single ncol=2 coupled-step compile.  The COLD boreal
    column (mean-annual T well below freezing with a strong seasonal cycle) keeps
    the SOM freeze modifier ACTIVE, so ``som_freeze_floor`` genuinely changes the
    equilibrium SOC there (non-zero gradient); both columns feel
    ``tor_som_active`` via the SOM decomposition -> analytic-reset chain.
    """
    from legoesm.land.carbon.global_init import ArchetypeTable
    return ArchetypeTable(
        pft_id=np.array([4, 2]),   # broadleaf_evergreen_tropical, needleleaf_evergreen_boreal
        mat_k=np.array([300.0, 267.0]),
        map_yr=np.array([2200.0, 500.0]),
        t_seasonal_amp_k=np.array([2.0, 16.0]),
        aridity=np.array([2.0, 1.0]),
        sw_mean_w=np.array([230.0, 130.0]),
        soil_class=np.array(["loam", "loam"], dtype=object),
    )


def _carbon_params():
    """The tier-``extended`` ``land.carbon`` trainables (production transforms /
    bounds from the ``__param_spec__``), float64 to match the x64 spin-up."""
    import jax.numpy as jnp
    from legoesm.training.param_collector import build_trainable_params
    return build_trainable_params(
        active_scheme_keys={"land.carbon"}, tier="extended", dtype=jnp.float64)


def _som_overrides(params):
    """Sigmoid-constrained SOM-field overrides ({field -> traced scalar})."""
    return {k: v for k, v in params.to_overrides()["land.carbon"].items()
            if k in _SOM_FIELDS}


def test_traced_forward_soc_is_physical():
    """The differentiable forward map itself yields finite, positive
    per-archetype SOC (no grad yet) -- a sanity gate before the gradient."""
    import jax.numpy as jnp
    from legoesm.land.carbon.config import som_total
    from legoesm.land.carbon.global_init import equilibrate_archetypes_traced

    table = _tiny_cold_warm_table()
    overrides = _som_overrides(_carbon_params())
    eq = equilibrate_archetypes_traced(
        table, overrides, n_spinup=_N_SPINUP, n_verify=_N_VERIFY,
        dt=_DT, n_layers=_N_LAYERS, soil_depth=_SOIL_DEPTH)
    soc = np.asarray(som_total(eq)) / _G_PER_KG          # kgC/m2
    assert soc.shape == (2,)
    assert np.all(np.isfinite(soc)), soc
    assert np.all(soc > 0.0), soc


def test_som_param_gradient_finite_and_nonzero():
    """jax.grad of an SOC MSE loss w.r.t. the SOM params is FINITE and NON-ZERO
    for som_freeze_floor and tor_som_active -- the params flow and AD is intact.
    """
    import jax.numpy as jnp
    import equinox as eqx
    from legoesm.land.carbon.config import som_total
    from legoesm.land.carbon.global_init import equilibrate_archetypes_traced

    table = _tiny_cold_warm_table()
    n_arch = int(table.pft_id.shape[0])
    params = _carbon_params()

    # Synthetic per-archetype observed SOC target [kgC/m2] (boreal holds more).
    observed = jnp.asarray([10.0, 45.0])
    weights = jnp.ones(n_arch)          # cover-ish weights (B2 uses map-space area weights)

    def loss(p):
        overrides = _som_overrides(p)
        eq = equilibrate_archetypes_traced(
            table, overrides, n_spinup=_N_SPINUP, n_verify=_N_VERIFY,
            dt=_DT, n_layers=_N_LAYERS, soil_depth=_SOIL_DEPTH)
        model_soc = som_total(eq) / _G_PER_KG            # (n_arch,) kgC/m2
        return jnp.sum(weights * (model_soc - observed) ** 2) / jnp.sum(weights)

    loss_val, grads = eqx.filter_value_and_grad(loss)(params)
    graw = grads.raw_values
    d_floor = float(graw["land.carbon.som_freeze_floor"])
    d_tor = float(graw["land.carbon.tor_som_active"])

    assert np.isfinite(float(loss_val)) and float(loss_val) > 0.0, loss_val
    assert np.isfinite(d_floor) and abs(d_floor) > 0.0, f"d(loss)/d(som_freeze_floor)={d_floor}"
    assert np.isfinite(d_tor) and abs(d_tor) > 0.0, f"d(loss)/d(tor_som_active)={d_tor}"
    # Report the gradient magnitudes (visible with pytest -s / on failure).
    print(f"\nB1 gradient smoke: loss={float(loss_val):.6e}  "
          f"d(loss)/d(som_freeze_floor)={d_floor:.6e}  "
          f"d(loss)/d(tor_som_active)={d_tor:.6e}")


def test_validate_guards_check_concrete_but_skip_traced():
    """Non-vacuous self-test for the guards that unblock AD, keyed off
    boolability (is_concrete): every CONCRETELY-KNOWABLE out-of-range value
    still raises -- a host scalar AND a concrete jnp array alike -- so the
    fail-early guard is NOT weakened; only genuinely abstract (traced / scan-
    lifted) values, whose bound the __param_spec__ sigmoid guarantees, are
    skipped (a Python bool on them would TracerBoolConversionError).
    """
    import jax
    import jax.numpy as jnp
    from legoesm.land.carbon.config import is_concrete, validate_som_transfer_fractions

    # Host-scalar out-of-range -> raises.
    with pytest.raises(ValueError):
        validate_som_transfer_fractions(1.5, 0.3, 0.3)
    with pytest.raises(ValueError):
        validate_som_transfer_fractions(0.3, -0.1, 0.3)
    # CONCRETE jnp array out-of-range (outside a trace) -> ALSO raises (guard not
    # weakened for concrete arrays; codex round-2 requirement).
    with pytest.raises(ValueError):
        validate_som_transfer_fractions(jnp.asarray(1.5), 0.3, 0.3)

    # Concreteness: host scalars and concrete arrays (outside a trace) ARE
    # concrete/checkable; a value inside a jit/scan trace (as in the calibration
    # spin-up body) is NOT boolable -> is_concrete is False and the guard skips.
    assert is_concrete(0.3) is True
    assert is_concrete(jnp.asarray(0.3)) is True

    @jax.jit
    def run(x):
        # Inside the jit trace ``x`` is a tracer whose ``bool(...)`` raises, matching
        # the lax.scan spin-up body where the SOM overrides are lifted; the guard
        # must skip rather than choke.
        assert is_concrete(x) is False
        validate_som_transfer_fractions(x, x, x)   # would TracerBoolConversion if not skipped
        return x * 2.0

    _ = run(jnp.asarray(0.3))   # must trace + run without raising
