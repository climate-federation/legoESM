"""Unit tests for the Lévy 2010 stretched z* helper in
``legoesm.ocean.vertical`` (promoted from DINO experiment 2026-05-14).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from legoesm.ocean.vertical import (
    _levy_depth_at_k,
    _levy_stretching_coefficients,
    create_levy_stretched_z_star,
)


# DINO defaults — used for cross-check vs the experiment-specific test
DINO_PARAMS = dict(n_levels=36, H_max=4000.0, dz_min=10.0, k_th=35.0, a_cr=10.5)


# -------------------- coefficient solver ----------------------------

def test_coefficients_satisfy_z_at_1_is_zero():
    K_formula = DINO_PARAMS["n_levels"] + 1
    a0, a1, a2 = _levy_stretching_coefficients(
        K_formula=K_formula, H=DINO_PARAMS["H_max"],
        dz_min=DINO_PARAMS["dz_min"],
        k_th=DINO_PARAMS["k_th"], a_cr=DINO_PARAMS["a_cr"],
    )
    z = _levy_depth_at_k(1.0, a0, a1, a2,
                        DINO_PARAMS["k_th"], DINO_PARAMS["a_cr"])
    assert z == pytest.approx(0.0, abs=1e-9)


def test_coefficients_satisfy_z_at_K_formula_is_H():
    K_formula = DINO_PARAMS["n_levels"] + 1
    a0, a1, a2 = _levy_stretching_coefficients(
        K_formula=K_formula, H=DINO_PARAMS["H_max"],
        dz_min=DINO_PARAMS["dz_min"],
        k_th=DINO_PARAMS["k_th"], a_cr=DINO_PARAMS["a_cr"],
    )
    z = _levy_depth_at_k(float(K_formula), a0, a1, a2,
                        DINO_PARAMS["k_th"], DINO_PARAMS["a_cr"])
    assert z == pytest.approx(DINO_PARAMS["H_max"], abs=1e-6)


def test_coefficients_satisfy_dz_dk_at_1_is_dz_min():
    K_formula = DINO_PARAMS["n_levels"] + 1
    a0, a1, a2 = _levy_stretching_coefficients(
        K_formula=K_formula, H=DINO_PARAMS["H_max"],
        dz_min=DINO_PARAMS["dz_min"],
        k_th=DINO_PARAMS["k_th"], a_cr=DINO_PARAMS["a_cr"],
    )
    deriv = a1 + a0 * math.tanh(
        (1 - DINO_PARAMS["k_th"]) / DINO_PARAMS["a_cr"]
    )
    assert deriv == pytest.approx(DINO_PARAMS["dz_min"], abs=1e-9)


# -------------------- create_levy_stretched_z_star -------------------

def test_grid_has_correct_n_levels():
    z = create_levy_stretched_z_star(**DINO_PARAMS)
    assert z.n_levels == DINO_PARAMS["n_levels"]


def test_total_depth_equals_H_max():
    z = create_levy_stretched_z_star(**DINO_PARAMS)
    assert float(jnp.sum(z.dz_ref)) == pytest.approx(DINO_PARAMS["H_max"], abs=1e-6)


def test_surface_interface_is_zero():
    z = create_levy_stretched_z_star(**DINO_PARAMS)
    assert float(z.z_half_ref[0]) == 0.0


def test_bottom_interface_is_minus_H():
    z = create_levy_stretched_z_star(**DINO_PARAMS)
    assert float(z.z_half_ref[-1]) == pytest.approx(-DINO_PARAMS["H_max"])


def test_layer_thickness_grows_monotonically():
    z = create_levy_stretched_z_star(**DINO_PARAMS)
    diffs = z.dz_ref[1:] - z.dz_ref[:-1]
    assert bool(jnp.all(diffs > 0))


def test_top_layer_close_to_dz_min():
    """The dz_min constraint is on the derivative; integrated top
    layer is slightly larger (~1% for default DINO params)."""
    z = create_levy_stretched_z_star(**DINO_PARAMS)
    assert float(z.dz_ref[0]) == pytest.approx(DINO_PARAMS["dz_min"], rel=0.02)


def test_validation_n_levels_too_small():
    with pytest.raises(ValueError, match="n_levels"):
        create_levy_stretched_z_star(
            n_levels=1, H_max=4000.0, dz_min=10.0, k_th=0.0, a_cr=5.0,
        )


def test_validation_negative_H_max():
    with pytest.raises(ValueError, match="H_max"):
        create_levy_stretched_z_star(
            n_levels=10, H_max=-100.0, dz_min=10.0, k_th=9.0, a_cr=5.0,
        )


def test_validation_negative_dz_min():
    with pytest.raises(ValueError, match="dz_min"):
        create_levy_stretched_z_star(
            n_levels=10, H_max=4000.0, dz_min=-1.0, k_th=9.0, a_cr=5.0,
        )


def test_alternate_parameters_n_levels_20():
    """Sanity: 20 layers, k_th=19, a_cr=8 produces a self-consistent grid."""
    z = create_levy_stretched_z_star(
        n_levels=20, H_max=3500.0, dz_min=5.0, k_th=19.0, a_cr=8.0,
    )
    assert z.n_levels == 20
    assert float(jnp.sum(z.dz_ref)) == pytest.approx(3500.0, abs=1e-6)
    assert float(z.z_half_ref[0]) == 0.0
    assert float(z.z_half_ref[-1]) == pytest.approx(-3500.0)
