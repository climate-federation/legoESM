import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round111_een_fraction_walk as gate,
)


def test_source_order_matches_compiled_three_fraction_expression():
    assert gate.SOURCE_ORDER == (
        "west_ff", "west_e3f0", "west_r3f", "west_mask", "west_denom", "frac_west",
        "center_ff", "center_e3f0", "center_r3f", "center_mask", "center_denom", "frac_center",
        "south_ff", "south_e3f0", "south_r3f", "south_mask", "south_denom", "frac_south",
        "sum_west_center", "sum_all",
    )


def test_round109_score_reports_magnitude_and_signed_zero_separately():
    candidate = np.array([0.0, 2.0], dtype=np.float64)
    reference = np.array([-0.0, 3.0], dtype=np.float64)
    row = gate.r109._score(candidate, reference)
    assert row["bit_unequal"] == 2
    assert row["magnitude_unequal"] == 1
    assert row["signed_zero_only"] == 1


def test_literal_builder_consumes_frozen_thickness_mask_not_slip_mask():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_een_coefficients,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_gyre_zco_card

    card = build_gyre_zco_card()
    z_coord = card.recipe.z_coord
    raw = z_coord.nemo_een_barotropic
    assert raw is not None
    eta = jnp.full(np.asarray(raw.ff_f).shape, 0.125, dtype=jnp.float64)

    def coefficients(z):
        return jax.device_get(jax.jit(
            lambda value: _nemo_literal_een_coefficients(
                value, z, jnp.float64, scheme="een", grid=card.recipe.grid)
        )(eta))

    baseline = coefficients(z_coord)
    slip_changed = coefficients(z_coord._replace(
        nemo_een_barotropic=raw._replace(fmask=jnp.zeros_like(raw.fmask))))
    thickness_changed = coefficients(z_coord._replace(
        nemo_een_barotropic=raw._replace(fe3mask=jnp.zeros_like(raw.fe3mask))))

    assert all(np.array_equal(baseline[name], slip_changed[name]) for name in baseline)
    assert any(not np.array_equal(baseline[name], thickness_changed[name])
               for name in baseline)
