"""Unit tests for the Andreae (2009) AOD → CCN diagnostic.

Reference anchors come straight from the paper (ACP 9, 543–556):
  - Fit: AOT500 = 0.0027 · CCN0.4^0.640 (Fig. 1, r²=0.88)
  - Clean continental average: CCN0.4 = 200±90 cm⁻³ at AOT = 0.075±0.025
  - Polluted continental average: CCN0.4 = 2900±2800 cm⁻³ at AOT = 0.45±0.27
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.microphysics.aerosol_activation import (
    CCNFromAODConfig,
    ccn_from_aod,
    specified_nc_field,
)

jax.config.update("jax_enable_x64", True)


def test_forward_inverse_roundtrip():
    cfg = CCNFromAODConfig()
    n_cm3 = jnp.array([50.0, 200.0, 1000.0, 5000.0])
    aot = cfg.aot_coeff * n_cm3 ** cfg.aot_exponent
    n_back = ccn_from_aod(aot) / 1.0e6
    assert jnp.allclose(n_back, n_cm3, rtol=1e-10)


def test_paper_clean_continental_anchor():
    # AOT 0.075 → ~200 cm⁻³ (paper Table 2 average 200±90)
    n = float(ccn_from_aod(jnp.array(0.075))) / 1.0e6
    assert 110.0 < n < 290.0, n


def test_paper_polluted_continental_anchor():
    # AOT 0.45 → within 2900±2800 cm⁻³ (paper Table 2 average)
    n = float(ccn_from_aod(jnp.array(0.45))) / 1.0e6
    assert 100.0 < n < 5700.0, n


def test_monotone_in_aod():
    aod = jnp.linspace(0.0, 1.5, 200)
    n = ccn_from_aod(aod)
    assert jnp.all(jnp.diff(n) >= 0.0)


def test_floor_and_cap():
    cfg = CCNFromAODConfig()
    n_lo = float(ccn_from_aod(jnp.array(0.0))) / 1.0e6
    n_hi = float(ccn_from_aod(jnp.array(50.0))) / 1.0e6
    assert n_lo == pytest.approx(cfg.n_ccn_min_cm3)
    assert n_hi == pytest.approx(cfg.n_ccn_max_cm3)


def test_differentiable_and_finite_grad_at_zero():
    g = jax.grad(lambda a: ccn_from_aod(a) / 1.0e6)(0.0)
    assert jnp.isfinite(g)
    g_mid = jax.grad(lambda a: ccn_from_aod(a) / 1.0e6)(0.2)
    assert jnp.isfinite(g_mid) and g_mid > 0.0


def test_shape_preserved():
    aod = jnp.ones((6, 4, 4)) * 0.1
    n = ccn_from_aod(aod)
    assert n.shape == aod.shape


# ---------------------------------------------------------------------------
# specified_nc_field — the column-AOD -> per-column specified-Nc glue shared by
# the coupled (cube/lat-lon) physics_pipeline and the combined-physics
# (MPAS/hydrostatic) microphysics + radiation factories.
# ---------------------------------------------------------------------------

def test_specified_nc_field_column_sum_and_broadcast():
    # Per-layer AOD summed over levels then inverted to CCN and broadcast.
    ncol, nlev = 5, 8
    aer_od = jnp.full((ncol, nlev), 0.075 / nlev)  # column AOD = 0.075
    n_c = specified_nc_field(aer_od, (ncol, nlev))
    assert n_c.shape == (ncol, nlev)
    # Same value at every level of a column (broadcast).
    assert jnp.allclose(n_c, n_c[:, :1])
    # Matches ccn_from_aod of the column-summed AOD.
    expect = ccn_from_aod(jnp.sum(aer_od, axis=-1))
    assert jnp.allclose(n_c[:, 0], expect)
    assert jnp.all(n_c > 0.0)


def test_specified_nc_field_monotone_in_column_aod():
    # Heavier column AOD -> more droplets (until the cap).
    nlev = 4
    low = jnp.full((1, nlev), 0.02 / nlev)
    high = jnp.full((1, nlev), 0.40 / nlev)
    assert float(specified_nc_field(high, (1, nlev))[0, 0]) > float(
        specified_nc_field(low, (1, nlev))[0, 0]
    )


def test_specified_nc_field_differentiable():
    nlev = 4

    def _mean_nc(scale):
        aer = jnp.full((2, nlev), 0.05 / nlev) * scale
        return jnp.mean(specified_nc_field(aer, (2, nlev)))

    g = jax.grad(_mean_nc)(1.0)
    assert jnp.isfinite(g) and g > 0.0


def test_specified_nc_field_units_are_per_cubic_metre():
    # UNIT LOCK.  Both consumers expect a per-VOLUME droplet number [#/m³]:
    #   * Morrison warm rain — ``effective_Nc`` / ``Nc_0`` (1e8 /m³, bounds
    #     1e7–1e9 in MorrisonConfig.__param_spec__);
    #   * RRTMGP cloud optics — ``n_cloud`` is per-VOLUME [#/m³] and only
    #     consumed where ``> 8`` (radiation/integration._extract_tracer_columns).
    # ``ccn_from_aod`` works in cm⁻³, so the helper MUST apply the cm⁻³→m⁻³
    # ×1e6 conversion.  Dropping it (returning ~10²–10³ cm⁻³ raw) would put
    # N_c six orders of magnitude low, below the >8 optics guard and outside
    # the Nc_0 bounds — a silent unit bug pytest would otherwise miss.  A
    # clean continental column (AOT ~0.075 → ~200 cm⁻³) must land at ~2e8 /m³.
    nlev = 8
    aer_od = jnp.full((3, nlev), 0.075 / nlev)  # column AOT = 0.075
    n_c = specified_nc_field(aer_od, (3, nlev))
    val = float(n_c[0, 0])
    assert 1.0e7 <= val <= 1.0e9, val          # inside Nc_0 spec bounds
    assert val > 8.0                            # above cloud-optics guard
    # ~200 cm⁻³ continental anchor -> ~2e8 /m³ (1.1e8–2.9e8 covers the ±90).
    assert 1.1e8 <= val <= 2.9e8, val
