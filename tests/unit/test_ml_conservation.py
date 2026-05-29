"""Direct unit tests for the atmospheric conservation correctors in
``legoesm.ml.conservation``.

Previously this module had no direct coverage — only indirect exercise via
``SFNOPrimitiveEquationModel._apply_conservation`` (which pins only the
near-zero-imbalance no-op regime).  These tests drive the correctors with
NON-trivial imbalances and check the actual conserved invariant, which is what
catches the ``correct_dry_air_mass`` longitude double-count regression
(``dp`` was ``n_lon`` times too large).
"""

import jax
import jax.numpy as jnp

from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.ml.conservation import (
    correct_dry_air_mass,
    correct_moisture,
    clip_humidity,
)

GRID = create_gaussian_grid(n_max=8)
N_LAT = GRID.weights.shape[0]
N_LON = GRID.n_lon


def _area_mean(field_2d):
    """Independent area-weighted global mean oracle (Gaussian weights)."""
    w = GRID.weights[:, None]
    return jnp.sum(field_2d * w) / (jnp.sum(w) * N_LON)


def _global_column_moisture(q, p_s, dsigma):
    """Independent global column-integrated moisture oracle."""
    w = GRID.weights[:, None]
    col = jnp.sum(q * dsigma[None, None, :], axis=-1) * p_s  # (n_lat, n_lon)
    return jnp.sum(col * w)


class TestCorrectDryAirMass:
    def test_uniform_offset_conserved(self):
        # p_s_new is a uniform +500 Pa offset; the correct shift is exactly
        # 500 Pa.  The pre-fix code shifted by n_lon*500, which would leave the
        # corrected global mean off by ~ (n_lon-1)*500 Pa.
        p_s_old = jnp.full((N_LAT, N_LON), 1.0e5)
        p_s_new = p_s_old + 500.0
        corrected = correct_dry_air_mass(p_s_new, p_s_old, GRID)
        assert jnp.allclose(_area_mean(corrected), _area_mean(p_s_old), rtol=1e-9)
        # Uniform field: corrected should be ~ p_s_old everywhere.
        assert jnp.allclose(corrected, p_s_old, atol=1e-4)

    def test_nonuniform_offset_conserved(self):
        # Latitude/longitude-varying perturbation with a clearly non-zero
        # area mean.
        lat_idx = jnp.arange(N_LAT)[:, None]
        lon_idx = jnp.arange(N_LON)[None, :]
        perturb = (
            300.0
            + 80.0 * jnp.sin(2.0 * jnp.pi * lat_idx / N_LAT)
            + 40.0 * jnp.cos(2.0 * jnp.pi * lon_idx / N_LON)
        )
        p_s_old = jnp.full((N_LAT, N_LON), 1.0e5)
        p_s_new = p_s_old + perturb
        corrected = correct_dry_air_mass(p_s_new, p_s_old, GRID)
        assert jnp.allclose(_area_mean(corrected), _area_mean(p_s_old), rtol=1e-10)

    def test_correction_is_uniform_shift(self):
        # The corrector only removes a global-mean offset, so (p_s_new -
        # corrected) must be spatially constant.
        p_s_old = jnp.full((N_LAT, N_LON), 1.0e5)
        p_s_new = p_s_old + jnp.arange(N_LAT * N_LON).reshape(N_LAT, N_LON) * 1e-2
        corrected = correct_dry_air_mass(p_s_new, p_s_old, GRID)
        shift = p_s_new - corrected
        assert jnp.allclose(shift, shift.reshape(-1)[0], atol=1e-6)

    def test_idempotent(self):
        # After one correction the global-mean imbalance is ~0, so a second
        # pass must be a no-op.  (The pre-fix corrector was NOT idempotent.)
        p_s_old = jnp.full((N_LAT, N_LON), 1.0e5)
        p_s_new = p_s_old + 250.0
        once = correct_dry_air_mass(p_s_new, p_s_old, GRID)
        twice = correct_dry_air_mass(once, p_s_old, GRID)
        assert jnp.allclose(once, twice, atol=1e-7)

    def test_differentiable(self):
        p_s_old = jnp.full((N_LAT, N_LON), 1.0e5)
        p_s_new = p_s_old + 300.0

        def loss(pn):
            return jnp.sum(correct_dry_air_mass(pn, p_s_old, GRID) ** 2)

        g = jax.grad(loss)(p_s_new)
        assert jnp.all(jnp.isfinite(g))


class TestCorrectMoisture:
    def test_global_column_moisture_conserved(self):
        nlev = 4
        dsigma = jnp.full((nlev,), 1.0 / nlev)
        p_s = jnp.full((N_LAT, N_LON), 1.0e5)
        key = jax.random.PRNGKey(0)
        q_old = 1e-3 * (1.0 + 0.5 * jax.random.uniform(key, (N_LAT, N_LON, nlev)))
        q_new = q_old * 1.2  # 20% too much moisture everywhere
        corrected = correct_moisture(q_new, q_old, p_s, dsigma, GRID)
        assert jnp.allclose(
            _global_column_moisture(corrected, p_s, dsigma),
            _global_column_moisture(q_old, p_s, dsigma),
            rtol=1e-9,
        )

    def test_proportional_scaling(self):
        # correct_moisture applies a single global scalar ratio, so the
        # corrected field must be a uniform multiple of the prediction.
        nlev = 3
        dsigma = jnp.full((nlev,), 1.0 / nlev)
        p_s = jnp.full((N_LAT, N_LON), 9.5e4)
        key = jax.random.PRNGKey(1)
        q_old = 2e-3 * jax.random.uniform(key, (N_LAT, N_LON, nlev)) + 1e-4
        q_new = q_old * 0.7
        corrected = correct_moisture(q_new, q_old, p_s, dsigma, GRID)
        ratio = corrected / q_new
        assert jnp.allclose(ratio, ratio.reshape(-1)[0], rtol=1e-9)


class TestClipHumidity:
    def test_clips_negative_only(self):
        q = jnp.array([-1.0, -1e-9, 0.0, 1e-4, 5.0])
        clipped = clip_humidity(q)
        assert jnp.all(clipped >= 0.0)
        # Non-negative entries are untouched.
        assert clipped[2] == 0.0
        assert clipped[3] == 1e-4
        assert clipped[4] == 5.0
        # Negatives become exactly zero.
        assert clipped[0] == 0.0
        assert clipped[1] == 0.0
