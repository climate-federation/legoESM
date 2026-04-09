"""Cross-discretization tests (Category 7).

Tests:
  7a) Spectral roundtrip with supergrid content — truncation fidelity
  7d) Spectral roundtrip exact test
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis, sh_synthesis
from legoesm import constants


# ---------------------------------------------------------------------------
# 7a  Spectral roundtrip with supergrid content
# ---------------------------------------------------------------------------

class TestSpectralTruncation:
    """After analysis->synthesis, content beyond n_max should be removed."""

    def test_supergrid_content_removed(self):
        n_max = 5
        grid = create_gaussian_grid(n_max)
        lat2d = grid.lat2d
        lon2d = grid.lon2d

        # Build a field with content within n_max (low-order modes)
        low = jnp.cos(lat2d) * jnp.cos(lon2d)  # ~ Y_1^1

        # Add high-wavenumber content that exceeds n_max
        # n_max=5 means modes 0..5 are retained; wavenumber 10 is beyond that.
        high = 0.5 * jnp.cos(10.0 * lon2d)   # zonal wavenumber 10 >> n_max=5

        field = low + high

        # Roundtrip
        coeffs = sh_analysis(grid, field)
        reconstructed = sh_synthesis(grid, coeffs)

        # The reconstructed field should match only the low-order part,
        # not the original field (which includes the high modes).
        err_vs_original = float(jnp.max(jnp.abs(reconstructed - field)))
        err_vs_low = float(jnp.max(jnp.abs(reconstructed - low)))

        # The high-mode content (amplitude ~0.5) should be gone
        assert err_vs_original > 0.1, (
            f"Roundtrip matches original too well ({err_vs_original:.2e}); "
            "high-frequency content was not removed"
        )
        # The reconstructed field should be close to the bandlimited part
        assert err_vs_low < 1e-8, (
            f"Roundtrip does not match bandlimited part: err={err_vs_low:.2e}"
        )




# ---------------------------------------------------------------------------
# 7d  Spectral roundtrip exact test
# ---------------------------------------------------------------------------

class TestSpectralRoundtripExact:
    """Create Y_3^2 in spectral space, synthesize and re-analyze.

    For a single spectral coefficient, synthesis -> analysis should
    recover the exact coefficient to machine precision.
    """

    def test_y32_roundtrip(self):
        from legoesm.grids.gaussian import _sh_idx

        grid = create_gaussian_grid(10)
        n_sh = grid.n_sh
        idx = _sh_idx(3, 2)

        # Create spectral array with only Y_3^2 coefficient = 1.0
        coeffs = jnp.zeros(n_sh, dtype=jnp.complex128)
        coeffs = coeffs.at[idx].set(1.0 + 0.0j)

        # Roundtrip: synthesis -> analysis
        field = sh_synthesis(grid, coeffs)
        coeffs_back = sh_analysis(grid, field)

        # The recovered (3,2) coefficient should match to < 1e-12
        recovered = coeffs_back[idx]
        err_target = abs(complex(recovered) - (1.0 + 0.0j))
        assert err_target < 1e-12, (
            f"Recovered Y_3^2 coefficient error = {err_target:.2e}, "
            f"expected < 1e-12. Got {complex(recovered)}"
        )

        # All other coefficients should be < 1e-12
        other_coeffs = coeffs_back.at[idx].set(0.0)
        max_other = float(jnp.max(jnp.abs(other_coeffs)))
        assert max_other < 1e-12, (
            f"Max non-target coefficient = {max_other:.2e}, expected < 1e-12"
        )


