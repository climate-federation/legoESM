"""The post-step spectral filter must not damp surface pressure by default.

Filtering ``lnps`` every step while the static orography ``phis_hat`` keeps
full truncation sharpness destroys the terrain-locked p_s structure that
hydrostatic balance requires: measured on T63L8, 36 steps of the 0.01
filter turned a 213 Pa fixed p_s offset into 1108 Pa (land-concentrated,
physics-independent) — the dominant term of the AIMIP dycore arms' WB2 mass
error. Standard spectral NWP applies horizontal diffusion to vor/div/T only.

The SFNO state-update path opts back in with ``filter_lnps=True`` because its
shipped checkpoints were trained and scored under the legacy behaviour.
"""
import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    apply_spectral_filter_to_state,
    compute_spectral_filter,
)
from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis


@pytest.fixture(scope="module")
def grid():
    return create_gaussian_grid(8, dealiasing="quadratic")


@pytest.fixture(scope="module")
def state(grid):
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        isothermal_rest_state_spectral,
    )
    from legoesm.grids.vertical import create_sigma_coordinate

    sigma = create_sigma_coordinate(4)
    s = isothermal_rest_state_spectral(grid, sigma)
    # Small-scale-rich lnps so the filter has something to bite into.
    lam = np.asarray(grid.lon)[None, :]
    phi = np.asarray(grid.lat)[:, None]
    lnps = np.log(1.0e5 * (1.0 + 0.03 * np.cos(5 * lam) * np.cos(phi) ** 2))
    return s._replace(
        lnps_hat=s.lnps_hat.replace(data=sh_analysis(grid, jnp.asarray(lnps))),
        T_hat=s.T_hat.replace(
            data=s.T_hat.data
            + 1e-3 * sh_analysis(grid, jnp.asarray(lnps))[:, None]),
    )


@pytest.fixture(scope="module")
def spectral_filter(grid):
    return compute_spectral_filter(grid.ls, grid.n_max, order=4,
                                   cutoff_fraction=0.01)


def test_default_leaves_lnps_untouched(state, spectral_filter):
    out = apply_spectral_filter_to_state(state, spectral_filter)
    np.testing.assert_array_equal(np.asarray(out.lnps_hat.data),
                                  np.asarray(state.lnps_hat.data))


def test_default_still_filters_vor_div_T(state, spectral_filter):
    out = apply_spectral_filter_to_state(state, spectral_filter)
    # Non-vacuous: T carries small-scale power, so the filter must change it.
    assert not np.array_equal(np.asarray(out.T_hat.data),
                              np.asarray(state.T_hat.data))


def test_optin_filters_lnps(state, spectral_filter):
    out = apply_spectral_filter_to_state(state, spectral_filter,
                                         filter_lnps=True)
    assert not np.array_equal(np.asarray(out.lnps_hat.data),
                              np.asarray(state.lnps_hat.data))


def test_sfno_step_pins_filter_lnps_true():
    """The SFNO state-update path must keep the legacy behaviour its
    checkpoints were trained under. Tripwire on the SOURCE OF THE METHOD
    THAT RUNS (``_apply_postprocess``, the one caller inside the SFNO step
    chain) — not the whole module, so a stray comment elsewhere cannot
    satisfy it. The executed filter-vs-mass ordering test lives in
    test_sfno_pe_rollout_stability."""
    import re

    from legoesm.atmosphere.dynamics.neural.sfno_pe import (
        SFNOPrimitiveEquationModel,
    )

    src = inspect.getsource(SFNOPrimitiveEquationModel._apply_postprocess)
    # Match the CALL, not a comment mentioning the flag (codex: the
    # explanatory comment next to the call contains the same literal, so a
    # bare substring check would pass with the argument deleted).
    assert re.search(
        r"apply_spectral_filter_to_state\((?:[^)]|\n)*filter_lnps=True",
        src), "SFNO _apply_postprocess no longer pins filter_lnps=True"
