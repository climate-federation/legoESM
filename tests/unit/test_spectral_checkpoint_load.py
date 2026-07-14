"""Offline spectral-checkpoint reconstruction (iter 92).

A ``discretization='spectral'`` run's ``ModelDriver.save_checkpoint`` writes the
five ``*_hat`` coefficient arrays + a ``spectral_layout`` marker via ``np.savez``
(NOT the grid layout ``load_checkpoint_auto`` reads).  ``load_restart`` now detects
that marker and rebuilds the :class:`SpectralHydrostaticState` through the shared
``reconstruct_spectral_state_from_npz`` (the SAME helper the in-driver restart uses),
so the one-shot compare CLI can load a spectral restart offline + synthesize grid
winds via ``grid_winds_from_spectral``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (  # noqa: E402
    SpectralHydrostaticState,
    reconstruct_spectral_state_from_npz,
)
from legoesm.core.field import Field  # noqa: E402

_NSH, _NLEV = 12, 5


def _spectral_npz_dict(*, tracers=True):
    """A minimal saved spectral-checkpoint dict (the keys ModelDriver writes)."""
    rng = np.arange(_NSH * _NLEV, dtype=np.float64).reshape(_NSH, _NLEV)
    d = {
        "vor_hat": (rng + 1j).astype(np.complex128),
        "div_hat": (2 * rng).astype(np.complex128),
        "T_hat": (rng - 0.5j).astype(np.complex128),
        "lnps_hat": (rng[:, 0] + 0j).astype(np.complex128),
        "phis_hat": (0 * rng[:, 0]).astype(np.complex128),
        "step": np.asarray(9),
        "day": np.asarray(4.25),
        "spectral_layout": np.asarray(1),
    }
    if tracers:
        d["tracer_names"] = np.asarray(["q_v"])
        d["trc_q_v"] = np.full((4, 6, _NLEV), 5e-3)
    return d


def _template():
    """A configured SpectralHydrostaticState of matching shapes (for the template path)."""
    z = jnp.zeros((_NSH, _NLEV), dtype=jnp.complex128)
    z1 = jnp.zeros((_NSH,), dtype=jnp.complex128)
    return SpectralHydrostaticState(
        vor_hat=Field(z, name="vor_hat"), div_hat=Field(z, name="div_hat"),
        T_hat=Field(z, name="T_hat"), lnps_hat=Field(z1, name="lnps_hat"),
        phis_hat=Field(z1, name="phis_hat"),
        tracers={"q_v": Field(jnp.zeros((4, 6, _NLEV)), name="q_v")})


def test_reconstruct_no_template_plain_fields():
    d = _spectral_npz_dict()
    state, step, day = reconstruct_spectral_state_from_npz(d)
    assert isinstance(state, SpectralHydrostaticState)
    assert step == 9 and day == pytest.approx(4.25)
    np.testing.assert_array_equal(np.asarray(state.vor_hat.data), d["vor_hat"])
    np.testing.assert_array_equal(np.asarray(state.div_hat.data), d["div_hat"])
    assert state.tracers is not None and "q_v" in state.tracers
    np.testing.assert_array_equal(np.asarray(state.tracers["q_v"].data), d["trc_q_v"])


def test_reconstruct_with_template_reuses_metadata_and_validates():
    d = _spectral_npz_dict()
    tmpl = _template()
    state, _, _ = reconstruct_spectral_state_from_npz(d, template=tmpl)
    # Field metadata (name) is reused from the template; only the data is replaced.
    assert state.vor_hat.name == "vor_hat"
    np.testing.assert_array_equal(np.asarray(state.T_hat.data), d["T_hat"])
    # A coefficient with the wrong shape fails LOUDLY (resolution/nlev mismatch).
    bad = dict(d)
    bad["vor_hat"] = np.zeros((_NSH + 1, _NLEV), dtype=np.complex128)
    with pytest.raises(ValueError, match="resolution/nlev mismatch"):
        reconstruct_spectral_state_from_npz(bad, template=tmpl)


def test_reconstruct_with_template_refuses_to_drop_water():
    """A checkpoint carrying tracers but a template (configured run) with none MUST
    raise — never silently drop water."""
    d = _spectral_npz_dict(tracers=True)
    dry = _template()._replace(tracers=None)
    with pytest.raises(ValueError, match="refusing to silently drop water"):
        reconstruct_spectral_state_from_npz(d, template=dry)


def test_reconstruct_non_spectral_raises():
    d = _spectral_npz_dict()
    del d["spectral_layout"]
    with pytest.raises(ValueError, match="not a spectral checkpoint"):
        reconstruct_spectral_state_from_npz(d)


def test_reconstruct_bytestring_tracer_names_decoded(tmp_path):
    """A tracer_names array saved as BYTE strings (numpy dtype-dependent) is decoded
    so the trc_<name> lookup key is correct (Codex iter 92 — `str(b'q_v')` would
    otherwise look up `trc_b'q_v'` → KeyError)."""
    d = _spectral_npz_dict()
    d["tracer_names"] = np.asarray([b"q_v"])     # byte strings
    # Round-trip through np.savez/np.load (how a real checkpoint is read).
    path = tmp_path / "c.npz"
    np.savez(path, **d)
    with np.load(path) as z:
        state, _, _ = reconstruct_spectral_state_from_npz(z)
    assert state.tracers is not None and "q_v" in state.tracers


def test_reconstruct_template_tracer_keyset_mismatch_raises():
    """A template with {q_v, q_c} but a checkpoint carrying only {q_v} is REJECTED —
    no silent add/drop of a tracer across restart."""
    d = _spectral_npz_dict(tracers=True)             # only q_v
    tmpl = _template()._replace(
        tracers={"q_v": _template().tracers["q_v"],
                 "q_c": Field(jnp.zeros((4, 6, _NLEV)), name="q_c")})
    with pytest.raises(ValueError, match="refusing to add/drop a tracer"):
        reconstruct_spectral_state_from_npz(d, template=tmpl)


def test_reconstruct_template_tracer_shape_mismatch_raises():
    d = _spectral_npz_dict(tracers=True)
    d["trc_q_v"] = np.full((4, 6, _NLEV + 1), 5e-3)   # wrong nlev
    with pytest.raises(ValueError, match="tracer q_v shape"):
        reconstruct_spectral_state_from_npz(d, template=_template())


def test_reconstruct_dry_checkpoint_moist_template_raises():
    """A checkpoint with NO tracers loaded against a MOIST template is rejected — the
    symmetric guard to refuse-to-drop-water (no silently inventing water)."""
    d = _spectral_npz_dict(tracers=False)            # no tracer_names
    with pytest.raises(ValueError, match="refusing to silently invent water"):
        reconstruct_spectral_state_from_npz(d, template=_template())   # moist


def test_load_restart_spectral_non_spectral_grid_raises(tmp_path):
    """strict=True + a SUPPLIED non-spectral grid (no .lap) for a spectral checkpoint
    is a config error → raise (not a silent skip)."""
    from types import SimpleNamespace

    from legoesm.driver.restart import load_restart
    from legoesm.grids.vertical import create_sigma_coordinate
    d = _spectral_npz_dict()
    path = tmp_path / "checkpoint_day_0000.npz"
    np.savez(path, **d)
    with pytest.raises(ValueError, match="non-spectral grid"):
        load_restart(path, grid=SimpleNamespace(), sigma=create_sigma_coordinate(_NLEV),
                     strict=True)


def test_load_restart_spectral_strict_shape_mismatch_raises(tmp_path):
    """strict=True validates the coefficient shapes against the configured grid/sigma
    BEFORE reconstructing — a wrong-resolution spectral checkpoint fails LOUDLY at the
    loader, not as an opaque downstream SH-synthesis error (Codex iter 92)."""
    from legoesm.driver.restart import load_restart
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    grid = create_gaussian_grid(n_max=21, dealiasing="linear")
    sigma = create_sigma_coordinate(_NLEV)
    d = _spectral_npz_dict()                          # _NSH=12 coeffs ≠ grid's n_sh
    path = tmp_path / "checkpoint_day_0000.npz"
    np.savez(path, **d)
    with pytest.raises(ValueError, match="!= expected"):
        load_restart(path, grid=grid, sigma=sigma, strict=True)


def test_load_restart_spectral_branch(tmp_path):
    """``load_restart`` on a spectral checkpoint returns the SpectralHydrostaticState
    + grid-space q_v + step/day (config/diag/metadata = None), so the offline compare
    CLI can load + convert it."""
    from legoesm.driver.restart import load_restart
    d = _spectral_npz_dict()
    path = tmp_path / "checkpoint_day_0004.npz"
    np.savez(path, **d)
    out = load_restart(path, grid=None, sigma=None, strict=True)
    state, q_v, step, day = out[0], out[1], out[2], out[3]
    assert isinstance(state, SpectralHydrostaticState)
    assert step == 9 and day == pytest.approx(4.25)
    np.testing.assert_array_equal(np.asarray(q_v), d["trc_q_v"])    # grid-space tracer
    assert out[4] is None and out[8] is None    # no config, no metadata for spectral
