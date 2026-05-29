"""EKE-active step regression lock (build-spec gate E8).

Pins the prognostic-EKE model step on a fixed baroclinic channel + config (a
committed golden), so the EKE numerics — the conservative E transport, the
semi-implicit local source/sink, and the prognostic kappa_GM coupling — cannot
drift silently. Mirrors the bit-identical pattern of
``test_baroclinic_decomposition.py``: a deterministic case, a committed
``.npz`` golden, comparison at a tight relative tolerance (robust to ULP-level
cross-platform BLAS differences), a key-set lock, and a ``regenerate_golden``
entry point.

The golden was generated with ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=1``. To
regenerate after an INTENTIONAL numerics change::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        .venv/bin/python tests/ocean/unit/test_eke_regression.py
"""

from __future__ import annotations

import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

GOLDEN_PATH = pathlib.Path(__file__).resolve().parent / "fixtures" / (
    "eke_step_regression_golden.npz"
)

# Tight relative tolerance: the step is deterministic + bit-identical on the
# generating machine; allow <=1e-12 rel for cross-platform BLAS ULP drift.
_RTOL = 1.0e-12
_ATOL = 1.0e-25

_N_LAT, _N_LON, _NLEV = 8, 16, 4
_H_MAX = 4000.0
_N_STEPS = 5
_DT = 1800.0


def _to_float64(state):
    """Cast every floating Field of the state to float64 so the locked step runs
    in x64 (the tolerance is 1e-12 — float32 would not hold cross-platform)."""
    from legoesm.ocean.state import Field
    kw = {}
    for name in state._fields:
        f = getattr(state, name)
        if isinstance(f, Field) and jnp.issubdtype(f.data.dtype, jnp.floating):
            kw[name] = f.replace(data=f.data.astype(jnp.float64))
    return state._replace(**kw)


def _build_case():
    """Deterministic EKE-active baroclinic channel (x64): a fixed meridional T
    front + eke seeded with a STRUCTURED field (a meridional Gaussian band), so
    one step genuinely exercises the E transport, the semi-implicit source/sink,
    AND the prognostic kappa_GM coupling — not a degenerate at-the-floor field."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig, Field
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid = create_latlon_grid(_N_LAT, _N_LON)
    z_coord = create_ocean_z_star(n_levels=_NLEV, H_max=_H_MAX)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=_H_MAX, land_lat_threshold=80.0)
    state = _to_float64(state)
    lat = np.degrees(np.asarray(grid.lat))
    front = 3.0 * np.tanh(lat / 18.0)[:, None, None]
    T = state.T.data + jnp.asarray(front, dtype=jnp.float64)
    state = state._replace(T=state.T.replace(data=T))
    eke_cfg = EKEConfig()
    # Structured initial E: a meridional Gaussian band (deterministic), well
    # above the floor so transport + dissipation are non-degenerate from step 1.
    band = eke_cfg.e_min + 0.05 * np.exp(-((lat) / 25.0) ** 2)
    E0 = jnp.broadcast_to(jnp.asarray(band, dtype=jnp.float64)[:, None],
                          (_N_LAT, _N_LON))
    state = state._replace(
        eke=Field(data=E0, name="eke", dims=("lat", "lon"), units="m^2/s^2"))
    gm = GMRediConfig(kappa_GM=1.0e3, kappa_Redi=1.0e3, eke=eke_cfg)
    cfg = LatLonCGridOceanConfig(
        A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
        n_barotropic_substeps=8, enable_runtime_checks=False, gm_redi=gm)
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    return state, model, cfg


def _produce() -> dict:
    """Run the EKE-active step N times; return the final eke field plus the
    prognostic kappa_GM / Eady rate / mixing length read off the final state —
    a full fingerprint of the EKE numerics.

    Forced to the fp64 precision policy (save/restore) so the WHOLE step runs in
    float64 — the model otherwise casts to its fp32 storage policy (finite-volume
    default), and a float32 golden would not hold at 1e-12 cross-platform."""
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        compute_eke_step_kappa,
    )
    _prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        state, model, cfg = _build_case()
        for _ in range(_N_STEPS):
            state = model.step(state, dt=_DT)
        lm = state.land_mask.data
        kappa, sigma, L = compute_eke_step_kappa(
            state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
            state.eke.data, model.grid, model.z_coord, cfg.gm_redi,
            eos=cfg.eos, eos_linear=cfg.eos_linear, mask=lm,
            rho_0=cfg.constants.rho_0, g=cfg.constants.g)
        return {
            "eke": np.asarray(state.eke.data),
            "kappa_gm": np.asarray(kappa),
            "sigma": np.asarray(sigma),
            "L": np.asarray(L),
            "T": np.asarray(state.T.data),
            "S": np.asarray(state.S.data),
        }
    finally:
        set_policy(_prev)


def regenerate_golden() -> None:
    """Write the golden .npz from the CURRENT step. Run as a script."""
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    blob = _produce()
    np.savez_compressed(GOLDEN_PATH, **blob)
    print(f"wrote {len(blob)} golden arrays to {GOLDEN_PATH}")


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden not generated")
def test_eke_step_regression_bit_identical():
    """The EKE-active step must reproduce the committed golden to within a tight
    relative tolerance — locks the EKE numerics against silent drift."""
    golden = np.load(GOLDEN_PATH)
    produced = _produce()
    for key, arr in produced.items():
        assert key in golden.files, f"golden missing {key}"
        np.testing.assert_allclose(
            arr, golden[key], rtol=_RTOL, atol=_ATOL,
            err_msg=f"EKE-step golden drift in {key!r}",
        )
    # Key-set lock (no silent add/drop).
    assert set(golden.files) == set(produced), (
        "golden key set drifted:\n"
        f"  only in golden: {sorted(set(golden.files) - set(produced))}\n"
        f"  only produced:  {sorted(set(produced) - set(golden.files))}"
    )
    # Sanity: the locked case is a non-degenerate EKE field (the structured
    # band survived the step) that stayed positive + finite.
    assert float(np.max(produced["eke"])) > 1.0e-2
    assert np.all(produced["eke"] >= 0.0) and np.all(np.isfinite(produced["eke"]))


def test_golden_exists():
    """Guard against an accidental fixture deletion."""
    assert GOLDEN_PATH.exists(), (
        f"missing golden {GOLDEN_PATH}; regenerate with "
        f"`python {pathlib.Path(__file__).name}`"
    )


if __name__ == "__main__":
    regenerate_golden()
