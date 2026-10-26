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

The three fixtures were re-baselined on 2026-09-18 after a first-bad
bisection identified 66ad4bf7f, the intentional correction that centres the
split-explicit barotropic averaging window on ``t + dt``.  All three locked
EKE paths advance through that window; their old fixtures were stale.
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
# Parallel golden for the Rhines-limited mixing length (the ACC-adopted scheme,
# gate L6). Locks the rhines eke_len path through the model step independently of
# the default "rossby" golden above.
RHINES_GOLDEN_PATH = pathlib.Path(__file__).resolve().parent / "fixtures" / (
    "eke_step_regression_rhines_golden.npz"
)
# Parallel golden for the ACC-adopted prognostic-Redi (K_iso=K_gm) path: rhines
# eke_len + isopycnal_diffusion=True (Veros enable_eke_isopycnal_diffusion). Locks
# the kappa_redi_override coupling through the model step (gate R2).
RHINES_KISO_GOLDEN_PATH = pathlib.Path(__file__).resolve().parent / "fixtures" / (
    "eke_step_regression_rhines_kiso_golden.npz"
)

# Relative tolerance for cross-platform BLAS ULP drift. The drift was MEASURED at
# ~2e-11 rel (a 7.73e-9 eke-step value differing in its ~11th significant digit
# between BLAS builds), so the original 1e-12 sat BELOW the ULP floor and failed
# spuriously on any machine other than the one that wrote the golden. 1e-9 is
# comfortably above the measured drift while still catching any real numerics
# change (eke-step values are O(1e-9..1e-3), so a genuine change is orders of
# magnitude larger than 1e-9 rel).
_RTOL = 1.0e-9
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


def _build_case(eke_cfg=None):
    """Deterministic EKE-active baroclinic channel (x64): a fixed meridional T
    front + eke seeded with a STRUCTURED field (a meridional Gaussian band), so
    one step genuinely exercises the E transport, the semi-implicit source/sink,
    AND the prognostic kappa_GM coupling — not a degenerate at-the-floor field.

    ``eke_cfg`` selects the EKE config (mixing-length scheme etc.); default is the
    ``"rossby"`` EKEConfig() locked by the original golden."""
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
    eke_cfg = EKEConfig() if eke_cfg is None else eke_cfg
    # Structured initial E: a meridional Gaussian band (deterministic), well
    # above the floor so transport + dissipation are non-degenerate from step 1.
    band = eke_cfg.e_min + 0.05 * np.exp(-((lat) / 25.0) ** 2)
    E0 = jnp.broadcast_to(jnp.asarray(band, dtype=jnp.float64)[:, None],
                          (_N_LAT, _N_LON))
    state = state._replace(
        eke=Field(data=E0, name="eke", dims=("lat", "lon"), units="m^2/s^2"))
    gm = GMRediConfig(kappa_GM=1.0e3, kappa_Redi=1.0e3, eke=eke_cfg)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
        n_barotropic_substeps=8, enable_runtime_checks=False, gm_redi=gm)
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    return state, model, cfg


def _produce(eke_cfg=None) -> dict:
    """Run the EKE-active step N times; return the final eke field plus the
    prognostic kappa_GM / Eady rate / mixing length read off the final state —
    a full fingerprint of the EKE numerics. ``eke_cfg`` selects the scheme.

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
        state, model, cfg = _build_case(eke_cfg)
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


def _rhines_cfg():
    """ACC-adopted Rhines mixing length (gate L6): eke_cross=2.0 as in Veros ACC."""
    from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig
    return EKEConfig(mixing_length_scheme="rhines", eke_cross=2.0, eke_crhin=1.0)


def _rhines_kiso_cfg():
    """The full ACC EKE config (gate R2): rhines eke_len + K_iso=K_gm prognostic Redi
    (isopycnal_diffusion=True, Veros enable_eke_isopycnal_diffusion)."""
    from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig
    return EKEConfig(mixing_length_scheme="rhines", eke_cross=2.0, eke_crhin=1.0,
                     isopycnal_diffusion=True)


def regenerate_golden() -> None:
    """Write the golden .npz files (rossby + rhines + rhines-kiso) from the CURRENT step."""
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    for path, cfg in ((GOLDEN_PATH, None),
                      (RHINES_GOLDEN_PATH, _rhines_cfg()),
                      (RHINES_KISO_GOLDEN_PATH, _rhines_kiso_cfg())):
        blob = _produce(cfg)
        np.savez_compressed(path, **blob)
        print(f"wrote {len(blob)} golden arrays to {path}")


def _check_golden(golden_path, eke_cfg) -> None:
    """Compare a fresh step against the committed golden at rtol=_RTOL + a key-set
    lock + a non-degeneracy sanity. Shared by the rossby + rhines locks."""
    golden = np.load(golden_path)
    produced = _produce(eke_cfg)
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


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden not generated")
def test_eke_step_regression_bit_identical():
    """The default ("rossby") EKE-active step reproduces its committed golden to a
    tight relative tolerance — locks the EKE numerics against silent drift."""
    _check_golden(GOLDEN_PATH, None)


@pytest.mark.skipif(not RHINES_GOLDEN_PATH.exists(),
                    reason="rhines golden not generated")
def test_eke_rhines_step_regression_bit_identical():
    """The ACC-adopted Rhines-`eke_len` EKE-active step reproduces its committed
    golden — locks the rhines mixing-length path through the model step (gate L6)."""
    _check_golden(RHINES_GOLDEN_PATH, _rhines_cfg())


@pytest.mark.skipif(not RHINES_KISO_GOLDEN_PATH.exists(),
                    reason="rhines-kiso golden not generated")
def test_eke_rhines_kiso_step_regression_bit_identical():
    """The full ACC EKE step (rhines + K_iso=K_gm prognostic Redi) reproduces its
    committed golden — locks the kappa_redi_override coupling (gate R2)."""
    _check_golden(RHINES_KISO_GOLDEN_PATH, _rhines_kiso_cfg())


def test_golden_exists():
    """Guard against an accidental fixture deletion."""
    for path in (GOLDEN_PATH, RHINES_GOLDEN_PATH, RHINES_KISO_GOLDEN_PATH):
        assert path.exists(), (
            f"missing golden {path}; regenerate with "
            f"`python {pathlib.Path(__file__).name}`"
        )


if __name__ == "__main__":
    regenerate_golden()
