"""Q8 — bit-identical regression gate for ``latlon_cgrid_ocean_baroclinic_tendencies``.

The Q8 task decomposes the ~1299-LOC ``latlon_cgrid_ocean_baroclinic_tendencies``
into named pure substages. The decomposition MUST be a pure extraction: same
operators, same order, no reassociation. This test is the gate that proves it —
it pins the tendency output on several frozen states/configs (a committed golden)
and fails if the decomposed implementation deviates beyond float round-off.

The golden was generated from the PRE-decomposition function (the reference
behaviour). Regenerate it ONLY when a deliberate, reviewed numeric change is
made — never to paper over an accidental decomposition drift:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
        .venv/bin/python tests/ocean/unit/test_baroclinic_decomposition.py

Cases are chosen to exercise as many config-gated stages as feasible on a small
deterministic state: implicit vertical mixing + GM/Redi + sponge + the momentum
diagnostics path, and explicit vertical mixing + biharmonic + Smagorinsky +
meridional viscosity. (Branches still not exercised here — physics_fn / external
surface_forcing, WENO advection — are a documented residual; the function is
unchanged for those paths and the extraction copies them verbatim.)
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

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.sponge import SpongeForcing
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star

GOLDEN_PATH = pathlib.Path(__file__).resolve().parent / "fixtures" / (
    "baroclinic_decomposition_golden.npz"
)

# Tolerance: a pure extraction is bit-identical on the generating machine; the
# committed golden is compared at a tight relative tolerance robust to ULP-level
# cross-platform BLAS differences (spec allows <=1e-12 rel in x64).
_RTOL = 1.0e-12
_ATOL = 1.0e-25

_N_LAT, _N_LON, _NLEV = 8, 16, 4
_H_MAX = 1500.0


def _base_state(grid, z_coord):
    """Deterministic rest state with small velocity + T/S perturbations so every
    stage produces a nonzero tendency (a zero state would hide data-flow bugs)."""
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=18.0, T_deep=2.0, S_uniform=35.0,
        H_max=_H_MAX, land_lat_threshold=85.0,
    )
    u_pert = 0.05 * jax.random.normal(
        jax.random.PRNGKey(101), state.u.data.shape, dtype=jnp.float64,
    )
    v_pert = 0.05 * jax.random.normal(
        jax.random.PRNGKey(202), state.v.data.shape, dtype=jnp.float64,
    )
    T_pert = 0.5 * jax.random.normal(
        jax.random.PRNGKey(303), state.T.data.shape, dtype=jnp.float64,
    )
    eta_pert = 0.02 * jax.random.normal(
        jax.random.PRNGKey(404), state.eta.data.shape, dtype=jnp.float64,
    )
    return state._replace(
        u=state.u.replace(data=state.u.data + u_pert),
        v=state.v.replace(data=state.v.data + v_pert),
        T=state.T.replace(data=state.T.data + T_pert),
        eta=state.eta.replace(data=state.eta.data + eta_pert),
    )


def _sponge(grid, z_coord):
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    gamma = np.zeros((n_lat, n_lon), dtype=np.float64)
    gamma[:2, :] = 1.0 / 86400.0
    gamma[-2:, :] = 1.0 / 86400.0
    return SpongeForcing(
        gamma=jnp.asarray(gamma),
        T_ref=jnp.full((n_lat, n_lon, nlev), 12.0, dtype=jnp.float64),
        S_ref=jnp.full((n_lat, n_lon, nlev), 35.0, dtype=jnp.float64),
    )


def _cases():
    """Yield (name, call_kwargs_dict). Each builds its own grid/state/config so
    the test is order-independent and self-contained."""
    grid = create_latlon_grid(_N_LAT, _N_LON)
    z_coord = create_ocean_z_star(n_levels=_NLEV, H_max=_H_MAX)
    state = _base_state(grid, z_coord)

    # Case 1: implicit vertical mixing + GM/Redi + sponge + diagnostics path.
    cfg1 = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e3, A_h_lat_scaling=True, A_h_cos_power=1, A_h_eq_boost=3.0,
        K_h=5.0e2, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
        gm_redi=GMRediConfig(), eos="wright", n_barotropic_substeps=2,
        enable_runtime_checks=False,
    )
    yield ("implicit_gmredi_sponge_diag", dict(
        state=state, grid=grid, z_coord=z_coord, config=cfg1,
        sponge=_sponge(grid, z_coord), dt=300.0, diagnose_momentum=True,
    ))

    # Case 2: explicit vertical mixing + biharmonic + Smagorinsky + meridional
    # viscosity (a different set of stages / branches).
    cfg2 = LatLonCGridOceanConfig.from_flat(
        A_h=1.0e3, A_h_merid=5.0e2, B_h=1.0e9, B_h_lat_scaling=True,
        C_smag=0.15, C_smag_lap=0.1, K_h=2.0e2, K_bih=1.0e8,
        A_v=1.0e-3, K_v=1.0e-4, bottom_drag_r=2.0e-3,
        implicit_vertical_mixing=False, eos="linear",
        n_barotropic_substeps=2, enable_runtime_checks=False,
    )
    yield ("explicit_biharmonic_smag_merid", dict(
        state=state, grid=grid, z_coord=z_coord, config=cfg2,
        dt=300.0, diagnose_momentum=False,
    ))

    # Case 3: Hollingsworth KE gradient (exercises the nkeg_HW branch that the
    # centered-default cases above skip).
    cfg3 = LatLonCGridOceanConfig.from_flat(
        A_h=1.0e3, ke_gradient_scheme="hollingsworth", K_h=2.0e2,
        bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
        n_barotropic_substeps=2, enable_runtime_checks=False,
    )
    yield ("hollingsworth_ke", dict(
        state=state, grid=grid, z_coord=z_coord, config=cfg3,
        dt=300.0, diagnose_momentum=False,
    ))

    # Case 4: WENO5 momentum + tracer advection (exercises the WENO KE/PV/tracer
    # branches that the centered/TVD cases skip).
    cfg4 = LatLonCGridOceanConfig.from_flat(
        A_h=1.0e3, momentum_advection="weno5", tracer_advection="weno5",
        K_h=2.0e2, bottom_drag_r=1.0e-3, implicit_vertical_mixing=False,
        A_v=1.0e-3, K_v=1.0e-4, n_barotropic_substeps=2,
        enable_runtime_checks=False,
    )
    yield ("weno5_momentum_tracer", dict(
        state=state, grid=grid, z_coord=z_coord, config=cfg4,
        dt=300.0, diagnose_momentum=False,
    ))

    # Case 5: flux-form horizontal momentum advection (regression-locks the
    # _bc_horizontal_momentum_advection_flux_form path + the KE-gradient-zeroing
    # dispatch). diagnose_momentum=True also locks the diagnostic slot.
    cfg5 = LatLonCGridOceanConfig.from_flat(
        A_h=1.0e3, momentum_advection="flux_form", momentum_flux_scheme="upwind",
        K_h=2.0e2, bottom_drag_r=1.0e-3, implicit_vertical_mixing=False,
        A_v=1.0e-3, K_v=1.0e-4, n_barotropic_substeps=2,
        enable_runtime_checks=False,
    )
    yield ("flux_form_momentum", dict(
        state=state, grid=grid, z_coord=z_coord, config=cfg5,
        dt=300.0, diagnose_momentum=True,
    ))


def _outputs_for_case(call_kwargs) -> dict[str, np.ndarray]:
    """Run the function and flatten its output (+ diagnostics) to a name->array
    dict for golden comparison."""
    diag_requested = call_kwargs.get("diagnose_momentum", False)
    result = latlon_cgrid_ocean_baroclinic_tendencies(**call_kwargs)
    if diag_requested:
        tend, diag = result
    else:
        tend, diag = result, None

    out: dict[str, np.ndarray] = {}
    for fld in ("du_dt", "dv_dt", "dT_dt", "dS_dt", "deta_dt"):
        out[fld] = np.asarray(getattr(tend, fld).data)
    for opt in ("K_v", "A_v"):
        val = getattr(tend, opt)
        if val is not None:
            out[opt] = np.asarray(val)
    if diag is not None:
        for name in diag._fields:
            val = getattr(diag, name)
            if val is None:
                continue
            # Diagnostics fields are Field objects; unwrap to the raw array.
            out[f"diag__{name}"] = np.asarray(getattr(val, "data", val))
    return out


def regenerate_golden() -> None:
    """Write the golden .npz from the CURRENT function. Run as a script."""
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, np.ndarray] = {}
    for name, kw in _cases():
        for key, arr in _outputs_for_case(kw).items():
            blob[f"{name}::{key}"] = arr
    np.savez_compressed(GOLDEN_PATH, **blob)
    print(f"wrote {len(blob)} golden arrays to {GOLDEN_PATH}")


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden not generated")
def test_baroclinic_decomposition_bit_identical():
    """The (decomposed) function must reproduce the committed golden to within
    float round-off on every case — the pure-extraction gate.

    RE-BASELINED 2026-09-18 after a first-bad bisection from the prior golden:
    9caa61f3e, the deliberate NEMO-fidelity/state-and-constants integration,
    moved these production tendencies.  The old fixture remained green at its
    parent and red at that commit; this fixture records the landed semantics.

    RE-BASELINED 2026-08-01 (#1388), and the earlier version of this note got
    the attribution WRONG — recorded here because the wrong version is the kind
    that gets repeated.

    What actually moved: ONE absolute shift of 3.284e-09 in the u-tendency
    (1.22e-10 in v), field-wide across all 544 elements, peaking at
    [row 7, col 0, k 3] — the SEAM column of the last row. The headline
    "6.14 relative" is not du_dt at all: it is
    ``implicit_gmredi_sponge_diag::diag__vortcor_u``, whose old values are
    ~3e-09, so the same 3.284e-09 absolute shift reads as 614%. Per case,
    max relative / max absolute / |old|max:

        implicit_gmredi_sponge_diag::diag__vortcor_u  6.14e+0 / 3.28e-9 / 3.09e-9
        hollingsworth_ke::du_dt                       5.71e-1 / 3.28e-9 / 4.27e-6
        implicit_gmredi_sponge_diag::du_dt            2.70e-1 / 3.28e-9 / 4.27e-6
        explicit_biharmonic_smag_merid::du_dt         3.54e-2 / 3.28e-9 / 4.78e-6
        weno5_momentum_tracer::dv_dt                  9.37e-3 / 1.22e-10 / 7.51e-7

    46 of 77 stored arrays changed. Relative deviations alone are misleading
    here: quote the absolute shift next to the scale it sits on.

    ATTRIBUTION (PLAUSIBLE, not confirmed by bisect): commit 1bfea62e5,
    "fix(ocean): #1226 periodic-seam index bug — barotropic solver leaking
    mass", is an intentional physics fix landed after the 2026-06-26 golden
    that touches the AL81/PV-flux path these fixture cases exercise, and the
    delta peaks at the seam column, which is its signature. 2cbbd837e (the
    #1418 vertex-Neumann seam fix) is a possible additional contributor. The
    July-28 Treguier kappa_min floor is NOT a candidate: these cases run
    GMRediConfig() with Treguier disabled.

    So this re-baseline asserts "future changes must be deliberate", NOT "the
    current values are correct". Anyone who suspects one of these drifts is a
    defect should bisect that case against the 2026-06-26 golden in git
    history — 1bfea62e5 is the first commit to try.
    """
    golden = np.load(GOLDEN_PATH)
    checked = 0
    for name, kw in _cases():
        outs = _outputs_for_case(kw)
        for key, arr in outs.items():
            gkey = f"{name}::{key}"
            assert gkey in golden.files, f"golden missing {gkey}"
            ref = golden[gkey]
            assert arr.shape == ref.shape, f"{gkey}: shape {arr.shape} != {ref.shape}"
            np.testing.assert_allclose(
                arr, ref, rtol=_RTOL, atol=_ATOL,
                err_msg=f"decomposition changed output for {gkey}",
            )
            checked += 1
    # Lock that the golden covers exactly the produced keys (no silent drop).
    produced = {
        f"{name}::{key}" for name, kw in _cases() for key in _outputs_for_case(kw)
    }
    assert set(golden.files) == produced, (
        "golden key set drifted from produced outputs:\n"
        f"  only in golden: {sorted(set(golden.files) - produced)}\n"
        f"  only produced:  {sorted(produced - set(golden.files))}"
    )
    assert checked > 0


def test_golden_exists():
    """The committed golden must be present (guards against an accidental
    delete that would silently skip the bit-identity gate above)."""
    assert GOLDEN_PATH.exists(), (
        f"missing golden {GOLDEN_PATH}; regenerate with "
        f"`python {pathlib.Path(__file__).name}`"
    )


if __name__ == "__main__":
    regenerate_golden()
