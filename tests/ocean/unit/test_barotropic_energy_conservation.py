"""Energy conservation of the nonlinear barotropic free-surface dynamics.

In the INVISCID, UNFORCED limit, the continuous shallow-water momentum equations
(advection + Coriolis + free-surface pressure gradient) conserve total mechanical
energy (KE + ½g∫η²) exactly. A faithful discretization must too.

This pins the root cause of the MITgcm barotropic-gyre oracle's residual. Direct
measurement (see docs/ocean_fidelity/mitgcm_gyre_energy_conservation.md) localized
the spurious, dt-independent energy injection to the **Coriolis ⟷ implicit-free-
surface-projection coupling** — NOT the momentum scheme: the source is byte-identical
across upwind / centered / vector-invariant advection (advection contributes
nothing), the Coriolis operator is itself energy-neutral (machine-zero work), the
free surface alone conserves exactly (f=0 -> machine-zero dE/dt), yet the sequence
"explicit Coriolis -> project onto the free-surface-balanced state" injects energy
when f != 0. Moving Coriolis between the FB-predictor and the AB2 F_slow leaves the
leak byte-identical (placement-independent). Against MITgcm's near-frictionless gyre
equilibrium the injection drives the gyre turbulent (|u|max 0.15-0.37 vs 0.031);
legoESM cannot even hold MITgcm's bridged 0.031 equilibrium (drifts to 0.11+).

The test is xfail until the C-grid implicit free-surface step conserves total energy
in the presence of Coriolis (a dycore fix; the oracle-faithful target is MITgcm's
unsplit explicit-Coriolis -> cg2d sequencing); when it does, this flips to a guard.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.eos import LinearEOSConfig
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star

NY = NX = 32
DX = 20.0e3


def _closed_box_geom_state():
    geom = create_beta_plane_cgrid_geometry(
        NY, NX, dx_m=DX, dy_m=DX, f0=1e-4, beta=1e-11, y_origin_m=-20e3
    )
    z = create_ocean_z_star(1, H_max=5000.0)
    mask = np.ones((NY, NX))
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0.0
    base = rest_state_latlon_cgrid_ocean(
        geom, z, land_mask_override=jnp.asarray(mask)
    )
    # MUNK-like western-intensified gyre IC from psi = A sin(pi y) (1 - exp(-x/delta)),
    # delta = 1.5 cells, so the western boundary current carries GRID-SCALE
    # gradients / divergent content (where the free-surface-projection energy
    # injection manifests — a smooth IC conserves fine; only sharp WBC-like
    # features with grid-scale divergence trigger the defect).
    delta = 1.5 / NX
    amp = 0.04
    yu = (np.arange(NY) + 0.5) / NY
    xu = (np.arange(NX + 1)) / NX
    gx_u = 1.0 - np.exp(-xu / delta)
    u = -amp * np.cos(np.pi * yu)[:, None] * gx_u[None, :]
    yv = (np.arange(NY + 1)) / NY
    xv = (np.arange(NX) + 0.5) / NX
    gpx_v = (1.0 / delta) * np.exp(-xv / delta) * delta  # ~exp(-x/delta), O(1) near wall
    v = amp * np.sin(np.pi * yv)[:, None] * gpx_v[None, :]
    u = u * np.asarray(base.u_mask.data)
    v = v * np.asarray(base.v_mask.data)
    state = base._replace(
        u=base.u.replace(data=jnp.asarray(u[..., None] if base.u.data.ndim == 3 else u)),
        v=base.v.replace(data=jnp.asarray(v[..., None] if base.v.data.ndim == 3 else v)),
    )
    return geom, z, state


def _inviscid_unforced_config():
    return LatLonCGridOceanConfig(
        g=9.81, rho_0=1000.0,
        eos="linear", eos_linear=LinearEOSConfig(rho_ref=1000.0, alpha_T=0.0, beta_S=0.0),
        A_h=0.0, A_h_lat_scaling=False, B_h=0.0, C_smag=0.0,
        bottom_drag_r=0.0, gm_redi=None, K_h=0.0,
        momentum_advection="flux_form", momentum_flux_scheme="centered",
        lateral_side_bc="free_slip",
        barotropic_solver="implicit_cn",
        barotropic_implicit_theta_eta=1.0, barotropic_implicit_theta_pgf=1.0,
        use_conservation_fixer=False, enable_runtime_checks=False,
    )


def _ke(s):
    return float(0.5 * (np.sum(np.asarray(s.u.data) ** 2) + np.sum(np.asarray(s.v.data) ** 2)))


@pytest.mark.xfail(
    reason="known root cause: the Coriolis<->implicit-free-surface-projection "
    "coupling is not energy-conserving (spurious dt-independent KE injection; "
    "advection-independent; Coriolis neutral and free surface alone conserves, "
    "but explicit-Coriolis-then-project injects when f!=0). Pending a dycore fix "
    "to the C-grid implicit free-surface step; see "
    "docs/ocean_fidelity/mitgcm_gyre_energy_conservation.md",
    strict=False,
)
def test_inviscid_unforced_barotropic_conserves_energy():
    geom, z, s0 = _closed_box_geom_state()
    model = LatLonCGridOceanModel(geom, z, _inviscid_unforced_config())
    step = jax.jit(lambda st: model.step(st, 1200.0, surface_forcing=None))
    s = s0
    ke0 = _ke(s)
    for _ in range(1000):
        s = step(s)
    jax.block_until_ready(s.u.data)
    ratio = _ke(s) / ke0
    # Inviscid + unforced => KE must be conserved to a few percent over 1000 steps.
    assert 0.97 < ratio < 1.03, f"KE not conserved: ratio={ratio:.4f}"


def test_energy_defect_is_present_and_bounded():
    """Non-xfail companion: the defect is real (KE is NOT conserved) but the run
    stays finite — locks in current behaviour so a future energy-conserving
    free-surface scheme is detectable. The sign is state-dependent (a sharp
    western-boundary jet numerically LOSES energy here; the equilibrated MITgcm
    WBC GAINS it) — the hallmark of a gradient-sensitive, non-energy-orthogonal
    free-surface projection."""
    geom, z, s0 = _closed_box_geom_state()
    model = LatLonCGridOceanModel(geom, z, _inviscid_unforced_config())
    step = jax.jit(lambda st: model.step(st, 1200.0, surface_forcing=None))
    s = s0
    ke0 = _ke(s)
    for _ in range(1000):
        s = step(s)
    jax.block_until_ready(s.u.data)
    ratio = _ke(s) / ke0
    assert np.isfinite(ratio)
    # NOT energy-conserving (the defect), in either direction.
    assert not (0.97 < ratio < 1.03), f"unexpectedly conserved: ratio={ratio:.4f}"
