"""Anisotropic Minimum-Dissipation (AMD) sub-grid eddy viscosity.

The defining AMD properties vs Vreman/Smagorinsky: ZERO viscosity where the
resolved flow needs no SGS dissipation (well-resolved laminar / 1-D shear), and
NEVER negative (no spurious backscatter) — the minimum-dissipation guarantee.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.turbulence.amd import amd_nu_t

_DX = _DY = _DZ = 10.0
_C = 0.3
_G = ("a11", "a12", "a13", "a21", "a22", "a23", "a31", "a32", "a33")


def _nu(**grads):
    a = {k: jnp.asarray(grads.get(k, 0.0)) for k in _G}
    return amd_nu_t(a["a11"], a["a12"], a["a13"], a["a21"], a["a22"], a["a23"],
                    a["a31"], a["a32"], a["a33"], _DX, _DY, _DZ, _C)


def test_offdiagonal_one_d_shear_gives_zero():
    """A pure off-diagonal 1-D shear (du/dz or dv/dz only) needs no SGS
    dissipation: AMD gives exactly zero (the minimum-dissipation property
    Smagorinsky lacks)."""
    assert float(_nu(a13=0.5)) == pytest.approx(0.0, abs=1e-20)
    assert float(_nu(a23=0.3)) == pytest.approx(0.0, abs=1e-20)
    assert float(_nu(a31=-0.7)) == pytest.approx(0.0, abs=1e-20)


def test_normal_strain_sign_dependence():
    """A single NORMAL strain is sign-dependent: expanding (positive a11) needs
    no dissipation (nu_t=0) while compressing (negative a11) draws positive
    viscosity — the intended minimum-dissipation behaviour, NOT a blanket
    single-gradient cutoff."""
    assert float(_nu(a11=0.4)) == pytest.approx(0.0, abs=1e-20)
    assert float(_nu(a11=-0.4)) > 0.0


def test_nonnegative_never_backscatter():
    """nu_t >= 0 for any gradient field, incl. ones that would give N<0 (no
    negative viscosity)."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        g = {k: float(rng.normal()) for k in _G}
        assert float(_nu(**g)) >= 0.0


def test_three_d_field_is_positive():
    """A genuinely 3-D straining field draws positive eddy viscosity."""
    nu = float(_nu(a11=0.2, a12=0.5, a13=-0.3, a21=-0.4, a22=0.1,
                   a23=0.6, a31=0.3, a32=-0.2, a33=-0.3))
    assert nu > 0.0


def test_rest_state_zero_and_finite_grad():
    """All-zero gradients -> nu_t = 0, and the gradient stays finite (floored
    denominator, no sqrt(0) trap)."""
    assert float(_nu()) == pytest.approx(0.0, abs=1e-20)
    g = jax.grad(lambda x: amd_nu_t(x, x, x, x, x, x, x, x, x,
                                    _DX, _DY, _DZ, _C))(jnp.asarray(0.0))
    assert jnp.isfinite(g)


def test_differentiable_general():
    def loss(s):
        return amd_nu_t(0.2 * s, 0.5, -0.3, -0.4, 0.1, 0.6, 0.3, -0.2, -0.3,
                        _DX, _DY, _DZ, _C)
    assert jnp.isfinite(jax.grad(loss)(jnp.asarray(1.0)))


def test_floor_is_additive():
    base = float(_nu(a11=0.2, a12=0.5, a13=-0.3, a21=-0.4, a22=0.1,
                     a23=0.6, a31=0.3, a32=-0.2, a33=-0.3))
    withfloor = float(amd_nu_t(0.2, 0.5, -0.3, -0.4, 0.1, 0.6, 0.3, -0.2, -0.3,
                               _DX, _DY, _DZ, _C, nu_floor=1.5))
    assert withfloor == pytest.approx(base + 1.5, rel=1e-6)


def test_anisotropy_direction_weights():
    """Per-direction filter widths matter: shrinking dz down-weights the
    vertical-gradient contribution to N (a known-positive 3-D field)."""
    g = dict(a11=0.2, a12=0.5, a13=-0.3, a21=-0.4, a22=0.1,
             a23=0.6, a31=0.3, a32=-0.2, a33=-0.3)
    a = {k: jnp.asarray(v) for k, v in g.items()}
    nu_iso = float(amd_nu_t(*[a[k] for k in _G], 10.0, 10.0, 10.0, _C))
    nu_thin = float(amd_nu_t(*[a[k] for k in _G], 10.0, 10.0, 1.0, _C))
    assert nu_iso > 0.0
    assert not np.isclose(nu_iso, nu_thin)


def test_plane_config_validation_accepts_amd():
    """The plane dycore validation accepts turbulence_closure='amd' and rejects
    a non-positive amd_c (dispatch hardening)."""
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig)
    from legoesm.atmosphere.dynamics.les import compressible_euler_plane as cep

    cep.validate_plane_config(
        CompressibleEulerConfig(turbulence_closure="amd", amd_c=0.3))
    with pytest.raises(ValueError, match="amd_c"):
        cep.validate_plane_config(
            CompressibleEulerConfig(turbulence_closure="amd", amd_c=0.0))


def test_plane_config_validation_rejects_amd_incompatibilities():
    """AMD is a static, self-contained closure with a K_h = K_m/Pr split, so the
    validation rejects a dynamic-Smagorinsky coefficient and a non-positive
    Prandtl number (dispatch hardening on the nested closure config)."""
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig)
    from legoesm.atmosphere.dynamics.les import compressible_euler_plane as cep

    with pytest.raises(ValueError, match="smagorinsky_dynamic"):
        cep.validate_plane_config(CompressibleEulerConfig(
            turbulence_closure="amd", amd_c=0.3, smagorinsky_dynamic=True))
    with pytest.raises(ValueError, match="smagorinsky_prandtl"):
        cep.validate_plane_config(CompressibleEulerConfig(
            turbulence_closure="amd", amd_c=0.3, smagorinsky_prandtl=0.0))


def test_amd_falls_back_on_halo_fast_path_not_silently_inviscid():
    """Regression: the JIT-split halo fast path (`slow_tendency_jit_split`) must
    NOT silently run inviscid for an 'amd' config.

    amd_c and smagorinsky_cs both default to 0.0, so unless the fast-path guard
    lists the closure by NAME an 'amd' config trips none of its
    (smag_cs>0 / tracers / coriolis / non-upwind1 / w_hyperdiff) fall-back
    conditions and skips the SGS block entirely (inviscid).  With the guard
    fixed, an 'amd' config falls back to the eager halo kernel, which then hits
    the explicit single-rank NotImplementedError guard — proving AMD is routed,
    not dropped.  (Reachable in production via scripts/bench/bench_dd_scaling.py.)
    """
    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig)
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        make_flat_plane_terrain_metric, make_rest_state)
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane_halo import (
        slow_tendency_jit_split)
    from legoesm.grids.plane import create_plane_grid
    from legoesm.grids.vertical import create_stretched_height_coordinate
    from legoesm.parallel.plane_mpi import make_plane_pencil_layout

    grid = create_plane_grid(nx=8, ny=8, nlev=6, dx=1000.0, dy=1000.0,
                             dtype=jnp.float64)
    hc = create_stretched_height_coordinate(6, H=20_000.0, dz_sfc=100.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    layout = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1, ny_global=8, nx_global=8)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    # AMD is the SOLE fall-back trigger: 0 tracers (make_rest_state default),
    # smag_cs=0, w_hyperdiff=0, upwind1 (all defaults); coriolis OFF.
    cfg = CompressibleEulerConfig(
        turbulence_closure="amd", amd_c=0.3, use_coriolis=False)
    with pytest.raises(NotImplementedError, match="single-rank"):
        slow_tendency_jit_split(state, grid, hc, tm, cfg, layout)
