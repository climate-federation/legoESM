"""TKEConfig.tke_langmuir_rhs_coupling: NEMO adds the Langmuir increment to en
BEFORE the matrix (zdftke.F90:367), so the explicit dissipation add-back
0.5*rn_ediss*dissl*en (:419) also acts on it.  'nemo_pre_solve' reproduces
that on the factored matrix; the historical 'separate' does not.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.physics.vertical_mixing.config import TKEConfig  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.tke import (  # noqa: E402
    _solve_tke_backward_euler, tke_integrate_post_mixing,
)

_SHAPE = (1, 1, 6)


def _solve(cfg, **kw):
    e = jnp.linspace(2e-4, 5e-5, 6).reshape(_SHAPE)
    return np.asarray(_solve_tke_backward_euler(
        e_old=e, K_M_old=jnp.full(_SHAPE, 1e-3), K_H_old=jnp.full(_SHAPE, 5e-4),
        P_s=jnp.full(_SHAPE, 1e-7), N2=jnp.full(_SHAPE, 1e-5),
        l_eps=jnp.linspace(3.0, 1.0, 6).reshape(_SHAPE),
        dz_half=jnp.full(_SHAPE, 2.0), surface_flux=jnp.zeros((1, 1)),
        dt=150.0, cfg=cfg, **kw))


def test_pre_solve_source_adds_exactly_the_dissipation_coupled_term():
    cfg = TKEConfig(dissipation_discretization="nemo_1p5_split")
    L = jnp.linspace(2e-7, 0.0, 6).reshape(_SHAPE)
    e = jnp.linspace(2e-4, 5e-5, 6).reshape(_SHAPE)
    l_eps = jnp.linspace(3.0, 1.0, 6).reshape(_SHAPE)
    diss_rate = cfg.c_eps * jnp.sqrt(e) / l_eps
    separate = _solve(cfg, external_source=L)
    coupled = _solve(cfg, pre_solve_source=L)
    # The solve is linear in its RHS: the only change is dt*0.5*diss_rate*dt*L.
    expected = _solve(cfg, external_source=L + 0.5 * diss_rate * 150.0 * L)
    np.testing.assert_allclose(coupled, expected, rtol=1e-12, atol=0.0)
    assert np.all(coupled[..., :5] > separate[..., :5])   # synthetic-violation guard


def test_pre_solve_source_refused_outside_its_assembly():
    L = jnp.full(_SHAPE, 1e-7)
    with pytest.raises(ValueError, match="pre_solve_source"):
        _solve(TKEConfig(dissipation_discretization="backward_euler"),
               pre_solve_source=L)


@pytest.mark.parametrize("bad", [
    dict(lc=False),
    dict(lc=True, tke_langmuir_evaluation="nemo_literal"),
    dict(lc=True, dissipation_discretization="backward_euler"),
])
def test_config_combinations_that_would_be_inert_or_double_raise(bad):
    from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing
    base = dict(dissipation_discretization="nemo_1p5_split",
                tke_langmuir_rhs_coupling="nemo_pre_solve")
    cfg = TKEConfig(**{**base, **bad})
    z = jnp.zeros((1, 1, 4))
    with pytest.raises(ValueError, match="tke_langmuir_rhs_coupling"):
        tke_vertical_mixing(z, z, z + 20.0, z + 35.0, z + 1025.0,
                            jnp.ones((1, 1, 3)), tke_old=jnp.full((1, 1, 3), 1e-6),
                            tau_x_surface=None, tau_y_surface=None, dt=150.0, cfg=cfg)


def test_unknown_coupling_raises():
    from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing
    z = jnp.zeros((1, 1, 4))
    with pytest.raises(ValueError, match="Unknown TKEConfig.tke_langmuir_rhs_coupling"):
        tke_vertical_mixing(z, z, z + 20.0, z + 35.0, z + 1025.0,
                            jnp.ones((1, 1, 3)), tke_old=jnp.full((1, 1, 3), 1e-6),
                            tau_x_surface=None, tau_y_surface=None, dt=150.0,
                            cfg=TKEConfig(tke_langmuir_rhs_coupling="nemo"))


def test_post_mixing_order_refuses_the_coupling():
    with pytest.raises(ValueError, match="post-mixing"):
        tke_integrate_post_mixing(None, None, None, None, 150.0,
                                  TKEConfig(tke_langmuir_rhs_coupling="nemo_pre_solve"))


def test_step_entry_shear_refuses_gradient_richardson_prandtl():
    from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing
    z = jnp.zeros((1, 1, 4))
    cfg = TKEConfig(tke_shear_evaluation_stage="step_entry", prandtl_mode="richardson")
    with pytest.raises(ValueError, match="prandtl_mode='richardson'"):
        tke_vertical_mixing(z, z, z + 20.0, z + 35.0, z + 1025.0,
                            jnp.ones((1, 1, 3)), tke_old=jnp.full((1, 1, 3), 1e-6),
                            tau_x_surface=None, tau_y_surface=None, dt=150.0, cfg=cfg,
                            precomputed_p_sh2=jnp.zeros((1, 1, 3)))


def test_cli_flag_reaches_the_card_and_is_tripole_only():
    from scripts.run import run_omip_core2 as r
    a = r._build_arg_parser().parse_args(
        ["--grid", "tripole", "--tke-langmuir-rhs-coupling", "nemo_pre_solve"])
    assert a.tke_langmuir_rhs_coupling == "nemo_pre_solve"
    assert r._build_arg_parser().parse_args(["--grid", "tripole"]).tke_langmuir_rhs_coupling is None
    cfg = r.orca1_zdftke_config(langmuir_rhs_coupling="nemo_pre_solve")
    assert cfg.tke_langmuir_rhs_coupling == "nemo_pre_solve"
    assert r.orca1_zdftke_config().tke_langmuir_rhs_coupling == "separate"
    vm = r.build_tripole_vmix_config("tke", tke_langmuir_rhs_coupling="nemo_pre_solve")
    assert vm.tke.tke_langmuir_rhs_coupling == "nemo_pre_solve"
    with pytest.raises(ValueError, match="requires --tripole-vmix tke"):
        r.build_tripole_vmix_config("kpp", tke_langmuir_rhs_coupling="nemo_pre_solve")
    with pytest.raises(SystemExit, match="tke-langmuir-rhs-coupling"):
        r._validate_tke_card_grid("mpas", tripole_vmix="none",
                                  tke_langmuir_rhs_coupling="nemo_pre_solve",
                                  mpas_vmix="tke")


def test_public_path_coupling_raises_forced_tke_above_floor():
    """tke_vertical_mixing with a forced, stably stratified column: the
    coupling must raise e inside the Langmuir layer and leave the rest."""
    from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing
    nz = 8
    zc = -(jnp.arange(nz) + 0.5) * 2.0                       # cell centres, 2 m cells
    T = (25.0 - 0.05 * jnp.arange(nz)).reshape(1, 1, nz)
    S = jnp.full((1, 1, nz), 35.0)
    rho = (1023.0 + 0.01 * jnp.arange(nz)).reshape(1, 1, nz)
    dz_half = jnp.full((1, 1, nz - 1), 2.0)
    z_int = 0.5 * (zc[1:] + zc[:-1])
    e0 = jnp.full((1, 1, nz - 1), 3e-4)
    base = TKEConfig(lc=True, dissipation_discretization="nemo_1p5_split")

    def run(coupling):
        out = tke_vertical_mixing(
            jnp.zeros((1, 1, nz)), jnp.zeros((1, 1, nz)), T, S, rho, dz_half,
            tke_old=e0, tau_x_surface=jnp.full((1, 1), 0.1), tau_y_surface=None,
            dt=150.0, cfg=base._replace(tke_langmuir_rhs_coupling=coupling),
            z_interface=z_int, n_iterations=1)
        return np.asarray(out.tke_new)[0, 0]

    sep, cpl = run("separate"), run("nemo_pre_solve")
    assert np.all(np.isfinite(cpl))
    assert np.max(cpl - sep) > 1e-3 * np.max(sep)            # the dispatch is live
    assert np.all(cpl >= sep - 1e-15)                        # only adds energy
