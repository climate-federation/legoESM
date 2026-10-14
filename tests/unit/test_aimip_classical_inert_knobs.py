"""AIMIP classical trainer: only the RESOLVED schemes' knobs train, and the
first-step no-inert gate is on (review F5).

Unselected families' knobs are frozen as plain floats: same value, same
checkpoint leaf, but invisible to ``eqx.filter_value_and_grad`` and the
optimizer.  The gate judges what remains.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.training.aimip_params import (  # noqa: E402
    AIMIP_CLASSICAL_CONSTRAINTS,
    AIMIPClassicalParams,
    AIMIPTrainableBundle,
    aimip_classical_trainable_names,
)

DEFAULT_ARM = dict(convection="tiedtke", turbulence="louis", gwd="mcfarlane",
                   microphysics="sundqvist", cloud="xu_randall",
                   radiation="rrtmgp", surface_bulk_scheme="constant")


def test_default_arm_excludes_unselected_families_and_dead_surface_knobs():
    names = aimip_classical_trainable_names(**DEFAULT_ARM)
    for dead in ("sbm_tau_c", "sbm_RH_ref", "surface_z0",
                 "surface_most_unstable_gamma", "surface_most_stable_beta",
                 "surface_z0h_z0_ratio"):
        assert dead not in names
    for live in ("tiedtke_epsilon_deep", "louis_Ri_crit", "mcfarlane_G_0",
                 "sundqvist_auto_rate", "cloud_rh_crit", "rrtmgp_sfc_albedo",
                 "surface_Cd_neutral", "surface_Ch_neutral"):
        assert live in names
    most = aimip_classical_trainable_names(
        **{**DEFAULT_ARM, "surface_bulk_scheme": "most", "convection": "sbm"})
    assert {"surface_z0", "surface_most_stable_beta", "sbm_tau_c"} <= most
    assert "surface_Cd_neutral" not in most and "tiedtke_delta_deep" not in most
    with pytest.raises(ValueError, match="bulk scheme"):
        aimip_classical_trainable_names(
            **{**DEFAULT_ARM, "surface_bulk_scheme": "constnat"})


def test_gray_radiation_freezes_the_cloud_knobs():
    gray = aimip_classical_trainable_names(**{**DEFAULT_ARM, "radiation": "gray"})
    assert "cloud_rh_crit" not in gray and "rrtmgp_sfc_albedo" not in gray


@pytest.mark.parametrize("bulk", ["constant", "most", "coare3", "large_yeager",
                                  "large_yeager_cesm", "nemo_si3_constant"])
def test_surface_knob_map_matches_measured_gradients(bulk):
    """The bulk-scheme -> surface-knob map is MEASURED: a knob is listed iff
    the surface fluxes carry a non-zero gradient to it (unstable, stable and
    near-neutral columns, so the stability branches are all exercised)."""
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        compute_surface_fluxes,
    )
    from legoesm.training import aimip_params as ap
    surf = sorted(c.name for c in AIMIP_CLASSICAL_CONSTRAINTS
                  if c.name.startswith("surface_"))
    p = AIMIPClassicalParams.from_defaults()
    u, v = jnp.asarray([5.0, 8.0, 2.0, 10.0]), jnp.asarray([1.0, -2.0, 0.5, 0.0])
    T = jnp.asarray([290.0, 300.0, 285.0, 296.0])
    T_sfc = jnp.asarray([295.0, 295.0, 290.0, 296.2])
    q = jnp.asarray([0.010, 0.012, 0.006, 0.014])
    q_sfc = jnp.asarray([0.016, 0.017, 0.012, 0.018])
    rho = jnp.full((4,), 1.2)
    w = jax.random.normal(jax.random.PRNGKey(0), (5, 4))

    def f(raw):
        cfg = eqx.tree_at(lambda m: m.raw_values, p, raw).to_surface_config(
            bulk_scheme=bulk)
        out = compute_surface_fluxes(u, v, T, q, T_sfc, q_sfc, rho, cfg)
        return sum(jnp.sum(wi * o) for wi, o in zip(w, out))

    g = jax.grad(f)(p.raw_values)
    live = {k for k in surf if float(jnp.abs(g[k])) > 0.0}
    names = aimip_classical_trainable_names(
        **{**DEFAULT_ARM, "surface_bulk_scheme": bulk})
    assert live == set(ap._SURFACE_KNOBS_BY_BULK[bulk]), live
    assert live == names & set(surf)


def test_frozen_knobs_keep_value_and_leaf_but_get_no_gradient(tmp_path):
    names = aimip_classical_trainable_names(**DEFAULT_ARM)
    full = AIMIPClassicalParams.from_defaults()
    part = AIMIPClassicalParams.from_defaults(trainable=names)
    assert set(full.raw_values) == set(part.raw_values)
    for k, v in part.raw_values.items():
        assert isinstance(v, float) == (k not in names)
        assert float(v) == float(full.raw_values[k])
    # the gradient reaches only the trainable knobs
    g = eqx.filter_grad(lambda m: sum(m.as_dict().values()))(part)
    assert g.raw_values["sbm_tau_c"] is None
    assert float(g.raw_values["tiedtke_epsilon_deep"]) != 0.0
    with pytest.raises(ValueError, match="unknown AIMIP"):
        AIMIPClassicalParams.from_defaults(trainable={"no_such_knob"})


def test_checkpoint_loads_into_array_and_frozen_templates(tmp_path):
    """A checkpoint with frozen knobs loads into an all-array template (every
    inference / eval / plot script builds ``from_defaults()``) at the template
    dtype, and into a frozen template as a float; fp32 knobs under x64 is the
    case where a raw float would land on disk as float64 (codex review)."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ml.training import save_checkpoint
    old = get_policy()
    set_policy(PrecisionPolicy.fp32())
    try:
        names = aimip_classical_trainable_names(**DEFAULT_ARM)
        full = AIMIPClassicalParams.from_defaults()
        part = AIMIPClassicalParams.from_defaults(trainable=names)
        part = eqx.tree_at(lambda m: m.raw_values["sbm_tau_c"], part, 0.25)
        p = tmp_path / "m.eqx"
        save_checkpoint(part, p)
        into_full = eqx.tree_deserialise_leaves(p, full)
        v = into_full.raw_values["sbm_tau_c"]
        assert v.dtype == full.raw_values["sbm_tau_c"].dtype == jnp.float32
        assert float(v) == 0.25
        into_part = eqx.tree_deserialise_leaves(
            p, AIMIPClassicalParams.from_defaults(trainable=names))
        assert into_part.raw_values["sbm_tau_c"] == 0.25
        assert isinstance(into_part.raw_values["sbm_tau_c"], float)
        # every knob frozen (no array to take the dtype from)
        save_checkpoint(AIMIPClassicalParams.from_defaults(trainable=()), p)
        assert eqx.tree_deserialise_leaves(p, full).raw_values[
            "sbm_tau_c"].dtype == jnp.float32
    finally:
        set_policy(old)


@pytest.mark.parametrize("bulk,frozen", [("constant", {"z0"}),
                                         ("most", {"Cd_neutral", "Ch_neutral"})])
def test_unread_spatial_fields_are_frozen(bulk, frozen):
    """Vote B: a spatial surface field is frozen exactly when its scalar knob
    is (same surface sub-config reads both); its gradient is cut, the live
    fields keep theirs."""
    from legoesm.grids.gaussian import create_gaussian_grid
    names = aimip_classical_trainable_names(
        **{**DEFAULT_ARM, "surface_bulk_scheme": bulk})
    p = AIMIPClassicalParams.from_defaults(spatial_surface=True,
                                           spatial_init_std=0.01,
                                           trainable=names)
    assert p.spatial_surface.frozen == frozen
    assert AIMIPClassicalParams.from_defaults(
        spatial_surface=True).spatial_surface.frozen == frozenset()
    grid = create_gaussian_grid(10, dealiasing="quadratic")
    sp = p.spatial_surface

    def f(s):
        return sum(jnp.sum(v) for v in s.evaluate(grid).values())
    g = eqx.filter_grad(f)(sp)
    for n, fld in g.fields.items():
        assert (float(jnp.abs(fld.coeffs).max()) == 0.0) == (n in frozen), n


def test_frozen_spatial_fields_keep_their_values_under_adamw():
    """The gradient cut alone would let AdamW's decoupled weight decay move a
    frozen field; the trainer's optimizer zeroes its updates (reviews r3)."""
    import optax

    from legoesm.training.neural_gcm_spectral import freeze_spatial_updates
    names = aimip_classical_trainable_names(**DEFAULT_ARM)   # constant: z0 frozen
    p = AIMIPClassicalParams.from_defaults(spatial_surface=True,
                                           spatial_init_std=0.01, trainable=names)
    opt = freeze_spatial_updates(optax.adamw(1e-2, weight_decay=0.5), p)
    params = eqx.filter(p, eqx.is_array)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    upd, _ = opt.update(grads, opt.init(params), params)
    f = upd.spatial_surface.fields
    assert float(jnp.abs(f["z0"].coeffs).max()) == 0.0
    assert float(jnp.abs(f["Cd_neutral"].coeffs).max()) > 0.0
    base = optax.adamw(1e-2)
    assert freeze_spatial_updates(
        base, AIMIPClassicalParams.from_defaults(spatial_surface=True)) is base


def test_gate_reads_bare_params_and_bundles():
    from legoesm.training.neural_gcm_spectral import _trainable_raw_values
    bare = types.SimpleNamespace(raw_values={"a": jnp.ones(()), "f": None})
    assert set(_trainable_raw_values(bare)) == {"a"}
    sp = types.SimpleNamespace(
        fields={n: types.SimpleNamespace(coeffs=jnp.ones(3)) for n in ("z0", "Cd")},
        frozen=frozenset({"z0"}))
    with_sp = types.SimpleNamespace(raw_values={"a": jnp.ones(())},
                                    spatial_surface=sp)
    assert set(_trainable_raw_values(with_sp)) == {"a", "spatial.Cd"}
    bundle = AIMIPTrainableBundle(
        classical=types.SimpleNamespace(raw_values={"a": jnp.ones(())}),
        schemes=types.SimpleNamespace(raw_values={"b": jnp.zeros(())}))
    assert set(_trainable_raw_values(bundle)) == {"classical.a", "schemes.b"}


def _toy_gate_run(params, tmp_path):
    """The shared loop with the gate ON, one epoch, a physics that reads only
    tiedtke_epsilon_deep (mirrors test_neural_gcm_spectral's gate harness)."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.losses import LossConfig
    from legoesm.training.neural_gcm_spectral import (
        NeuralGCMSpectralConfig, _train_spectral_loop, carry_to_spectral_state,
    )
    sys_path = str(Path(__file__).resolve().parent)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from test_neural_gcm_spectral import _make_gaussian_carry
    grid = create_gaussian_grid(10, dealiasing="quadratic")
    sigma = create_sigma_coordinate(3, sigma_top=0.1)
    ic = carry_to_spectral_state(_make_gaussian_carry(T_val=280.0), grid)

    def mk(p, grid_):
        d = p.as_dict()

        def physics_fn(state, g, s, forcing=None):
            tend = jax.tree.map(jnp.zeros_like, state)
            amp = 1.0 * d["tiedtke_epsilon_deep"]
            return tend._replace(T_hat=jax.tree.map(
                lambda x: x.at[0].add(amp), tend.T_hat))
        return physics_fn

    cfg = NeuralGCMSpectralConfig(
        n_max=10, n_levels=3, dt=1800.0,
        pe_config=SpectralPEConfig(time_integrator="ssp_rk3"),
        n_epochs=1, lr=1e-4, warmup_steps=0,
        loss_config=LossConfig(multi_step_hours=(1,), multi_step_weights=(1.0,)),
        log_every=1, checkpoint_dir=str(tmp_path))
    return _train_spectral_loop(
        params, mk, grid, sigma, [ic],
        [(_make_gaussian_carry(T_val=282.0),)], cfg,
        assert_no_inert_params=True)


def test_gate_passes_with_frozen_families_and_aborts_without(tmp_path):
    # every knob trainable: all but one are inert -> the gate aborts
    with pytest.raises(ValueError, match="identically-zero"):
        _toy_gate_run(AIMIPClassicalParams.from_defaults(), tmp_path / "a")
    # only the read knob trainable: passes, and the frozen ones are untouched
    p0 = AIMIPClassicalParams.from_defaults(trainable={"tiedtke_epsilon_deep"})
    model, _ = _toy_gate_run(p0, tmp_path / "b")
    assert model.raw_values["sbm_tau_c"] == p0.raw_values["sbm_tau_c"]


def test_run_aimip_classical_trains_resolved_knobs_with_gate_on(monkeypatch):
    import legoesm.training.aimip_params as ap
    import legoesm.training.neural_gcm_spectral as ngs
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "run_aimip_inert_probe", root / "scripts" / "run" / "run_aimip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod._merge(mod._load_yaml(root / "config" / "aimip" / "aimip_era5.yaml"), {
        "n_max": 8, "aimip_variant": "classical", "aimip_spatial_surface": False,
        "aimip_radiation": "gray", "aimip_rad_update_interval": 1})
    spec_cfg = mod._build_spectral_config(cfg)

    def fake_data(spec_cfg_, grid, sigma, cache_dir, **kw):
        carry = types.SimpleNamespace(
            phis=jnp.zeros((grid.n_lat, grid.n_lon)))
        return [carry], [carry], [0]
    seen = {}
    monkeypatch.setattr(ngs, "load_training_data", fake_data)
    monkeypatch.setattr(ngs, "_train_spectral_loop",
                        lambda params, *a, **k: seen.update(params=params, kw=k))
    monkeypatch.setattr(ap, "make_aimip_classical_spectral_physics",
                        lambda *a, **k: (lambda *a2, **k2: None))
    mod._train_aimip_classical(spec_cfg, "unused", cfg=cfg)
    assert seen["kw"].get("assert_no_inert_params") is True
    arrays = {k for k, v in seen["params"].raw_values.items()
              if isinstance(v, jax.Array)}
    ds = ap.CLASSICAL_DEFAULT_SCHEMES
    expect = aimip_classical_trainable_names(
        convection=cfg.get("aimip_convection", ds["convection"]),
        turbulence=cfg.get("aimip_turbulence", ds["turbulence"]),
        gwd=cfg.get("aimip_gwd", ds["gwd"]),
        microphysics=cfg.get("aimip_microphysics", ds["microphysics"]),
        cloud=cfg.get("aimip_cloud", ds["cloud"]), radiation="gray",
        surface_bulk_scheme=cfg.get("aimip_surface_bulk_scheme", "constant"))
    assert arrays == set(expect)
    assert len(arrays) < len(AIMIP_CLASSICAL_CONSTRAINTS)
    np.testing.assert_equal(len(seen["params"].raw_values),
                            len(AIMIP_CLASSICAL_CONSTRAINTS))
