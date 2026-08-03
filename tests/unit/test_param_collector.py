"""Tests for the spec-driven trainable-parameter collector."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training.param_collector import (
    ParamMeta,
    SPEC_MODULES,
    _resolve_shape,
    _seed_raw,
    build_registry,
    build_trainable_params,
)
from legoesm.core.param_overrides import apply_param_overrides
from legoesm.training.trainable_params import sigmoid_to_range


# --- registry --------------------------------------------------------------
def test_registry_reads_specs_with_live_defaults() -> None:
    reg = {m.qualified_name: m for m in build_registry()}
    assert "land.soil_thermal.Q_geothermal" in reg
    assert "ocean.sw_penetration.rgb_ir_fraction" in reg
    # default comes from the live NamedTuple, not the spec
    assert reg["land.soil_thermal.Q_geothermal"].default == 0.05
    assert reg["land.soil_thermal.k_dry_coeff_a"].tunable_tier == 2
    assert reg["land.soil_thermal.Q_geothermal"].tunable_tier == 1


def test_spec_modules_matches_gate_annotated_set() -> None:
    """Drift guard: SPEC_MODULES must equal the set of in-scope modules the
    ``__param_spec__`` gate sees as annotated (scope - TODO)."""
    from tests.test_param_specs import (
        PARAM_SPEC_TODO,
        _SCOPE_MODULES,
        extract_param_spec,
    )
    from tests import _ratchet_audit as ra

    def _dotted(rel: str) -> str:
        i = rel.index("legoesm/")
        return rel[i:].removesuffix(".py").replace("/", ".")

    annotated = {
        _dotted(rel)
        for rel in _SCOPE_MODULES
        if rel not in PARAM_SPEC_TODO
        and extract_param_spec((ra.repo_root() / rel).read_text()) is not None
    }
    assert set(SPEC_MODULES) == annotated, (
        f"SPEC_MODULES drift: registry={set(SPEC_MODULES)} vs annotated={annotated}"
    )


# --- tier continuum --------------------------------------------------------
def test_tier_core_selects_only_tier1() -> None:
    reg = {m.qualified_name: m for m in build_registry()}
    p = build_trainable_params(tier="core")
    names = {c.name for c in p.constraints}
    assert "land.soil_thermal.Q_geothermal" in names
    # every collected param at tier=core must be tier 1 (robust to added specs)
    assert names and all(reg[n].tunable_tier == 1 for n in names)
    # a known tier-2 param must NOT be present at core
    assert "land.soil_thermal.k_dry_coeff_a" not in names


def test_tier_extended_adds_tier2() -> None:
    core = {c.name for c in build_trainable_params(tier="core").constraints}
    ext = {c.name for c in build_trainable_params(tier="extended").constraints}
    assert core < ext
    assert "land.soil_thermal.k_dry_coeff_a" in ext
    assert "ocean.sw_penetration.rgb_ir_fraction" in ext


def test_active_scheme_keys_filter() -> None:
    p = build_trainable_params(tier="extended", active_scheme_keys={"land.soil_thermal"})
    schemes = {c.scheme_key for c in p.constraints}
    assert schemes == {"land.soil_thermal"}


def test_include_exclude() -> None:
    # include a tier-2 param while staying at core tier
    p = build_trainable_params(tier="core", include=("ocean.sw_penetration.rgb_ir_fraction",))
    names = {c.name for c in p.constraints}
    assert "ocean.sw_penetration.rgb_ir_fraction" in names
    # exclude removes it
    p2 = build_trainable_params(tier="extended", exclude=("ocean.sw_penetration.rgb_ir_fraction",))
    assert "ocean.sw_penetration.rgb_ir_fraction" not in {c.name for c in p2.constraints}


def test_unknown_include_raises() -> None:
    with pytest.raises(ValueError, match="unknown parameters"):
        build_trainable_params(include=("does.not.exist",))


def test_unknown_active_scheme_key_raises() -> None:
    # a typo/stale scheme key must raise, not silently yield an empty set
    with pytest.raises(ValueError, match="unknown scheme"):
        build_trainable_params(active_scheme_keys={"land.soil_thermalX"})


def test_uninstalled_spec_module_is_skipped(monkeypatch) -> None:
    """A federation component that is not installed (ModuleNotFoundError on a
    SPEC_MODULES import) is skipped, not fatal — ocean-only collection still
    works without legoesm-land present."""
    import legoesm.training.param_collector as pc

    monkeypatch.setattr(
        pc, "SPEC_MODULES",
        ("legoesm.ocean.physics.shortwave_penetration", "legoesm.not_installed.xyz"),
    )
    skipped: list[str] = []
    reg = pc.build_registry(skipped=skipped)
    assert skipped == ["legoesm.not_installed.xyz"]
    assert {m.scheme_key for m in reg} == {"ocean.sw_penetration"}
    # ocean-only collection succeeds despite the missing module
    p = pc.build_trainable_params(tier="extended", active_scheme_keys={"ocean.sw_penetration"})
    assert {c.scheme_key for c in p.constraints} == {"ocean.sw_penetration"}


def test_transitive_import_failure_reraises(monkeypatch) -> None:
    """A ModuleNotFoundError for a dependency that is NOT on the spec module's
    own path (a broken/renamed transitive import in an installed module) must
    fail loudly, not silently drop the scheme's parameters."""
    import legoesm.training.param_collector as pc

    monkeypatch.setattr(pc, "SPEC_MODULES", ("legoesm.ocean.physics.shortwave_penetration",))

    def boom(name):
        raise ModuleNotFoundError(
            "No module named 'legoesm.ocean.renamed_dep'",
            name="legoesm.ocean.renamed_dep",
        )

    monkeypatch.setattr(pc.importlib, "import_module", boom)
    with pytest.raises(ModuleNotFoundError, match="renamed_dep"):
        pc.build_registry()


# --- raw seeding round-trips to the default --------------------------------
def test_seeded_values_recover_defaults() -> None:
    # "aggressive" = all tiers (1..3): every specced param must seed back to its
    # default through the float32-pinned transform inverse. This is the tripwire
    # that catches (a) a softplus param whose default is too small to round-trip
    # in float32 (the inverse compresses small positives into large-negative
    # seeds below float32 resolution) and (b) a sigmoid param whose default sits
    # on a bound (logit -> +/-inf) or spans >~2 decades near a floor. Fix in the
    # spec: bounded params use sigmoid with the default strictly interior, and
    # ranges are kept narrow enough to stay float32-recoverable.
    p = build_trainable_params(tier="aggressive")
    physical = p.as_dict()
    reg = {m.qualified_name: m for m in build_registry()}
    for name, val in physical.items():
        np.testing.assert_allclose(float(val), reg[name].default, rtol=1e-5)


# --- variable-size machinery (no array scheme specced yet -> unit-level) ----
def _synthetic_pft_meta() -> ParamMeta:
    return ParamMeta(
        scheme_key="land.veg", qualified_name="land.veg.albedo", field="albedo",
        module="x", config_class="X", default=0.2, bounds=(0.05, 0.4),
        tunable_tier=2, transform="sigmoid", units="1", category="surface",
        reference="r", shape_key="n_pft", legacy_name=None,
    )


def test_variable_size_resolves_dim_and_broadcasts_bounds() -> None:
    meta = _synthetic_pft_meta()
    shape = _resolve_shape(meta, {"n_pft": 14})
    assert shape == (14,)
    raw = _seed_raw(meta, shape, jnp.float32)
    assert raw.shape == (14,) and raw.dtype == jnp.float32
    # scalar bounds broadcast over the array, recovering the scalar default
    recovered = sigmoid_to_range(raw, meta.bounds[0], meta.bounds[1])
    np.testing.assert_allclose(np.asarray(recovered), 0.2, rtol=1e-5)


def test_missing_dim_raises() -> None:
    with pytest.raises(ValueError, match="needs dimension 'n_pft'"):
        _resolve_shape(_synthetic_pft_meta(), None)


def test_seed_raw_dtype_pinned() -> None:
    raw = _seed_raw(_synthetic_pft_meta(), (3,), jnp.float32)
    assert raw.dtype == jnp.float32  # guards float64->float32 scan-carry hazard


# --- overrides round-trip + injection --------------------------------------
def test_to_overrides_groups_by_scheme() -> None:
    p = build_trainable_params(tier="extended")
    ov = p.to_overrides()
    assert "land.soil_thermal" in ov and "ocean.sw_penetration" in ov
    assert "Q_geothermal" in ov["land.soil_thermal"]


def test_apply_overrides_into_real_config() -> None:
    from legoesm.land.soil_thermal import SoilThermalConfig

    p = build_trainable_params(tier="core")  # Q_geothermal only
    ov = p.to_overrides()["land.soil_thermal"]
    cfg = apply_param_overrides(SoilThermalConfig(), ov)
    np.testing.assert_allclose(float(cfg.Q_geothermal), 0.05, rtol=1e-5)


def test_apply_overrides_unknown_field_raises() -> None:
    from legoesm.land.soil_thermal import SoilThermalConfig

    with pytest.raises(ValueError, match="no field"):
        apply_param_overrides(SoilThermalConfig(), {"not_a_field": 1.0})


def test_to_segment_kwargs_rejects_scheme_qualified() -> None:
    p = build_trainable_params(tier="core")
    with pytest.raises(ValueError, match="scheme-qualified"):
        p.to_segment_kwargs()


# --- end-to-end differentiability through the injected config ---------------
def test_grad_flows_through_collected_param_into_scheme() -> None:
    from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
    from legoesm.land.soil_thermal import (
        SoilThermalConfig,
        compute_thermal_conductivity,
    )

    hcfg = SoilHydraulicsConfig()
    theta = jnp.full((4,), 0.18)
    p = build_trainable_params(tier="extended", active_scheme_keys={"land.soil_thermal"})

    def loss(params) -> jax.Array:
        ov = params.to_overrides()["land.soil_thermal"]
        cfg = apply_param_overrides(SoilThermalConfig(), ov)
        return jnp.sum(compute_thermal_conductivity(theta, hcfg, cfg))

    import equinox as eqx

    val, grads = eqx.filter_value_and_grad(loss)(p)
    assert np.isfinite(float(val))
    leaves = [g for g in jax.tree_util.tree_leaves(grads.raw_values)]
    assert leaves and all(np.all(np.isfinite(np.asarray(g))) for g in leaves)
    # k_dry_coeff_a has a nonzero effect on conductivity -> nonzero grad
    assert any(np.any(np.asarray(g) != 0.0) for g in leaves)


def test_pytree_shapes_stable_across_two_builds() -> None:
    a = build_trainable_params(tier="extended")
    b = build_trainable_params(tier="extended")
    ta = jax.tree_util.tree_structure(a)
    tb = jax.tree_util.tree_structure(b)
    assert ta == tb


def test_rrtmgp_cache_key_fields_never_trainable() -> None:
    """RRTMGP keys its optics/solver instance cache on these Python config
    floats (rrtmgp.py _optics_cache_key / _instance_cache_key). Exposing them
    as trainable leaves would break the Python hash or rebuild the solver per
    value, so they must be ``excluded`` (tier 0) and never appear in a built
    trainable set -- not even at the broadest "aggressive" tier."""
    # All RRTMGP config floats that _optics_cache_key / _instance_cache_key
    # fold into the Python solver-cache key (gas concentrations, bulk aerosol
    # optics, and the surface albedo/emissivity routed through _hashable/float).
    cache_key_fields = {
        "co2_ppmv", "ch4_ppbv", "n2o_ppbv", "aerosol_ssa", "aerosol_g",
        "sfc_albedo", "sfc_emissivity",
    }
    p = build_trainable_params(tier="aggressive")
    rrtmgp_fields = {c.field for c in p.constraints if c.scheme_key == "atm.rad.RRTMGPConfig"}
    leaked = cache_key_fields & rrtmgp_fields
    assert not leaked, f"RRTMGP cache-key fields leaked into trainables: {leaked}"
    # also assert they are not present anywhere in the registry's param set
    reg_fields = {(m.scheme_key, m.field) for m in build_registry()}
    assert not any(("atm.rad.RRTMGPConfig", f) in reg_fields for f in cache_key_fields)


def test_as_dict_respects_declared_finite_bounds() -> None:
    """Every collected trainable's constrained value must stay within its
    declared (lo, hi) for any raw input. ``sigmoid`` enforces this; bare
    ``softplus``/``none`` do NOT (softplus is unbounded above, none is the
    identity), so a finite-bounded param MUST use a bounds-respecting transform.
    Drive raw to +/-1e4 and assert as_dict() never escapes the declared range --
    the tripwire for the false-bounds class codex flagged."""
    import equinox as eqx

    p = build_trainable_params(tier="aggressive")
    bounds = {c.name: (c.min_val, c.max_val, c.transform) for c in p.constraints}
    for sign in (1.0, -1.0):
        extreme = {k: jnp.full_like(v, sign * 1.0e4) for k, v in p.raw_values.items()}
        q = eqx.tree_at(lambda t: t.raw_values, p, extreme)
        for name, val in q.as_dict().items():
            lo, hi, transform = bounds[name]
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
                continue  # per-element / open bounds checked elsewhere
            arr = np.asarray(val)
            # tolerance scaled to the bound magnitude: float32 cannot represent
            # an O(10-100) bound exactly, so saturated sigmoid output rounds a
            # few ULP past it. This still catches the gross escapes (softplus ->
            # +inf, identity "none" -> raw) the tripwire targets.
            tol = 1e-4 * max(1.0, abs(lo), abs(hi))
            assert np.all(arr >= lo - tol) and np.all(arr <= hi + tol), (
                f"{name} ({transform}) -> {arr.reshape(-1)[:3]} escaped declared "
                f"bounds [{lo}, {hi}] at raw={sign:+}e4"
            )
