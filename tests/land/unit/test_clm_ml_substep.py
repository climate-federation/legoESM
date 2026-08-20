"""CLM-ML canopy sub-step resolution (``num_ml_steps`` / ``dtime_ml_target_s``).

CLM-ML integrates a canopy AIR-SPACE budget whose storage term is stiff on a
timescale of minutes; upstream ``MLclm_varctl.dtime_ml`` is 300 s.  legoESM sets
``dtime_ml = dt / num_ml_steps``, so a host step much longer than the design
sub-step drives that term as a numerical artefact: measured at FLUXNET US-MMS
with a 1800 s step and ``num_ml_steps=1``, the storage term reached +157 W/m² at
midday and -100 W/m² at night, delaying the sensible-heat peak by ~4 h and
inflating nighttime latent heat to 44 W/m² against an observed ~2 W/m²
(2000-step window; 78.7 vs ~3.9 on the shorter sub-step-sweep window).
Restoring the 300 s sub-step cut the latent-heat diurnal RMSE by 58%.

These tests pin the resolution rule so the canopy cannot silently drift back to
running at the host timestep.  No ``clm-ml-jax`` needed — the rule is pure
legoESM-side arithmetic.
"""

from __future__ import annotations

import warnings

import pytest

from legoesm.land.canopy.config import CLMMLCanopyConfig
from legoesm.land.canopy.clm_ml_interface import resolve_num_ml_steps

# The sub-step CLM-ML is designed for, per upstream MLclm_varctl.dtime_ml.
DESIGN_DTIME_ML_S = 300.0


def test_default_derives_the_substep_instead_of_following_the_host_step():
    assert CLMMLCanopyConfig().num_ml_steps is None


def test_design_target_matches_upstream():
    assert CLMMLCanopyConfig().dtime_ml_target_s == DESIGN_DTIME_ML_S


@pytest.mark.parametrize("dt,expected", [
    (300.0, 1),     # already at the design sub-step
    (600.0, 2),
    (1800.0, 6),    # the common land/coupled step
    (3600.0, 12),
])
def test_derived_count_holds_the_canopy_at_its_design_substep(dt, expected):
    cfg = CLMMLCanopyConfig()
    n = resolve_num_ml_steps(cfg, dt)
    assert n == expected
    assert dt / n == pytest.approx(DESIGN_DTIME_ML_S)


def test_substep_never_coarser_than_design_over_a_dense_host_step_sweep():
    """The property that actually matters, swept densely rather than sampled.

    A hand-picked sweep of round host steps hid a real defect: deriving the
    count with round() instead of ceil() lands on a COARSER sub-step whenever
    dt/target has a fractional part <= 0.5 (dt=750 -> round(2.5)=2 -> 375 s).
    Every such dt is a multiple of 30 s that a real configuration could use, so
    the sweep is dense over the plausible range rather than illustrative.
    """
    cfg = CLMMLCanopyConfig()
    for dt_int in range(30, 7230, 30):
        dt = float(dt_int)
        effective = dt / resolve_num_ml_steps(cfg, dt)
        # Sub-steps finer than the target are fine (a short host step cannot be
        # sub-cycled below itself); coarser is the failure mode.
        assert effective <= DESIGN_DTIME_ML_S + 1e-9, (
            f"dt={dt}s resolves to a {effective}s canopy sub-step, coarser "
            f"than the {DESIGN_DTIME_ML_S}s design value")


@pytest.mark.parametrize("dt", [601.0, 750.0, 1050.0, 1350.0, 2100.0])
def test_round_to_nearest_regression_cases(dt):
    """The exact host steps a round()-based rule gets wrong.

    Pinned by name so the rule cannot quietly revert to round-to-nearest.
    """
    effective = dt / resolve_num_ml_steps(CLMMLCanopyConfig(), dt)
    assert effective <= DESIGN_DTIME_ML_S + 1e-9


def test_a_short_host_step_is_not_sub_cycled_below_itself():
    cfg = CLMMLCanopyConfig()
    assert resolve_num_ml_steps(cfg, 150.0) == 1


def test_explicit_count_is_honoured():
    cfg = CLMMLCanopyConfig(num_ml_steps=4)
    assert resolve_num_ml_steps(cfg, 1200.0) == 4


def test_explicit_count_coarser_than_design_is_refused():
    """The pre-fix default (1 substep at 1800 s) must not run silently.

    Refused rather than warned: a RuntimeWarning is routinely suppressed or
    buried in batch output, and this setting silently degrades production
    fluxes (58% of the latent-heat diurnal skill at US-MMS).
    """
    cfg = CLMMLCanopyConfig(num_ml_steps=1)
    with pytest.raises(ValueError, match="coarser than the 300 s"):
        resolve_num_ml_steps(cfg, 1800.0)


def test_coarse_substep_can_be_opted_into_for_legacy_reproduction():
    cfg = CLMMLCanopyConfig(num_ml_steps=1, allow_coarse_ml_substep=True)
    with pytest.warns(RuntimeWarning, match="coarser than the 300 s"):
        assert resolve_num_ml_steps(cfg, 1800.0) == 1


@pytest.mark.parametrize("bad", [1.9, 6.0, "6", True])
def test_non_integer_explicit_count_is_refused_not_coerced(bad):
    """int()-coercing a 1.9 or a YAML "6" would run a sub-step nobody asked for.

    ``True`` is refused too: ``num_ml_steps=True`` is a mistake, not a request
    for one sub-step.
    """
    with pytest.raises(TypeError, match="must be an integer"):
        resolve_num_ml_steps(CLMMLCanopyConfig(num_ml_steps=bad), 1800.0)


def test_numpy_integer_count_is_accepted():
    """A count assembled from numpy is still a count."""
    np = pytest.importorskip("numpy")
    cfg = CLMMLCanopyConfig(num_ml_steps=np.int64(6))
    assert resolve_num_ml_steps(cfg, 1800.0) == 6


def test_non_finite_target_cannot_defeat_the_guard():
    """An inf target would make ceil() return 1 and silently disable the check."""
    for bad in (float("inf"), float("nan"), True, "300"):
        with pytest.raises(ValueError, match="must be a finite number > 0"):
            resolve_num_ml_steps(CLMMLCanopyConfig(dtime_ml_target_s=bad), 1800.0)


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), True, "1800"])
def test_non_finite_or_non_numeric_dt_is_rejected(bad):
    """`dt=True` would be read as a 1-second step; a string must not coerce."""
    with pytest.raises(ValueError, match="must be a finite number > 0"):
        resolve_num_ml_steps(CLMMLCanopyConfig(), bad)


def test_floating_point_dt_just_above_a_multiple_stays_within_target():
    """ceil() may add one conservative sub-step; it must never exceed target."""
    dt = 900.0 + 1e-9
    n = resolve_num_ml_steps(CLMMLCanopyConfig(), dt)
    assert dt / n <= DESIGN_DTIME_ML_S + 1e-9


def test_explicit_count_at_or_below_design_is_silent():
    cfg = CLMMLCanopyConfig(num_ml_steps=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_num_ml_steps(cfg, 1800.0) == 6


@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_explicit_count_is_rejected(bad):
    with pytest.raises(ValueError, match="num_ml_steps must be >= 1"):
        resolve_num_ml_steps(CLMMLCanopyConfig(num_ml_steps=bad), 1800.0)


def test_nonpositive_target_is_rejected():
    with pytest.raises(ValueError, match="must be a finite number > 0"):
        resolve_num_ml_steps(CLMMLCanopyConfig(dtime_ml_target_s=0.0), 1800.0)


@pytest.mark.parametrize("dt", [0.0, -600.0])
def test_nonpositive_dt_is_rejected(dt):
    with pytest.raises(ValueError, match="must be a finite number > 0"):
        resolve_num_ml_steps(CLMMLCanopyConfig(), dt)


def test_denormal_target_is_rejected_not_an_overflow():
    """A ratio that overflows must raise the documented error, not OverflowError."""
    with pytest.raises(ValueError):
        resolve_num_ml_steps(CLMMLCanopyConfig(dtime_ml_target_s=5e-324), 1800.0)


@pytest.mark.parametrize("truthy", [1, "false", "yes"])
def test_coarse_opt_in_requires_a_real_bool(truthy):
    """A stray truthy value must not enable a known-degraded configuration."""
    cfg = CLMMLCanopyConfig(num_ml_steps=1, allow_coarse_ml_substep=truthy)
    with pytest.raises(ValueError, match="coarser than the 300 s"):
        resolve_num_ml_steps(cfg, 1800.0)


def test_resolution_runs_at_flux_entry_before_clm_state_is_touched():
    """Tripwire: an invalid sub-step must abort before CLM globals are mutated.

    ``compute_clm_ml_canopy_fluxes`` initialises CLM and calls
    ``_setup_clm_time`` partway through; resolving the sub-step after that would
    leave process-global CLM module state half-updated on a config error.
    """
    import ast
    import inspect
    from legoesm.land.canopy import clm_ml_interface as iface

    tree = ast.parse(inspect.getsource(iface.compute_clm_ml_canopy_fluxes))
    line = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            line.setdefault(node.func.id, []).append(node.lineno)
    assert "resolve_num_ml_steps" in line, (
        "the flux entry no longer resolves the canopy sub-step; "
        "CLMMLCanopyConfig.num_ml_steps would be ignored")
    for later in ("_ensure_clm_initialized", "_setup_clm_time"):
        assert min(line["resolve_num_ml_steps"]) < min(line[later]), (
            f"sub-step resolution runs after {later}, so an invalid config "
            "would abort with CLM module state already mutated")


@pytest.mark.parametrize("kwargs,dt", [
    ({}, None),
    ({"dtime_ml_target_s": None}, 1800.0),
])
def test_none_is_rejected_with_the_documented_error(kwargs, dt):
    """The shared validator treats None as "not set"; here both are REQUIRED.

    Without an explicit check this falls through to float(None) and raises
    TypeError rather than the documented ValueError.
    """
    with pytest.raises(ValueError, match="must be a finite number > 0"):
        resolve_num_ml_steps(CLMMLCanopyConfig(**kwargs), dt)


def test_the_resolved_count_is_the_one_that_sets_dtime_ml():
    """Ordering alone is not enough — the resolved value must be the value used.

    Guards against the resolution being called for validation only while
    ``dtime_ml`` is still computed from something else (e.g. a re-read of
    ``canopy_config.num_ml_steps``, which is None by default and would divide
    by None or silently fall back).
    """
    import ast
    import inspect
    from legoesm.land.canopy import clm_ml_interface as iface

    src = inspect.getsource(iface.compute_clm_ml_canopy_fluxes)
    tree = ast.parse(src)

    # The local that receives resolve_num_ml_steps(...).
    resolved = {
        t.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "resolve_num_ml_steps"
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert resolved, "resolve_num_ml_steps result is not bound to a local"

    # ... must be what dtime_ml is computed from.
    assigns = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "dtime_ml"
                for t in node.targets)
    ]
    assert len(assigns) == 1, "expected exactly one dtime_ml assignment"
    names = {n.id for n in ast.walk(assigns[0].value) if isinstance(n, ast.Name)}
    assert names & resolved, (
        "dtime_ml is not computed from the resolved sub-step count; the "
        "validated value is being discarded")
