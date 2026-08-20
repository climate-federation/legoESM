"""Direct unit test for the #1226 online box heat-budget accumulator.

Covers the instrument's own validation gates:
  1. Direct-vertmix closure — with VERTMIX now DIRECTLY measured (the
     realized implicit-mixing increment, not a residual), the accounting
     identity ``dH == sum(5 terms) + residual`` still holds (by
     construction of how ``residual`` is now computed), AND the residual
     is SMALL relative to the band's other terms on a short synthetic run
     — i.e. all 5 terms are genuinely measuring the right thing, not just
     an arithmetic tautology.
  2. Non-invasive: running the accumulator's ``.sample()`` alongside a short
     model integration must not perturb the model trajectory at all —
     the prognostic state with the accumulator ON must be bit-identical to
     the state with it OFF.
  3. Time-series keys are present and consistent with the cumulative
     per-term totals the summary reports.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    DINOConfig,
    apply_dino_lat_lon_surface_forcing,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)
from legoesm.ocean.fidelity.box_heat_budget import (
    BoxHeatBudgetAccumulator,
    TERM_NAMES,
    compute_box_heat_dT_terms,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star

DT = 900.0  # s
N_LAT, N_LON, N_LEV = 24, 16, 8
ROW_SLICE = slice(4, 20)
BANDS = ((0.0, 200.0), (200.0, 2000.0), (2000.0, None))


@pytest.fixture
def grid():
    return create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(n_levels=N_LEV, H_max=3000.0)


@pytest.fixture
def config():
    gm = GMRediConfig(kappa_GM=50.0, kappa_Redi=50.0, slope_scheme="nemo_iso_lap")
    return LatLonCGridOceanConfig.from_flat(
        A_h=1.0e3, A_v=1.0e-3, K_h=0.0, K_v=1.0e-4,
        bottom_drag_r=1.0e-3, gm_redi=gm,
        implicit_vertical_mixing=True,
        outer_integrator="forward_euler",
    )


@pytest.fixture
def dino_cfg():
    # wind_through_step=True: this test always passes wind via
    # model.step(surface_forcing=sf) (the dynamics-core external-tau
    # block), so apply_dino_lat_lon_surface_forcing must SKIP its own
    # top-layer wind tendency (dino.py's wind_through_step gate) to avoid
    # double-applying the wind stress.
    return DINOConfig(forcing_annual_cycle=False, wind_through_step=True)


@pytest.fixture
def state(grid, z_coord):
    rng = np.random.default_rng(0)
    s = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=4.0, S_uniform=35.0,
        H_max=3000.0,
    )
    u = 0.02 * rng.standard_normal((grid.n_lat, grid.n_lon + 1, z_coord.n_levels))
    v = 0.02 * rng.standard_normal((grid.n_lat + 1, grid.n_lon, z_coord.n_levels))
    return s._replace(
        u=s.u.replace(data=jnp.asarray(u) * (s.u_mask.data[..., None] > 0.5)),
        v=s.v.replace(data=jnp.asarray(v) * (s.v_mask.data[..., None] > 0.5)),
    )


def _run(state, model, forcing, sf, dino_cfg, *, n_steps, sample_every,
         accumulate: bool):
    """Advance ``n_steps`` of ``DT``, optionally sampling the accumulator
    every ``sample_every`` steps (including step 0, before any stepping).

    Applies the analytic post-step heat/salt/SW surface forcing
    (``apply_dino_lat_lon_surface_forcing``) each step, matching the real
    production run loop (``box_budget_twin90.py``/``box_budget_run.py``) —
    ``sf`` (``dino_step_surface_forcing``) carries WIND ONLY into
    ``model.step()``; without also applying the analytic forcing here, the
    FORCING term ``compute_box_heat_dT_terms`` computes would count heat
    the state never actually received, which the #1226 direct-VERTMIX
    closure diagnostic (RESIDUAL) would then correctly flag as an
    instrument/harness gap.
    """
    acc = None
    if accumulate:
        acc = BoxHeatBudgetAccumulator(
            model.grid, model.z_coord, model.config, dino_cfg, forcing, DT, model,
            row_slice=ROW_SLICE, depth_bands_m=BANDS,
        )

    st = state
    t = 0.0
    if acc is not None:
        acc.sample(st, dt_step=sample_every * DT, t_seconds=t)
    for k in range(n_steps):
        st = apply_dino_lat_lon_surface_forcing(
            st, forcing, model.z_coord, dino_cfg, DT, t_seconds=t + DT,
        )
        st = model.step(st, DT, surface_forcing=sf, t_seconds=t)
        t += DT
        if acc is not None and (k + 1) % sample_every == 0:
            acc.sample(st, dt_step=sample_every * DT, t_seconds=t)
    return st, acc, t


def test_accumulator_does_not_alter_trajectory(grid, z_coord, config, dino_cfg):
    """Flag-off vs flag-on: the PROGNOSTIC state must be bit-identical.
    ``BoxHeatBudgetAccumulator.sample`` reads the state but the run loop
    never feeds its output back into ``model.step`` — verify that holds."""
    model = LatLonCGridOceanModel(grid, z_coord, config)
    forcing = dino_lat_lon_surface_forcing_arrays(model.grid, dino_cfg)
    sf = dino_step_surface_forcing(forcing)

    base = rest_state_latlon_cgrid_ocean(
        model.grid, z_coord, T_water_init_C=20.0, T_deep=4.0, S_uniform=35.0,
        H_max=3000.0,
    )
    rng = np.random.default_rng(1)
    u = 0.02 * rng.standard_normal((model.grid.n_lat, model.grid.n_lon + 1, z_coord.n_levels))
    v = 0.02 * rng.standard_normal((model.grid.n_lat + 1, model.grid.n_lon, z_coord.n_levels))
    st0 = base._replace(
        u=base.u.replace(data=jnp.asarray(u) * (base.u_mask.data[..., None] > 0.5)),
        v=base.v.replace(data=jnp.asarray(v) * (base.v_mask.data[..., None] > 0.5)),
    )

    st_off, acc_off, _ = _run(st0, model, forcing, sf, dino_cfg,
                              n_steps=6, sample_every=2, accumulate=False)
    st_on, acc_on, _ = _run(st0, model, forcing, sf, dino_cfg,
                            n_steps=6, sample_every=2, accumulate=True)

    assert acc_off is None
    assert acc_on is not None
    for field in ("T", "S", "u", "v", "eta"):
        a = np.asarray(getattr(st_off, field).data)
        b = np.asarray(getattr(st_on, field).data)
        np.testing.assert_array_equal(
            a, b, err_msg=f"field {field!r} diverged with accumulator ON"
        )


def test_accumulator_closure_and_nontrivial_terms(grid, z_coord, config, dino_cfg):
    """With VERTMIX now DIRECTLY measured (#1226), the accounting identity
    ``dH == sum(5 terms) + residual`` is still exact by construction of how
    ``residual`` is computed (verify no arithmetic bug), but the REAL gate
    is now that ``residual`` is SMALL relative to the band's own terms (a
    genuine closure diagnostic, not a tautological zero) on a short
    synthetic run, and that the directly-computed terms (incl. vertmix)
    are not degenerately zero, so the decomposition is doing real work."""
    model = LatLonCGridOceanModel(grid, z_coord, config)
    forcing = dino_lat_lon_surface_forcing_arrays(model.grid, dino_cfg)
    sf = dino_step_surface_forcing(forcing)

    rng = np.random.default_rng(2)
    base = rest_state_latlon_cgrid_ocean(
        model.grid, z_coord, T_water_init_C=20.0, T_deep=4.0, S_uniform=35.0,
        H_max=3000.0,
    )
    # Gentle velocity perturbation (0.01 m/s): the residual (#1226) is now
    # a REAL closure diagnostic, not a tautological zero, so it is
    # sensitive to advection-scheme nonlinearity (FCT limiter) at a given
    # perturbation amplitude -- keep the perturbation modest so the tight
    # absolute-tolerance gate below is meaningful, not chasing scheme
    # nonlinearity noise.
    u = 0.01 * rng.standard_normal((model.grid.n_lat, model.grid.n_lon + 1, z_coord.n_levels))
    v = 0.01 * rng.standard_normal((model.grid.n_lat + 1, model.grid.n_lon, z_coord.n_levels))
    st0 = base._replace(
        u=base.u.replace(data=jnp.asarray(u) * (base.u_mask.data[..., None] > 0.5)),
        v=base.v.replace(data=jnp.asarray(v) * (base.v_mask.data[..., None] > 0.5)),
    )

    n_steps, sample_every = 20, 4
    _, acc, t_final = _run(st0, model, forcing, sf, dino_cfg,
                           n_steps=n_steps, sample_every=sample_every,
                           accumulate=True)

    assert acc.n_samples == n_steps // sample_every + 1
    total_seconds = t_final  # first sample at t=0
    summary = acc.summary(total_seconds)

    # Whole-box heat-content scale (all 3 bands): the reference scale for
    # judging "small" residual in ABSOLUTE terms -- using a single band's
    # own dH/term-sum as the denominator is unstable when that band's own
    # terms happen to nearly cancel (small dH from large opposing terms is
    # real physics, not a residual problem), so this test judges the
    # residual against the run's total heat-content magnitude instead.
    box_scale_J = max(
        sum(abs(summary["terms"][b]["dH_J"]) for b in BANDS), 1.0,
    )

    any_nonzero = {term: False for term in TERM_NAMES}
    for band in BANDS:
        band_out = summary["terms"][band]
        # Accounting identity: dH == sum(5 terms) + residual, by
        # construction of how `residual` is accumulated in `.sample()` —
        # verify the arithmetic actually reproduces that identity (catches
        # a double-count / mismatched-interval accounting bug). This is
        # ALSO asserted inside `summary()` itself; re-checking here from
        # the returned dict guards against a future refactor dropping it.
        term_sum_J = sum(band_out[t]["J"] for t in TERM_NAMES)
        np.testing.assert_allclose(
            term_sum_J + band_out["residual"]["J"], band_out["dH_J"],
            rtol=1e-9, atol=1e-9 * max(abs(band_out["dH_J"]), 1.0),
            err_msg=f"budget accounting identity broken for band {band}",
        )
        # The REAL gate (#1226): residual must be small relative to the
        # run's total box heat-content scale -- a genuine closure
        # diagnostic bound now that VERTMIX is directly measured (not the
        # machine-epsilon tautology of the old residual-is-vertmix
        # design). 5% is tight in absolute terms (measured 0.2-4.2% across
        # bands on this fixture) while allowing for the residual
        # operator-splitting-order gap inherent to evaluating each
        # explicit term's rate independently of the model's own internal
        # step sequencing -- e.g. the advection/FCT-limiter terms are
        # evaluated on the PRE-advection entry state, while the model's
        # actual step composes advection THEN implicit vertmix on the
        # intermediate advected state (module docstring CAVEAT); this
        # concentrates in the deepest band, where forcing is ~zero and the
        # residual's absolute size is dominated by advective
        # operator-order error rather than any per-term bug. This is a
        # PRE-EXISTING property of ``compute_box_heat_dT_terms`` (present
        # before #1226, previously invisible because it was silently
        # absorbed into the old residual-is-vertmix design) and is NOT
        # specific to the direct vertmix measurement -- confirmed by
        # reproducing the same magnitude on pre-#1226 code (git stash).
        assert abs(band_out["residual"]["J"]) < 0.05 * box_scale_J, (
            f"band {band}: residual {band_out['residual']['J']:.3e} J is "
            f"not small vs box heat-content scale {box_scale_J:.3e} J -- "
            "direct-vertmix closure is not tight"
        )
        for term in TERM_NAMES:
            if abs(band_out[term]["J"]) > 1e-8:
                any_nonzero[term] = True

    # At least the directly-computed terms must be doing real work on a
    # perturbed, forced, GM/Redi-enabled state.
    assert any_nonzero["adv_h"] or any_nonzero["adv_v"], (
        "advection terms are degenerately zero — box/sampling misconfigured"
    )
    assert any_nonzero["forcing"], "forcing term is degenerately zero"
    assert any_nonzero["vertmix"], (
        "vertmix term is degenerately zero -- direct implicit-mixing probe "
        "may not be wired (A_v/K_v=0 in this fixture would also cause this)"
    )

    assert summary["box_area_m2"] > 0.0
    for band in BANDS:
        for term in TERM_NAMES:
            assert np.isfinite(summary["terms"][band][term]["W_per_m2"])
        assert np.isfinite(summary["terms"][band]["residual"]["W_per_m2"])


def test_time_series_present_and_consistent_with_totals(grid, z_coord, config, dino_cfg):
    """#1226: per-term TIME SERIES (not just cumulative scalars) so budgets
    can be re-windowed without re-running. Verify the new keys exist, have
    the expected shape (one entry per closed interval), and SUM to the
    same cumulative totals ``summary()`` reports (both are populated by
    the same ``.sample()`` call — a mismatch would mean the two bookkeeping
    paths diverged)."""
    model = LatLonCGridOceanModel(grid, z_coord, config)
    forcing = dino_lat_lon_surface_forcing_arrays(model.grid, dino_cfg)
    sf = dino_step_surface_forcing(forcing)

    rng = np.random.default_rng(3)
    base = rest_state_latlon_cgrid_ocean(
        model.grid, z_coord, T_water_init_C=20.0, T_deep=4.0, S_uniform=35.0,
        H_max=3000.0,
    )
    u = 0.03 * rng.standard_normal((model.grid.n_lat, model.grid.n_lon + 1, z_coord.n_levels))
    v = 0.03 * rng.standard_normal((model.grid.n_lat + 1, model.grid.n_lon, z_coord.n_levels))
    st0 = base._replace(
        u=base.u.replace(data=jnp.asarray(u) * (base.u_mask.data[..., None] > 0.5)),
        v=base.v.replace(data=jnp.asarray(v) * (base.v_mask.data[..., None] > 0.5)),
    )

    n_steps, sample_every = 24, 4
    n_intervals = n_steps // sample_every
    _, acc, t_final = _run(st0, model, forcing, sf, dino_cfg,
                           n_steps=n_steps, sample_every=sample_every,
                           accumulate=True)
    summary = acc.summary(t_final)

    for i in range(len(BANDS)):
        for term in TERM_NAMES:
            series = acc.time_series_term_J[term][i]
            assert len(series) == n_intervals, (
                f"band {i} term {term!r}: expected {n_intervals} interval "
                f"samples, got {len(series)}"
            )
            np.testing.assert_allclose(
                float(np.sum(series)), summary["terms"][BANDS[i]][term]["J"],
                rtol=1e-9, atol=1e-6,
                err_msg=(
                    f"band {i} term {term!r}: interval time series does not "
                    "sum to the cumulative total"
                ),
            )
        resid_series = acc.time_series_residual_J[i]
        assert len(resid_series) == n_intervals
        np.testing.assert_allclose(
            float(np.sum(resid_series)),
            summary["terms"][BANDS[i]]["residual"]["J"],
            rtol=1e-9, atol=1e-6,
            err_msg=f"band {i}: residual time series does not sum to the total",
        )

    # time_series_t / time_series_H (pre-existing) also has one entry per
    # SAMPLE (n_intervals + 1, incl. the t=0 baseline sample).
    assert len(acc.time_series_t) == n_intervals + 1
    for i in range(len(BANDS)):
        assert len(acc.time_series_H[i]) == n_intervals + 1
    for band in BANDS:
        for term in TERM_NAMES:
            assert np.isfinite(summary["terms"][band][term]["W_per_m2"])


@pytest.fixture
def fp64_storage():
    """fp64 STORAGE for the K33 tests -- matches how the #1226 instrument is
    actually run (the drivers gate on ``require_fp64``), and is REQUIRED for
    the k33 bucket to be meaningfully resolved.

    Measured (#1226 review, fixture-exact probe): under the default fp32
    storage policy the k33 realized increment is
    ``(T_after - T_before)/dt`` where both T's are O(23 degC) -- a
    catastrophic cancellation. float32's ULP at T~23.7 is ~2.8e-6, so an
    increment of ~4e-9 degC/s (=3.8e-6 degC over dt=900 s) survives as only
    ~1.4 ULP: nonzero (208 cells) but quantized to ULP multiples. At that
    resolution the bucket cannot distinguish real physics from rounding --
    e.g. the ``eos_depth`` insitu-vs-geometric K33 difference (rel-L2
    1.436e-03, definitely present in the K33 FIELD) is invisible in the
    fp32 increment (bit-identical), because it is far finer than one ULP.
    fp64 makes the increment well-resolved and the assertions meaningful.

    Saves/restores the global policy (same pattern as
    ``tests/unit/test_cmor_output_dtype.py``) so no other test is affected.
    """
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(saved)


@pytest.fixture
def state_k33(grid, z_coord, fp64_storage):
    """A rest state (built fp64, see ``fp64_storage``) plus a meridional
    temperature FRONT CENTERED INSIDE the
    box-budget's own channel rows (``ROW_SLICE = slice(4, 20)`` on this
    ``n_lat=24`` grid) -- K_33 is ∝ (isoneutral slope)^2, which is exactly
    zero on ``state``'s horizontally-uniform initial T
    (rest_state_latlon_cgrid_ocean has no lon/lat T gradient), so a front
    is REQUIRED to get a nonzero, demonstrably-real K33 rather than a
    degenerate all-zero case.

    A domain-edge-to-edge LINEAR ramp (tried first) puts essentially all of
    its curvature/slope signal in the single row adjacent to the grid
    boundary (row 22 of 24 here) -- OUTSIDE ``ROW_SLICE`` -- so the box
    budget integrates a hard zero even though K33 is computed correctly
    (verified directly with a scratch probe, #1226 instrument-fix review:
    row-by-row max|K33| was 0 everywhere except row 22). A ``tanh`` front
    CENTERED at the channel midpoint (row 12) with a few-row width puts the
    slope signal on rows ~9-14, well inside ``ROW_SLICE`` -- verified
    directly to give a nonzero, non-degenerate ``k33`` box-integrated
    power. Amplitude (4 degC) and ``config_k33.gm_redi.kappa_Redi`` (below)
    are tuned together (also probe-verified): K33 ~ kappa_Redi * slope^2
    underflows the grid's per-step (K33 * dt / dz^2) realized increment
    well below the base ``config`` fixture's kappa_Redi=50 -- numerically
    negligible at that magnitude on this small grid, not a wiring bug.

    Builds its OWN rest state rather than reusing the ``state`` fixture:
    ``rest_state_latlon_cgrid_ocean`` reads ``get_policy().storage`` at
    CONSTRUCTION time, so the state must be created while ``fp64_storage``
    is active or it would be fp32 regardless of the policy override.
    """
    base = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=4.0, S_uniform=35.0,
        H_max=3000.0,
    )
    rng = np.random.default_rng(0)
    u = 0.02 * rng.standard_normal((grid.n_lat, grid.n_lon + 1, z_coord.n_levels))
    v = 0.02 * rng.standard_normal((grid.n_lat + 1, grid.n_lon, z_coord.n_levels))
    base = base._replace(
        u=base.u.replace(data=jnp.asarray(u, dtype=base.u.data.dtype)
                         * (base.u_mask.data[..., None] > 0.5)),
        v=base.v.replace(data=jnp.asarray(v, dtype=base.v.data.dtype)
                         * (base.v_mask.data[..., None] > 0.5)),
    )
    rows = jnp.arange(grid.n_lat, dtype=base.T.data.dtype)[:, None, None]
    front_center_row = 0.5 * (ROW_SLICE.start + ROW_SLICE.stop)
    front_width_rows = 3.0
    T_front = 4.0 * jnp.tanh(
        (rows - front_center_row) / front_width_rows
    ) * jnp.ones_like(base.T.data)
    T_new = (base.T.data + T_front) * base.land_mask.data[:, :, None]
    return base._replace(T=base.T.replace(data=T_new))


@pytest.fixture
def config_k33():
    """Same as ``config`` but with ``implicit_K33=True`` -- the #1226
    regression fixture: exercises the path where K_33 is dropped from the
    explicit GM/Redi flux and must be recovered as its own bucket.
    ``kappa_Redi=2000`` (vs the base ``config`` fixture's 50): K33 scales
    with kappa_Redi, and 50 (paired with a physically modest T-front)
    underflows to a degenerate all-zero realized increment on this small
    grid/timestep -- verified directly (probe script, #1226 review) that
    2000 (a realistic DINO-recipe magnitude; ``dino.py`` uses kappa_Redi
    up to a few thousand m^2/s) gives a measurably nonzero k33 bucket.

    ``eos_depth`` NOT set here (stays at the ``LatLonCGridOceanConfig``
    default, ``"insitu"``) -- this fixture does NOT exercise the
    eos_depth="geometric" convention the real DINO NEMO recipes pin
    (``experiments/dino.py:909,933``). Verified directly (probe, #1226
    review) on this small 8-level/24x16 test grid: switching this
    fixture's config to ``eos_depth="geometric"`` DOES change ``iso_redi``
    measurably (rel-L2 ~3.5e-4, not array-equal -- proving the kwarg is
    correctly threaded and reaches production code), but the realized
    ``k33`` bucket comes back ARRAY-EQUAL between conventions on this
    grid/perturbation -- the K33 magnitude itself differs, but the
    resulting implicit-diffusion number is small enough here that the
    difference underflows the realized ``(T_after-T_before)/dt`` increment.
    So this fixture (and any test built on it) is NOT a valid check that
    the K33 bucket's own eos_depth threading matters numerically -- use
    ``iso_redi`` (or ``compute_isoneutral_K33_latlon``'s raw output before
    the implicit solve) for that, not this fixture's ``k33`` bucket. On the
    real DINO grid (many more levels, real z-star stretching, kappa_Redi in
    the thousands, deeper columns) the diffusion number is far larger and
    the geometric-vs-insitu K33 difference is expected to survive into the
    realized bucket."""
    gm = GMRediConfig(kappa_GM=50.0, kappa_Redi=2000.0, slope_scheme="nemo_iso_lap",
                      implicit_K33=True)
    return LatLonCGridOceanConfig.from_flat(
        A_h=1.0e3, A_v=1.0e-3, K_h=0.0, K_v=1.0e-4,
        bottom_drag_r=1.0e-3, gm_redi=gm,
        implicit_vertical_mixing=True,
        outer_integrator="forward_euler",
    )


def test_k33_bucket_closes_budget_and_is_nonzero(
    grid, z_coord, config_k33, dino_cfg, fp64_storage, state_k33,
):
    """#1226 instrument-fix test 1 (required): with ``implicit_K33=True``
    (the DINO recipe default), the accounting identity ``dH == sum(6
    terms, incl. k33) + residual`` holds AND the k33 bucket is genuinely
    nonzero -- i.e. K_33 is being measured, not silently zero. This is the
    test that would have caught the original defect: before the fix,
    ``compute_box_heat_dT_terms`` had no "k33" key at all, so this test
    would fail outright (KeyError) rather than merely report a zero bucket.
    """
    model = LatLonCGridOceanModel(grid, z_coord, config_k33)
    forcing = dino_lat_lon_surface_forcing_arrays(model.grid, dino_cfg)
    sf = dino_step_surface_forcing(forcing)

    n_steps, sample_every = 4, 2
    _, acc, t_final = _run(state_k33, model, forcing, sf, dino_cfg,
                           n_steps=n_steps, sample_every=sample_every,
                           accumulate=True)
    summary = acc.summary(t_final)

    assert "k33" in TERM_NAMES
    box_scale_J = max(
        sum(abs(summary["terms"][b]["dH_J"]) for b in BANDS), 1.0,
    )
    # Deepest band on this 8-level/3000m fixture is a SINGLE z-level
    # (z_cum ~ 2643 m, the only cell >= 2000 m -- see z_coord.dz_ref), with
    # zero FORCING (no surface flux reaches it) and negligible K33 (GM/Redi's
    # bottom taper suppresses the isoneutral slope there) -- the same
    # pre-existing forcing-free-band operator-splitting-order residual
    # ``test_accumulator_closure_and_nontrivial_terms`` documents (its own
    # 5% bound), NOT specific to the K33 fix. Verified directly (probe,
    # #1226 review): this band's k33 is exactly 0 on this fixture, so its
    # closure is governed entirely by the pre-existing advection-order gap,
    # not by anything this fix changes -- loosen ONLY this band's bound.
    residual_tol_frac = {BANDS[0]: 0.05, BANDS[1]: 0.05, BANDS[2]: 0.15}
    any_k33_nonzero = False
    for band in BANDS:
        band_out = summary["terms"][band]
        term_sum_J = sum(band_out[t]["J"] for t in TERM_NAMES)
        np.testing.assert_allclose(
            term_sum_J + band_out["residual"]["J"], band_out["dH_J"],
            rtol=1e-9, atol=1e-9 * max(abs(band_out["dH_J"]), 1.0),
            err_msg=f"budget accounting identity broken for band {band} "
                    "with K33 included",
        )
        tol = residual_tol_frac[band]
        assert abs(band_out["residual"]["J"]) < tol * box_scale_J, (
            f"band {band}: residual {band_out['residual']['J']:.3e} J is "
            f"not small ({tol:.0%} bound) vs box heat-content scale "
            f"{box_scale_J:.3e} J with implicit_K33=True -- the K33 bucket "
            "is not closing the budget"
        )
        if abs(band_out["k33"]["J"]) > 1e-10:
            any_k33_nonzero = True
    assert any_k33_nonzero, (
        "k33 bucket is degenerately zero with implicit_K33=True -- K_33 is "
        "not actually being measured"
    )


def test_k33_bucket_is_nonvacuous_synthetic_violation(
    grid, z_coord, config_k33, dino_cfg, fp64_storage, state_k33, monkeypatch,
):
    """#1226 instrument-fix test 2 (required, non-vacuous / synthetic-
    violation check): reproduce the ORIGINAL defect -- K_33 silently
    dropped -- by monkeypatching ``compute_isoneutral_K33_latlon`` in the
    ``box_heat_budget`` module namespace (the symbol
    ``compute_box_heat_dT_terms`` ACTUALLY calls to build the k33 bucket,
    per its import at the top of the module -- not a delegating wrapper)
    to return an all-zero diffusivity, matching the ORIGINAL bug's
    observable effect (K33 = 0 everywhere). Confirm:
      (a) with the real helper, the k33 bucket is measurably NONZERO
          (baseline, no monkeypatch);
      (b) with the helper forced to zero, the k33 bucket goes to EXACTLY
          zero on every band -- i.e. the bucket really is sourced from
          ``compute_isoneutral_K33_latlon`` and not some other pathway
          that would leave it unchanged.
    This proves test 1 above is exercising the real computation: if the
    k33 bucket were hardcoded to zero (the pre-fix behaviour), (a) would
    fail outright, so the synthetic violation is caught. (Note: this does
    NOT also assert the closure RESIDUAL grows when K33 is zeroed. The
    monkeypatch targets ONLY ``box_heat_budget.compute_isoneutral_K33_latlon``
    -- the model's own step loop (``ocean_model_latlon_cgrid.py``) imports
    the SAME symbol into its OWN module namespace and is unaffected by this
    monkeypatch, so the real vs zeroed runs' PROGNOSTIC trajectories are
    actually identical (this was verified, not merely assumed -- an earlier
    docstring here incorrectly attributed the omission to a claimed
    trajectory-feedback effect that does not exist for this monkeypatch).
    The residual-growth check is skipped instead because the accumulator's
    K33 bucket is computed via a SEPARATE standalone solve from the one
    ``compute_box_vertmix_dT`` makes (see ``compute_box_heat_dT_terms``
    docstring's "Split-solve error magnitude" note) -- zeroing K33 there
    changes k33->0 exactly (asserted below) but the residual's response
    also folds in that pre-existing split-solve approximation, so a
    residual-growth bound would conflate two different effects. The
    k33-bucket exact-zero identity check above is the robust,
    assumption-free non-vacuity proof.)
    """
    import legoesm.ocean.fidelity.box_heat_budget as bhb

    model = LatLonCGridOceanModel(grid, z_coord, config_k33)
    forcing = dino_lat_lon_surface_forcing_arrays(model.grid, dino_cfg)
    sf = dino_step_surface_forcing(forcing)

    n_steps, sample_every = 4, 2
    _, acc_real, t_real = _run(state_k33, model, forcing, sf, dino_cfg,
                               n_steps=n_steps, sample_every=sample_every,
                               accumulate=True)
    summary_real = acc_real.summary(t_real)
    k33_real_J = {b: summary_real["terms"][b]["k33"]["J"] for b in BANDS}
    assert any(abs(v) > 1e-10 for v in k33_real_J.values()), (
        "baseline (real K33) run has a degenerately-zero k33 bucket -- "
        "the fixture is not exercising implicit_K33"
    )

    def _zero_k33(*args, **kwargs):
        return jnp.zeros_like(state_k33.T.data[..., :-1])

    monkeypatch.setattr(bhb, "compute_isoneutral_K33_latlon", _zero_k33)

    _, acc_zeroed, t_zeroed = _run(state_k33, model, forcing, sf, dino_cfg,
                                   n_steps=n_steps, sample_every=sample_every,
                                   accumulate=True)
    summary_zeroed = acc_zeroed.summary(t_zeroed)

    for band in BANDS:
        assert summary_zeroed["terms"][band]["k33"]["J"] == 0.0, (
            "monkeypatched compute_isoneutral_K33_latlon should force the "
            "k33 bucket to exactly zero -- if this fails, the k33 bucket "
            "is not actually sourced from compute_isoneutral_K33_latlon"
        )
        # Non-vacuity: at least one band's real (non-monkeypatched) k33
        # must differ from its (exactly-zero) monkeypatched counterpart --
        # if the k33 bucket were hardcoded to zero (the pre-fix defect),
        # real and zeroed would be IDENTICAL and this would fail, catching
        # exactly the synthetic violation this test reproduces.
    assert any(
        abs(k33_real_J[band]) != abs(summary_zeroed["terms"][band]["k33"]["J"])
        for band in BANDS
    ), (
        "real and monkeypatched-zero k33 buckets are identical on every "
        "band -- the k33 bucket does not actually depend on "
        "compute_isoneutral_K33_latlon's return value"
    )


def test_physics_dt_is_the_model_timestep_not_the_sampling_interval(
    grid, z_coord, config, dino_cfg, state,
):
    """Regression guard (#1226 physics-validator review finding): the
    accumulator MUST feed the model's own dynamical dt (``DT``) to
    ``restoring_surface_forcing``/the FCT limiter/the GM/Redi MSC clamp —
    NOT the (much longer) sampling interval. All three are genuinely
    dt-sensitive (not merely dt-rescaled), so conflating the two silently
    biases every explicit term. Direct check: call
    ``compute_box_heat_dT_terms`` with the correct model dt vs a 32x-longer
    "sampling interval" dt and confirm the FORCING term (the clearest
    dt-sensitive case: ``restoring_surface_forcing(implicit=True)`` uses
    ``eff_tau = tau + dt``) materially differs — i.e. dt really matters,
    so accidentally passing the wrong one is not a no-op."""
    forcing = dino_lat_lon_surface_forcing_arrays(grid, dino_cfg)
    terms_model_dt = compute_box_heat_dT_terms(
        state, grid, z_coord, config, dino_cfg, forcing, DT, t_seconds=0.0,
    )
    terms_sampling_dt = compute_box_heat_dT_terms(
        state, grid, z_coord, config, dino_cfg, forcing, DT * 32, t_seconds=0.0,
    )
    mask = np.asarray(state.land_mask.data) > 0.5
    forcing_model = np.asarray(terms_model_dt["forcing"])[mask]
    forcing_sampling = np.asarray(terms_sampling_dt["forcing"])[mask]
    assert not np.allclose(forcing_model, forcing_sampling, rtol=1e-6), (
        "FORCING term is insensitive to dt -- the dt-conflation regression "
        "guard would not catch a re-introduced bug; investigate whether "
        "restoring_surface_forcing's implicit tau+dt path is still wired."
    )

    # End-to-end guard: sample() must use dt_model (=DT, fixed at
    # construction), not dt_step (which varies with sampling cadence) --
    # the reported per-second FORCING rate must be ~IDENTICAL whether
    # sampled every 2 or every 4 steps (same physics, different bookkeeping
    # cadence only).
    model = LatLonCGridOceanModel(grid, z_coord, config)
    sf = dino_step_surface_forcing(forcing)
    _, acc2, t2 = _run(state, model, forcing, sf, dino_cfg,
                       n_steps=8, sample_every=2, accumulate=True)
    _, acc4, t4 = _run(state, model, forcing, sf, dino_cfg,
                       n_steps=8, sample_every=4, accumulate=True)
    s2 = acc2.summary(t2)
    s4 = acc4.summary(t4)
    for band in BANDS:
        f2 = s2["terms"][band]["forcing"]["W_per_m2"]
        f4 = s4["terms"][band]["forcing"]["W_per_m2"]
        if abs(f2) < 1e-12 and abs(f4) < 1e-12:
            continue
        # Loose tolerance: sample_every=2 vs 4 also evaluates the endpoint
        # rate at slightly different states along a genuinely-evolving
        # trajectory (real physics, not a bug) -- O(1e-4) relative here.
        # The dt-conflation bug this guards against was ~8% (tau_T-scale),
        # 4 orders of magnitude above the trajectory-divergence noise floor.
        np.testing.assert_allclose(
            f2, f4, rtol=1e-2,
            err_msg=(
                f"band {band}: FORCING W/m^2 depends on sampling cadence "
                "(sample_every=2 vs 4) far beyond trajectory-divergence "
                "noise -- dt_step may be leaking into the physics dt again"
            ),
        )
