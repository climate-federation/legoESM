"""Deterministic max-random-overlap subcolumn generator.

The offline validation that justified this scheme (prodA day 310, Monte-Carlo
ICA reference) depended on four properties of the generator. They are pinned
here so a future edit cannot quietly break the thing the measurement rested on:
per-layer marginal = cf, total cover = the model's OWN maximum_random_overlap,
grid-mean water conserved, and the clear/overcast limits exact.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.clouds.subcolumns import (
    N_SUBCOLUMNS_DEFAULT,
    average_over_subcolumns,
    expand_to_subcolumns,
    generate_subcolumns,
    subcolumn_paths,
)
from legoesm.diagnostics.cloud_overlap import maximum_random_overlap

NLEV = 30


def _cf(ncol=16, seed=3):
    """Cloud profiles spanning clear, thin, broken and overcast layers."""
    r = np.random.default_rng(seed)
    cf = r.random((ncol, NLEV)) ** 2          # skewed toward small cf, as in the model
    cf[:, :4] = 0.0                            # clear top
    cf[0, :] = 0.0                             # a fully clear column
    cf[1, :] = 1.0                             # a fully overcast column
    return jnp.asarray(cf)


# ------------------------------------------------------------------- limits

def test_clear_column_is_never_cloudy():
    m = np.asarray(generate_subcolumns(jnp.zeros((3, NLEV))))
    assert not m.any()


def test_overcast_column_is_always_cloudy():
    m = np.asarray(generate_subcolumns(jnp.ones((3, NLEV))))
    assert m.all()


def test_shape_and_dtype():
    for n in (4, 8, 12):
        m = generate_subcolumns(_cf(), n)
        assert m.shape == (n, 16, NLEV)
        assert m.dtype == jnp.bool_


# --------------------------------------------------------------- statistics

def test_per_layer_marginal_reproduces_cloud_fraction():
    """Stratification makes this far tighter than random sampling would.

    With n_sub draws the binomial standard error is ~sqrt(p(1-p)/n) ~ 0.09 at
    n=32; stratification removes most of that, so a loose-but-real bound is
    used. An earlier version of this measurement rejected a CORRECT generator
    by comparing against an arbitrary constant instead of the noise floor.
    """
    cf = _cf()
    m = np.asarray(generate_subcolumns(cf, 32))
    p = m.mean(axis=0)
    err = np.abs(p - np.asarray(cf))
    assert err.mean() < 0.02, f"mean marginal error {err.mean():.4f}"
    assert err.max() < 0.30


def test_total_cover_matches_the_models_own_overlap_function():
    """Must agree with maximum_random_overlap, not a formula re-derived here."""
    cf = _cf(ncol=24)
    m = np.asarray(generate_subcolumns(cf, 64))
    got = m.any(axis=2).mean(axis=0)
    want = np.asarray(maximum_random_overlap(cf))
    assert np.abs(got - want).mean() < 0.03
    assert got[0] == pytest.approx(0.0)        # clear column
    assert got[1] == pytest.approx(1.0)        # overcast column


def test_adjacent_cloudy_layers_overlap_maximally():
    """Two identical adjacent cloudy layers must give cover == cf, not 1-(1-cf)^2.

    This is the property that distinguishes maximum-random from purely random
    overlap; if the recursion were wrong the cover would be too large.
    """
    cf = jnp.full((4, 2), 0.5)
    m = np.asarray(generate_subcolumns(cf, 64))
    assert m.any(axis=2).mean(axis=0).mean() == pytest.approx(0.5, abs=0.05)
    both = (m[:, :, 0] & m[:, :, 1]).mean()
    assert both == pytest.approx(0.5, abs=0.05), "layers are not maximally overlapped"


def test_separated_cloud_groups_overlap_randomly():
    """A clear layer between two cloudy ones resets to random overlap."""
    cf = jnp.asarray(np.array([[0.5, 0.0, 0.5]] * 4))
    m = np.asarray(generate_subcolumns(cf, 256))
    cover = m.any(axis=2).mean(axis=0).mean()
    assert cover == pytest.approx(1.0 - 0.5 * 0.5, abs=0.06)


# ---------------------------------------------------------------- water use

def test_grid_mean_water_is_conserved():
    cf = _cf()
    lwp = jnp.asarray(np.asarray(cf) * 2.0e-4)     # grid-mean = cf * in-cloud
    iwp = jnp.asarray(np.asarray(cf) * 5.0e-5)
    n = 64
    m = generate_subcolumns(cf, n)
    ls, is_ = subcolumn_paths(m, cf, lwp, iwp)
    lm = np.asarray(average_over_subcolumns(ls, n, cf.shape[0]))
    im = np.asarray(average_over_subcolumns(is_, n, cf.shape[0]))
    # Per-layer, only where n_sub can RESOLVE the fraction: at cf < ~1/n_sub a
    # layer gets 0 or 1 cloudy subcolumns, so its relative error is unbounded
    # by construction and an elementwise rtol there tests nothing. (Same trap
    # as the earlier Monte-Carlo probe -- an elementwise tolerance on a sampled
    # mean with a tiny denominator is meaningless.)
    sel = np.asarray(cf) > 5.0 / n
    assert sel.sum() > 50, "fixture must have enough resolvable layers"
    # Sampled quantity: assert the DISTRIBUTION, not every element. Demanding
    # every layer inside a tolerance fails on one outlier in ~300 by chance and
    # says nothing about conservation.
    rel = np.abs(lm[sel] - np.asarray(lwp)[sel]) / np.asarray(lwp)[sel]
    assert np.median(rel) < 0.05, f"median per-layer error {np.median(rel):.3f}"
    assert np.quantile(rel, 0.95) < 0.35, f"p95 {np.quantile(rel, 0.95):.3f}"
    reli = np.abs(im[sel] - np.asarray(iwp)[sel]) / np.asarray(iwp)[sel]
    assert np.median(reli) < 0.05
    # The COLUMN TOTAL is what radiation integrates, and it averages the
    # granularity out -- this is the assertion that matters for conservation.
    assert lm.sum() == pytest.approx(float(np.asarray(lwp).sum()), rel=0.05)
    assert im.sum() == pytest.approx(float(np.asarray(iwp).sum()), rel=0.05)


def test_clear_cells_carry_no_water():
    cf = _cf()
    lwp = jnp.asarray(np.asarray(cf) * 2.0e-4)
    m = generate_subcolumns(cf, 8)
    ls, _ = subcolumn_paths(m, cf, lwp, lwp)
    assert np.all(np.asarray(ls).reshape(np.asarray(m).shape)[~np.asarray(m)] == 0.0)


def test_expand_matches_the_subcolumn_flattening_order():
    """expand_to_subcolumns must use the SAME (n_sub, ncol) -> flat order as
    subcolumn_paths, or every column would be paired with the wrong profile."""
    cf = _cf(ncol=5)
    n = 8
    m = generate_subcolumns(cf, n)
    T = jnp.asarray(np.arange(5 * NLEV, dtype=float).reshape(5, NLEV))
    e = np.asarray(expand_to_subcolumns(m, T))
    assert e.shape == (n * 5, NLEV)
    np.testing.assert_array_equal(e.reshape(n, 5, NLEV)[3], np.asarray(T))


# -------------------------------------------------------------- determinism

def test_is_deterministic_across_calls_and_processes():
    """No PRNG in the hot loop: repeated calls must be bit-identical."""
    cf = _cf()
    a = np.asarray(generate_subcolumns(cf, 8))
    b = np.asarray(generate_subcolumns(cf, 8))
    np.testing.assert_array_equal(a, b)


def test_jit_parity():
    cf = _cf()
    eager = np.asarray(generate_subcolumns(cf, 8))
    jitted = np.asarray(jax.jit(lambda c: generate_subcolumns(c, 8))(cf))
    np.testing.assert_array_equal(eager, jitted)


def test_water_path_gradient_is_finite():
    """The mask thresholds, but the PATHS must still carry a gradient."""
    cf = _cf(ncol=4)
    m = generate_subcolumns(cf, 8)

    def total(lwp):
        ls, _ = subcolumn_paths(m, cf, lwp, jnp.zeros_like(lwp))
        return jnp.sum(ls)

    g = np.asarray(jax.grad(total)(jnp.asarray(np.asarray(cf) * 2.0e-4)))
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)


def test_default_subcolumn_count_is_the_validated_one():
    """8 was chosen by a measured sweep (4 leaves 18% of the signal, 8 leaves
    2.5%). A change here invalidates the offline validation in the module
    docstring and must be re-measured."""
    assert N_SUBCOLUMNS_DEFAULT == 8


# --------------------------------------------------- solver-plumbing helpers

def test_expand_kwargs_expands_only_the_declared_column_keys():
    """Explicit key list, not a shape heuristic. Anything not declared
    per-column passes through untouched, whatever its shape."""
    from legoesm.atmosphere.physics.clouds.subcolumns import expand_kwargs

    ncol, n = 5, 4
    kw = {
        "T": jnp.ones((ncol, NLEV)),
        "cos_zenith": jnp.arange(ncol, dtype=float),
        "sfc_albedo": None,
        "sfc_emissivity": 0.98,                 # scalar, must stay scalar
        "ghg_vmr_override": {"co2": 4.15e-4},   # not per-column
    }
    out = expand_kwargs(kw, n, ncol)
    assert out["T"].shape == (n * ncol, NLEV)
    assert out["cos_zenith"].shape == (n * ncol,)
    assert out["sfc_albedo"] is None
    assert out["sfc_emissivity"] == 0.98
    assert out["ghg_vmr_override"] == {"co2": 4.15e-4}
    # ordering must match the subcolumn flattening used for the water paths
    np.testing.assert_array_equal(
        np.asarray(out["cos_zenith"]).reshape(n, ncol)[2], np.arange(ncol))


def test_average_output_inverts_expansion_for_identical_subcolumns():
    """If every subcolumn is identical, averaging must return the original."""
    from legoesm.atmosphere.physics.clouds.subcolumns import (
        average_output, expand_kwargs)
    from legoesm.atmosphere.physics.radiation.output import RadiationOutput

    ncol, n = 6, 8
    f = jnp.asarray(np.arange(ncol * (NLEV + 1), dtype=float
                              ).reshape(ncol, NLEV + 1))
    h = jnp.asarray(np.arange(ncol * NLEV, dtype=float).reshape(ncol, NLEV))
    big = expand_kwargs({"p_half": f, "T": h}, n, ncol)
    r = RadiationOutput(big["p_half"], big["p_half"], big["p_half"],
                        big["p_half"], big["T"], big["T"], big["T"], None)
    avg = average_output(r, n, ncol)
    np.testing.assert_allclose(np.asarray(avg.sw_flux_up), np.asarray(f))
    np.testing.assert_allclose(np.asarray(avg.heating_rate), np.asarray(h))
    assert avg.toa_insolation is None


def test_stratified_table_is_not_mutable_by_callers():
    """The lru_cache hands back the SAME object; a mutation would corrupt
    every later radiation call (codex review 2026-07-31)."""
    from legoesm.atmosphere.physics.clouds.subcolumns import _stratified_table

    a = _stratified_table(8, 12)
    b = _stratified_table(8, 12)
    assert a is not b, "callers must not share one mutable table"
    np.testing.assert_array_equal(a, b)
    a[0, 0] = 99.0                      # mutate the copy...
    np.testing.assert_array_equal(_stratified_table(8, 12), b)   # ...no effect
    # and the generator itself is unaffected
    cf = _cf(ncol=3)
    np.testing.assert_array_equal(
        np.asarray(generate_subcolumns(cf, 8)),
        np.asarray(generate_subcolumns(cf, 8)))


def test_expand_kwargs_refuses_a_wrong_leading_dimension():
    """A per-column key whose layout changed must fail loudly, not expand."""
    from legoesm.atmosphere.physics.clouds.subcolumns import expand_kwargs

    with pytest.raises(ValueError, match="leading dimension"):
        expand_kwargs({"T": jnp.zeros((7, NLEV))}, 4, ncol=5)


def test_expand_kwargs_leaves_gpoint_arrays_alone():
    """solar_spectral_fraction is (ngpt_sw,), NOT per-column. The old shape
    heuristic would have mis-expanded it whenever ngpt == ncol."""
    from legoesm.atmosphere.physics.clouds.subcolumns import expand_kwargs

    ncol = 112                                   # == a realistic ngpt_sw
    kw = {"solar_spectral_fraction": jnp.zeros(ncol),
          "T": jnp.zeros((ncol, NLEV))}
    out = expand_kwargs(kw, 4, ncol)
    assert out["solar_spectral_fraction"].shape == (ncol,), "g-point array expanded!"
    assert out["T"].shape == (4 * ncol, NLEV)


# --------------------------------------------- driver / CLI reachability

def test_mpas_lane_forwards_the_overlap_flags():
    """`_standalone_cloud_config` must not silently drop the new knobs.

    PR #1385 fixed exactly this for five other cloud kwargs: the CLI accepted
    them and the MPAS lane discarded them, so arms ran without the physics they
    were launched to test.
    """
    from legoesm.driver.model_driver import _standalone_cloud_config

    class _Cfg:
        cloud_vertical_overlap_optics = "max_random"
        cloud_n_subcolumns = 12

    got = _standalone_cloud_config(_Cfg(), "sundqvist")
    assert got.cloud_vertical_overlap_optics == "max_random"
    assert got.cloud_n_subcolumns == 12


def test_validate_strict_rejects_unknown_overlap_scheme():
    from legoesm.driver.config import ExperimentConfig

    with pytest.raises((ValueError, SystemExit), match="vertical_overlap"):
        ExperimentConfig(cloud_vertical_overlap_optics="nope").validate_strict()


def test_validate_strict_rejects_stacking_both_coverage_corrections():
    """Fail at startup, not inside the first radiation call."""
    from legoesm.driver.config import ExperimentConfig

    with pytest.raises((ValueError, SystemExit), match="mutually exclusive"):
        ExperimentConfig(
            cloud_partial_coverage_optics="two_column",
            cloud_vertical_overlap_optics="max_random").validate_strict()


def test_validate_strict_bounds_the_subcolumn_count():
    from legoesm.driver.config import ExperimentConfig

    for bad in (0, 65):
        with pytest.raises((ValueError, SystemExit), match="n_subcolumns"):
            ExperimentConfig(cloud_n_subcolumns=bad).validate_strict()
