"""Unit tests for the Stage-B (B2) carbon-parameter calibration trainer
(``scripts/run/train_carbon_params.py``).

Fast: NO full calibration.  Three checks -- the modular loss registry computes a
finite cover-weighted SOC MSE; ``to_overrides()`` -> ``apply_param_overrides``
round-trips a trainable leaf into ``CarbonConfig``; and ONE optimizer step on a
tiny ``--dry-run-synthetic`` world changes the loss finitely and writes a
well-formed ``tuned_carbon_parameters.json``.

Compute-node scale for the ``--quick`` step (JIT-compiles the coupled land+carbon
spin-up + its reverse-mode) -- run via the sbatch/srun wrapper, NOT the login
node.  ``JAX_ENABLE_X64=1``; forced CPU here.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import numpy.testing as npt
import pytest

from scripts.run import train_carbon_params as tcp


def test_cover_weighted_mse_matches_hand_value():
    """area_weighted_mse mapped onto the archetype axis == sum_a w_a e_a^2 / sum_a w_a."""
    import jax.numpy as jnp

    pred = jnp.asarray([1.0, 2.0, 3.0])
    target = jnp.asarray([1.5, 2.0, 5.0])
    weight = jnp.asarray([1.0, 3.0, 0.0])
    # sum w e^2 = 1*0.25 + 3*0 + 0*4 = 0.25 ; sum w = 4 -> 0.0625
    got = float(tcp.cover_weighted_mse(pred, target, weight))
    npt.assert_allclose(got, 0.0625, rtol=1e-12)


def test_loss_registry_sums_weighted_terms():
    """The modular registry sums w_k * cover_weighted_mse without touching the loop."""
    import jax.numpy as jnp

    target = jnp.asarray([1.5, 2.0, 5.0])
    cover = jnp.asarray([1.0, 3.0, 0.0])
    # extractor is identity on a fake "equilibrium" -> reuse the MSE math above.
    fake_eq = jnp.asarray([1.0, 2.0, 3.0])
    losses = {"soc": tcp.LossTerm(target=target, extractor=lambda x: x, weight=2.0)}
    total = float(tcp._total_loss(fake_eq, losses, cover))
    npt.assert_allclose(total, 2.0 * 0.0625, rtol=1e-12)
    assert np.isfinite(total)


def test_nan_target_is_dropped_by_zero_weight():
    """A NaN observed target with weight 0 must NOT poison the (finite) loss."""
    import jax.numpy as jnp

    pred = jnp.asarray([1.0, 2.0])
    target = jnp.asarray([1.5, 0.0])       # 2nd entry sanitised from NaN -> 0
    weight = jnp.asarray([2.0, 0.0])       # ... with zero cover weight
    got = float(tcp.cover_weighted_mse(pred, target, weight))
    npt.assert_allclose(got, 0.25, rtol=1e-12)   # 2*0.25 / 2
    assert np.isfinite(got)


def test_overrides_roundtrip_into_carbon_config():
    """to_overrides() -> apply_param_overrides splices a traced leaf into CarbonConfig,
    and the warm start equals the production defaults (sub-clamp precision)."""
    import jax.numpy as jnp
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.training.param_collector import apply_param_overrides

    params = tcp.build_carbon_trainables(som_only=True)
    names = {c.field for c in params.constraints}
    assert set(tcp.SOM_FIELDS) == names, names

    overrides = tcp._carbon_overrides(params)
    assert set(overrides) == set(tcp.SOM_FIELDS)

    cfg = apply_param_overrides(CarbonConfig(), overrides)
    for field in tcp.SOM_FIELDS:
        v = float(np.asarray(getattr(cfg, field)))
        assert np.isfinite(v)
    # Warm start recovers the production default for som_freeze_floor.
    npt.assert_allclose(
        float(np.asarray(getattr(cfg, "som_freeze_floor"))),
        CarbonConfig().som_freeze_floor, rtol=1e-6)


def test_som_only_filter_selects_exactly_seven_and_excludes_q10():
    """The fast-analytic-valid SOM set is the 7 SOM-pool-only fields; codex fix:
    ``Q10_het_exp`` is deliberately EXCLUDED (its fast-mode gradient would be
    partial -- see the comment on ``SOM_FIELDS``), even though it is a genuine
    tier-2 ``land.carbon`` trainable (it remains available via
    ``--slow-spinup-grad``)."""
    params = tcp.build_carbon_trainables(som_only=True)
    names = {c.field for c in params.constraints}
    assert len(params.constraints) == len(tcp.SOM_FIELDS) == 9
    assert "Q10_het_exp" not in tcp.SOM_FIELDS
    assert "Q10_het_exp" not in names
    # the two perennial-frost / anaerobic SOM protection knobs ARE fast-valid.
    assert "permafrost_protection_min" in names
    assert "permafrost_frozen_fraction_threshold" in names


def test_quick_dry_run_writes_wellformed_json(tmp_path):
    """One optimizer step on a 2-3-archetype synthetic world changes the loss
    finitely and writes a well-formed tuned_carbon_parameters.json + scorecard."""
    outdir = tmp_path / "carbon_calib"
    rc = tcp.main(["--quick", "--output", str(outdir)])
    assert rc == 0

    tuned = json.loads((outdir / "tuned_carbon_parameters.json").read_text())
    assert tuned["scheme_key"] == "land.carbon"
    assert tuned["note"].startswith("RECOMMENDED")
    params = tuned["tuned_carbon_parameters"]
    assert isinstance(params, dict) and len(params) >= 1
    for field, value in params.items():
        assert np.isfinite(value), (field, value)
        assert field in tcp.SOM_FIELDS
    # loss history present; the trainer records at least the initial loss.
    hist = tuned["loss_history"]
    assert len(hist) >= 1 and all(np.isfinite(x) for x in hist)
    assert tuned["training"]["som_only"] is True
    # Each parameter row carries default/tuned/bounds.
    for row in tuned["parameters"]:
        assert {"field", "production_default", "tuned", "lower", "upper"} <= set(row)
        assert row["lower"] <= row["tuned"] <= row["upper"]
        assert np.isfinite(row["tuned"])

    scorecard = json.loads((outdir / "scorecard.json").read_text())
    assert "per_pft_soc" in scorecard
    rmse = scorecard["cover_weighted_soc_rmse_kgC_m2"]
    assert np.isfinite(rmse["default"]) and np.isfinite(rmse["tuned"])
    # A single MUON line-search step must not INCREASE the loss.
    assert hist[-1] <= hist[0] + 1e-9
    # --quick uses the DEFAULT fast-analytic forward: the run records the forward
    # kind + the analytic-vs-spin-up match, and the closed form must be finite.
    training = tuned["training"]
    assert training["forward"] == "fast_analytic"
    match = training["analytic_vs_spinup_match"]
    assert match["n_archetypes"] >= 1
    assert np.isfinite(match["mean_abs_rel_err"]) and match["mean_abs_rel_err"] >= 0.0


def test_fast_analytic_is_default_and_slow_flag_flips_it():
    """--fast-analytic (default True) trains the closed form; --slow-spinup-grad
    selects the B1 grad-through-spin-up forward."""
    p = tcp.build_arg_parser()
    assert p.parse_args(["--dry-run-synthetic"]).fast_analytic is True
    assert p.parse_args(["--dry-run-synthetic", "--fast-analytic"]).fast_analytic is True
    assert p.parse_args(
        ["--dry-run-synthetic", "--slow-spinup-grad"]).fast_analytic is False


def test_fast_analytic_and_all_carbon_params_raises():
    """codex fix: --fast-analytic (default True) + --all-carbon-params must raise
    -- the fast closed form freezes every non-SOM tier-2 input (GPP/phenology/
    allocation) at the DEFAULT params in the one-time precompute, so those fields
    would get a stale/zero gradient, not a real calibration signal.
    --slow-spinup-grad --all-carbon-params (the correct grad-through-spin-up
    forward) is the valid way to train the full tier-2 set and must NOT raise."""
    with pytest.raises(SystemExit, match="not supported"):
        tcp.main(["--dry-run-synthetic", "--fast-analytic", "--all-carbon-params",
                  "--steps", "1"])
    # --fast-analytic defaults True, so omitting it still hits the same guard.
    with pytest.raises(SystemExit, match="not supported"):
        tcp.main(["--dry-run-synthetic", "--all-carbon-params", "--steps", "1"])
    # The slow-path + --all-carbon-params combo remains valid: _finalize_args
    # must NOT raise for it.
    args = tcp._finalize_args(tcp.build_arg_parser().parse_args(
        ["--dry-run-synthetic", "--slow-spinup-grad", "--all-carbon-params"]))
    assert args.fast_analytic is False
    assert args.all_carbon_params is True


def test_quick_slow_spinup_grad_path_still_runs(tmp_path):
    """The slow grad-through-spin-up forward remains available + monotone (one
    optimizer step on a tiny synthetic world)."""
    outdir = tmp_path / "carbon_calib_slow"
    rc = tcp.main(["--quick", "--slow-spinup-grad", "--output", str(outdir)])
    assert rc == 0
    tuned = json.loads((outdir / "tuned_carbon_parameters.json").read_text())
    assert tuned["training"]["forward"] == "slow_spinup_grad"
    hist = tuned["loss_history"]
    assert len(hist) >= 1 and all(np.isfinite(x) for x in hist)
    assert hist[-1] <= hist[0] + 1e-9


def test_chunk_bounds_partitions_exactly():
    """``_chunk_bounds`` tiles ``[0, n)`` with contiguous ``<=chunk`` blocks;
    ``chunk<=0`` or ``chunk>=n`` collapses to a single batch."""
    assert tcp._chunk_bounds(4, 2) == [(0, 2), (2, 4)]
    assert tcp._chunk_bounds(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert tcp._chunk_bounds(4, 0) == [(0, 4)]      # disabled -> single batch
    assert tcp._chunk_bounds(4, 9) == [(0, 4)]      # chunk>=n -> single batch
    assert tcp._chunk_bounds(4, 1) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    # Every archetype index is covered exactly once (no gap, no overlap).
    for n, c in [(7, 3), (10, 4), (1, 5), (16, 4)]:
        covered = [i for (s, e) in tcp._chunk_bounds(n, c) for i in range(s, e)]
        assert covered == list(range(n)), (n, c)


def test_weighted_sse_is_the_unnormalized_mse_numerator():
    """``_weighted_sse`` is the numerator of ``cover_weighted_mse``:
    ``_total_loss == _weighted_sse / sum(cover)``; a zero-cover chunk contributes a
    clean 0 (no 0/0 NaN), which is why the chunked accumulation uses it."""
    import jax.numpy as jnp

    fake_eq = jnp.asarray([1.0, 2.0, 3.0])
    target = jnp.asarray([1.5, 2.0, 5.0])
    cover = jnp.asarray([1.0, 3.0, 0.0])
    losses = {"soc": tcp.LossTerm(target=target, extractor=lambda x: x, weight=2.0)}
    # 2.0 * (1*0.25 + 3*0 + 0*4) = 0.5
    sse = float(tcp._weighted_sse(fake_eq, losses, cover))
    npt.assert_allclose(sse, 0.5, rtol=1e-12)
    # numerator / sum(cover) == the normalised registry loss.
    total = float(tcp._total_loss(fake_eq, losses, cover))
    npt.assert_allclose(sse / float(jnp.sum(cover)), total, rtol=1e-12)
    # an all-zero-cover chunk -> exactly 0, never NaN.
    assert float(tcp._weighted_sse(fake_eq, losses, jnp.zeros(3))) == 0.0


def test_grad_chunk_cli_and_quick_disables_it():
    """``--grad-chunk`` parses to the default; ``--quick`` forces the single-batch
    path (0) so a tiny world is never chunked."""
    p = tcp.build_arg_parser()
    assert p.parse_args(["--dry-run-synthetic"]).grad_chunk == tcp.DEFAULT_GRAD_CHUNK
    assert p.parse_args(["--dry-run-synthetic", "--grad-chunk", "6"]).grad_chunk == 6
    q = tcp._finalize_args(p.parse_args(["--quick", "--grad-chunk", "6"]))
    assert q.grad_chunk == 0


def test_compilation_cache_cli_flags_roundtrip(monkeypatch):
    """--compilation-cache-dir / --cache-min-compile-secs parse and default
    correctly: the threshold default is the small carbon floor (NOT the campaign's
    30 s), the dir defaults to $JAX_COMPILATION_CACHE_DIR (empty when unset), and
    both round-trip from explicit CLI values."""
    # default threshold is the carbon floor, dir empty when the env is unset
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    a = tcp.build_arg_parser().parse_args(["--dry-run-synthetic"])
    assert a.cache_min_compile_secs == tcp.DEFAULT_CACHE_MIN_COMPILE_SECS
    assert tcp.DEFAULT_CACHE_MIN_COMPILE_SECS < 30.0  # must beat the rrtmgp floor
    assert a.compilation_cache_dir == ""
    # env-driven default for the dir
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/shared/jaxcache")
    b = tcp.build_arg_parser().parse_args(["--dry-run-synthetic"])
    assert b.compilation_cache_dir == "/shared/jaxcache"
    # explicit values round-trip and override the env default
    c = tcp.build_arg_parser().parse_args(
        ["--dry-run-synthetic", "--compilation-cache-dir", "/x/y", "--cache-min-compile-secs", "7.5"])
    assert c.compilation_cache_dir == "/x/y"
    assert c.cache_min_compile_secs == 7.5


# ---------------------------------------------------------------------------
# Deterministic precompute RESULT cache (_load_or_precompute / cache key).
# Pure + fast: the expensive precompute is STUBBED with a counter, so these run
# with no model compile and prove the cache hit/miss/round-trip contract.
# ---------------------------------------------------------------------------
def _tiny_archetype_table(n_arch: int = 3):
    """Minimal real ArchetypeTable (fixed values -> a stable key)."""
    from legoesm.land.carbon.global_init import ArchetypeTable
    return ArchetypeTable(
        pft_id=np.arange(n_arch, dtype=np.int64),
        mat_k=np.linspace(280.0, 300.0, n_arch),
        map_yr=np.linspace(0.2, 2.0, n_arch),
        t_seasonal_amp_k=np.linspace(2.0, 20.0, n_arch),
        aridity=np.linspace(0.1, 0.9, n_arch),
        sw_mean_w=np.linspace(100.0, 300.0, n_arch),
        soil_class=np.array(["loam", "sand", "clay"][:n_arch], dtype=object),
    )


def _stub_precompute_result(n_arch: int = 3):
    """Fixed (FastAnalyticInputs, real_som) the stub returns (dt_days a float)."""
    from legoesm.land.carbon.fast_analytic import FastAnalyticInputs
    fai = FastAnalyticInputs(
        lit_to_som_annual=np.linspace(1.0, 3.0, n_arch),
        a_wood_annual=np.linspace(0.5, 1.5, n_arch),
        soil_T_traj=np.full((n_arch, 4), 285.0),
        soil_frozen_fraction=np.zeros(n_arch),
        precip=np.full(n_arch, 3.0e-5),
        dt_days=0.0833333333,
        npp_pos_annual=np.linspace(300.0, 900.0, n_arch),
        is_woody=np.array([1.0, 0.0, 1.0][:n_arch]),
        is_evergreen=np.array([1.0, 0.0, 0.0][:n_arch]),
        live_ref_C_fol=np.linspace(100.0, 300.0, n_arch),
        live_ref_C_root=np.linspace(200.0, 500.0, n_arch),
        live_ref_C_wood=np.linspace(1000.0, 5000.0, n_arch),
    )
    return fai, np.linspace(5000.0, 9000.0, n_arch)


def _counting_stub(fai, real_som, calls):
    def stub(table, *, n_spinup, n_verify, dt, n_layers, soil_depth):
        calls["n"] += 1
        return fai, real_som
    return stub


_SPIN = {"n_spinup": 20, "n_verify": 4, "dt": 7200.0, "n_layers": 8, "soil_depth": 3.0}


def test_load_or_precompute_hit_miss_and_roundtrip(tmp_path, monkeypatch):
    """Call #1 computes + writes the npz; call #2 (same table+spin) is a CACHE HIT
    (the wrapped precompute is NOT called again) reconstructing the SAME arrays
    EXACTLY (dtypes, dt_days as float, real_som shape); a changed spin and a bumped
    _PRECOMPUTE_CACHE_VERSION both MISS (recompute)."""
    calls = {"n": 0}
    fai, real_som = _stub_precompute_result()
    monkeypatch.setattr(
        "legoesm.land.carbon.global_init.precompute_fast_analytic_inputs",
        _counting_stub(fai, real_som, calls))

    table = _tiny_archetype_table()
    cache_dir = str(tmp_path / "pc")

    # call #1 -> MISS: computes and writes the npz
    p1, s1 = tcp._load_or_precompute(table, _SPIN, cache_dir=cache_dir, rebuild=False)
    assert calls["n"] == 1
    key = tcp._precompute_cache_key(table, _SPIN)
    assert os.path.exists(os.path.join(cache_dir, key + ".npz"))

    # call #2 -> CACHE HIT: stub NOT called again; arrays reconstruct exactly
    p2, s2 = tcp._load_or_precompute(table, _SPIN, cache_dir=cache_dir, rebuild=False)
    assert calls["n"] == 1  # unchanged -> served from disk
    for f in ("lit_to_som_annual", "a_wood_annual", "soil_T_traj", "precip",
              "npp_pos_annual", "is_woody", "is_evergreen", "live_ref_C_fol",
              "live_ref_C_root", "live_ref_C_wood"):
        a1, a2 = np.asarray(getattr(p1, f)), np.asarray(getattr(p2, f))
        npt.assert_array_equal(a1, a2)
        assert a1.dtype == a2.dtype, f
    npt.assert_array_equal(np.asarray(s1), np.asarray(s2))
    assert np.asarray(s2).shape == np.asarray(real_som).shape
    # dt_days round-trips as a Python float, bit-for-bit.
    assert isinstance(p2.dt_days, float)
    assert p2.dt_days == fai.dt_days

    # a changed spin param -> MISS (recompute)
    tcp._load_or_precompute(table, dict(_SPIN, n_spinup=40),
                            cache_dir=cache_dir, rebuild=False)
    assert calls["n"] == 2

    # bump the stale-physics guard -> MISS even with identical table+spin
    monkeypatch.setattr(tcp, "_PRECOMPUTE_CACHE_VERSION", "v2-test")
    tcp._load_or_precompute(table, _SPIN, cache_dir=cache_dir, rebuild=False)
    assert calls["n"] == 3


def test_load_or_precompute_rebuild_bypasses_cache(tmp_path, monkeypatch):
    """rebuild=True (--rebuild-precompute) recomputes + overwrites even on a hit."""
    calls = {"n": 0}
    fai, real_som = _stub_precompute_result()
    monkeypatch.setattr(
        "legoesm.land.carbon.global_init.precompute_fast_analytic_inputs",
        _counting_stub(fai, real_som, calls))
    table = _tiny_archetype_table()
    cache_dir = str(tmp_path / "pc")
    tcp._load_or_precompute(table, _SPIN, cache_dir=cache_dir, rebuild=False)  # writes
    assert calls["n"] == 1
    tcp._load_or_precompute(table, _SPIN, cache_dir=cache_dir, rebuild=True)   # forced
    assert calls["n"] == 2


def test_load_or_precompute_empty_dir_disables_cache(tmp_path, monkeypatch):
    """cache_dir='' always recomputes and writes NOTHING to disk."""
    calls = {"n": 0}
    fai, real_som = _stub_precompute_result()
    monkeypatch.setattr(
        "legoesm.land.carbon.global_init.precompute_fast_analytic_inputs",
        _counting_stub(fai, real_som, calls))
    table = _tiny_archetype_table()
    tcp._load_or_precompute(table, _SPIN, cache_dir="", rebuild=False)
    tcp._load_or_precompute(table, _SPIN, cache_dir="", rebuild=False)
    assert calls["n"] == 2  # never cached


def test_load_or_precompute_recovers_from_corrupt_npz(tmp_path, monkeypatch):
    """A corrupt/truncated npz at the cache path is dropped + recomputed (never a
    hard crash on every future run), then re-saved as a clean cache."""
    calls = {"n": 0}
    fai, real_som = _stub_precompute_result()
    monkeypatch.setattr(
        "legoesm.land.carbon.global_init.precompute_fast_analytic_inputs",
        _counting_stub(fai, real_som, calls))
    table = _tiny_archetype_table()
    cache_dir = str(tmp_path / "pc")
    os.makedirs(cache_dir, exist_ok=True)
    key = tcp._precompute_cache_key(table, _SPIN)
    with open(os.path.join(cache_dir, key + ".npz"), "wb") as fh:
        fh.write(b"not a real npz zip archive")  # garbage -> BadZipFile on load
    # must NOT raise: drops the bad file and recomputes
    _p, s = tcp._load_or_precompute(table, _SPIN, cache_dir=cache_dir, rebuild=False)
    assert calls["n"] == 1
    npt.assert_array_equal(np.asarray(s), np.asarray(real_som))
    # a GOOD file was re-saved atomically -> the next call is a clean CACHE HIT
    tcp._load_or_precompute(table, _SPIN, cache_dir=cache_dir, rebuild=False)
    assert calls["n"] == 1


def test_precompute_cache_key_deterministic_and_input_sensitive():
    """Same inputs -> same key (deterministic); any table/spin change -> a new key.

    (Guards collision-safety: distinct inputs must not alias to one digest.)"""
    t = _tiny_archetype_table()
    k = tcp._precompute_cache_key(t, _SPIN)
    assert k == tcp._precompute_cache_key(t, _SPIN)          # deterministic
    assert len(k) == 64                                       # sha256 hexdigest
    assert tcp._precompute_cache_key(t._replace(mat_k=t.mat_k + 1.0), _SPIN) != k
    assert tcp._precompute_cache_key(
        t._replace(soil_class=np.array(["sand", "sand", "clay"], dtype=object)),
        _SPIN) != k
    for f, v in [("n_spinup", 21), ("n_verify", 5), ("dt", 3600.0),
                 ("n_layers", 6), ("soil_depth", 2.0)]:
        assert tcp._precompute_cache_key(t, dict(_SPIN, **{f: v})) != k, f


def test_precompute_cache_cli_flags_roundtrip(monkeypatch):
    """--precompute-cache-dir defaults to $CARBON_PRECOMPUTE_CACHE_DIR (else the
    repo .cache path), round-trips an explicit value, and --rebuild-precompute is
    a store_true default-False; --quick disables the cache (hermetic smoke)."""
    a = tcp.build_arg_parser().parse_args(["--dry-run-synthetic"])
    # default cache dir is the module default (import-time: $CARBON_PRECOMPUTE_CACHE_DIR
    # or the repo .cache/carbon_precompute fallback -- ON by default, unlike the
    # compile cache which is off by default).
    assert a.precompute_cache_dir == tcp.DEFAULT_PRECOMPUTE_CACHE_DIR
    assert isinstance(a.precompute_cache_dir, str) and a.precompute_cache_dir
    assert a.rebuild_precompute is False
    # explicit dir + --rebuild-precompute round-trip
    b = tcp.build_arg_parser().parse_args(
        ["--dry-run-synthetic", "--precompute-cache-dir", "/x/pc",
         "--rebuild-precompute"])
    assert b.precompute_cache_dir == "/x/pc"
    assert b.rebuild_precompute is True
    # --quick makes the smoke hermetic: the result cache is disabled.
    q = tcp._finalize_args(tcp.build_arg_parser().parse_args(
        ["--quick", "--precompute-cache-dir", "/x/pc"]))
    assert q.precompute_cache_dir == ""


def test_chunked_grad_equals_single_batch_slow():
    """EXACTness gate: on a tiny 4-archetype table where the SINGLE-batch
    slow-spinup-grad works, the CHUNKED value-and-grad (chunk in {2,3} -- an even
    2+2 split and an uneven 3+1 split) equals the single-batch value-and-grad to
    rtol 1e-6 for EVERY SOM param -- the per-chunk UN-normalized weighted SSE grads
    sum and are divided by the GLOBAL cover-weight sum exactly once (the
    accumulation is exact).  The chunked forward-only loss (line-search path)
    matches too.  (``_chunk_bounds`` covers the partition arithmetic incl. single
    archetypes; ``_weighted_sse`` covers the zero-cover case -- both pure/cheap.)

    Compute-node scale (JIT-compiles the coupled land+carbon spin-up + its
    reverse-mode over up to 4 archetypes); run via the sbatch/srun wrapper, NOT the
    login node.  ``JAX_ENABLE_X64=1``; CPU here.  ``jax.clear_caches()`` is called
    between the single-batch and each chunked build so the retained coupled-model
    executables do not accumulate and exhaust the XLA/LLVM compile memory."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    # Drop any coupled-model executables compiled by earlier tests in this process
    # so their retained code does not stack with this test's compiles.
    jax.clear_caches()

    # A tiny REAL synthetic target, reusing the trainer's own build_target
    # assembly, sliced to exactly 4 archetypes so the single batch is crash-free.
    argv = ["--dry-run-synthetic", "--slow-spinup-grad",
            "--max-archetypes", "0", "--steps", "1"]
    args = tcp._finalize_args(tcp.build_arg_parser().parse_args(argv))
    full = tcp.build_target(args)
    n_full = int(np.asarray(full["table"].pft_id).shape[0])
    n = 4
    assert n_full >= n, f"synthetic built only {n_full} archetypes (< {n})"
    idx = np.arange(n)
    table = tcp._slice_table(full["table"], idx)
    cover = jnp.asarray(np.asarray(full["cover_weight"])[idx], dtype=jnp.float64)
    observed = np.asarray(full["observed_soc"])[idx]
    losses = {"soc": tcp.LossTerm(
        target=jnp.asarray(observed, dtype=jnp.float64),
        extractor=tcp._soc_extractor, weight=1.0)}
    bundle = {"table": table, "losses": losses, "cover_weight": cover}
    spin = {"n_spinup": 4, "n_verify": 2, "dt": 7200.0, "n_layers": 6,
            "soil_depth": 2.0}
    params = tcp.build_carbon_trainables(som_only=True)

    # SINGLE-batch reference: the existing full-table slow loss + its grad.  Cache
    # the grad as plain numpy so the compiled executable can be released before the
    # chunked builds compile.
    loss_fn = tcp.make_loss_fn(table=table, losses=losses, cover_weight=cover, **spin)
    ref_loss, ref_grad = eqx.filter_value_and_grad(loss_fn)(params)
    ref_loss = float(ref_loss)
    assert np.isfinite(ref_loss)
    ref_grad_vals = {c.name: np.asarray(ref_grad.raw_values[c.name])
                     for c in params.constraints}
    del loss_fn, ref_grad
    jax.clear_caches()

    for chunk in (2, 3):
        vg, le = tcp._make_chunked_slow(bundle, spin, chunk=chunk)
        c_loss, c_grad = vg(params)
        npt.assert_allclose(float(c_loss), ref_loss, rtol=1e-6,
                            err_msg=f"chunk={chunk} loss")
        # The forward-only line-search loss must match the value-and-grad value.
        npt.assert_allclose(float(le(params)), ref_loss, rtol=1e-6,
                            err_msg=f"chunk={chunk} loss_eval")
        for c in params.constraints:
            npt.assert_allclose(
                np.asarray(c_grad.raw_values[c.name]),
                ref_grad_vals[c.name],
                rtol=1e-6, atol=1e-12,
                err_msg=f"chunk={chunk} grad[{c.field}]")
        del vg, le, c_grad
        jax.clear_caches()


def test_resolve_om_to_oc_is_preset_aware():
    """Default om_to_oc: 0.58 (van Bemmelen) for CLM5 organic MATTER, 1.0 for the
    HWSD-carbon legoesm_surfdata; an explicit --om-to-oc overrides either."""
    import argparse

    clm5 = argparse.Namespace(om_to_oc=None, surfdata_preset="clm5_surfdata")
    lego = argparse.Namespace(om_to_oc=None, surfdata_preset="legoesm_surfdata")
    explicit = argparse.Namespace(om_to_oc=0.7, surfdata_preset="clm5_surfdata")
    assert tcp._resolve_om_to_oc(clm5) == tcp._VAN_BEMMELEN_OC_PER_OM == 0.58
    assert tcp._resolve_om_to_oc(lego) == 1.0
    assert tcp._resolve_om_to_oc(explicit) == 0.7


def test_input_mode_required():
    """No input mode -> a clear SystemExit (never a silent empty run)."""
    with pytest.raises(SystemExit):
        tcp.main(["--steps", "1"])


# ---------------------------------------------------------------------------
# SIF observation stream (--with-sif)
# ---------------------------------------------------------------------------
def test_total_loss_sums_soc_and_sif_from_bundle():
    """The registry sums soc + sif from ONE predictions bundle, without touching the
    loop: each extractor pulls its own key (``preds["soc"]`` / ``preds["sif"]``)."""
    import jax.numpy as jnp

    preds = {"soc": jnp.asarray([1.0, 2.0]), "sif": jnp.asarray([0.5, 1.5])}
    cover = jnp.asarray([1.0, 1.0])
    losses = {
        "soc": tcp.LossTerm(target=jnp.asarray([1.5, 2.0]),
                            extractor=tcp._fast_soc_extractor, weight=1.0),
        "sif": tcp.LossTerm(target=jnp.asarray([0.0, 1.0]),
                            extractor=tcp._sif_extractor, weight=2.0),
    }
    total = float(tcp._total_loss(preds, losses, cover))
    # soc MSE = mean(0.25, 0) = 0.125 ; sif MSE = mean(0.25, 0.25) = 0.25
    # total = 1.0*0.125 + 2.0*0.25 = 0.625
    npt.assert_allclose(total, 0.125 + 2.0 * 0.25, rtol=1e-12)
    assert np.isfinite(total)


def test_per_term_mask_isolates_missing_streams():
    """A per-term finite mask lets one stream's missing archetype NOT discard another
    stream's signal, over the SINGLE global cover denominator; the masked archetype's
    (bogus) target is ignored, and _total_loss == _weighted_sse / sum(cover) still holds."""
    import jax.numpy as jnp

    cover = jnp.asarray([1.0, 1.0])
    preds = {"soc": jnp.asarray([1.0, 2.0]), "sif": jnp.asarray([0.5, 1.5])}
    losses = {
        # soc present at both archetypes.
        "soc": tcp.LossTerm(target=jnp.asarray([1.5, 2.0]),
                            extractor=tcp._fast_soc_extractor, weight=1.0,
                            mask=jnp.asarray([1.0, 1.0])),
        # sif MISSING at archetype 0 (mask 0 -> its bogus 9.9 target is ignored), present at 1.
        "sif": tcp.LossTerm(target=jnp.asarray([9.9, 1.0]),
                            extractor=tcp._sif_extractor, weight=1.0,
                            mask=jnp.asarray([0.0, 1.0])),
    }
    # soc = sum(cover*[(1-1.5)^2, 0])/2 = 0.125 ; sif = sum(cover*mask*[(.5-9.9)^2,(1.5-1)^2])/2
    #     = (0 + 1*0.25)/2 = 0.125 ; total = 0.25 (the masked-out arch0 sif is ignored).
    total = float(tcp._total_loss(preds, losses, cover))
    npt.assert_allclose(total, 0.25, rtol=1e-12)
    sse = float(tcp._weighted_sse(preds, losses, cover))
    npt.assert_allclose(sse / float(jnp.sum(cover)), total, rtol=1e-12)


def test_build_carbon_trainables_with_sif_includes_fluorescence():
    """--with-sif adds the tier-1/2 SIFConfig fluorescence params ALONGSIDE the SOM set;
    the SOM-only filter never drops them.  sif OFF (default) -> no SIF params."""
    params = tcp.build_carbon_trainables(som_only=True, with_sif=True)
    names = {c.name for c in params.constraints}
    assert {f"{tcp.CARBON_SCHEME_KEY}.{f}" for f in tcp.SOM_FIELDS} <= names
    sif = {n for n in names if n.startswith(tcp.SIF_SCHEME_KEY + ".")}
    assert {f"{tcp.SIF_SCHEME_KEY}.{f}" for f in
            ("kn0", "max_electron_yield", "escape_probability", "kf", "kd", "kp")} <= sif

    off = {c.name for c in tcp.build_carbon_trainables(som_only=True).constraints}
    assert not any(n.startswith(tcp.SIF_SCHEME_KEY + ".") for n in off)


def test_sif_overrides_selects_only_sif_scheme():
    """``_sif_overrides`` returns the SIF slice; ``_carbon_overrides`` the carbon slice
    (disjoint, both traceable into their own config)."""
    params = tcp.build_carbon_trainables(som_only=True, with_sif=True)
    sif_ov = tcp._sif_overrides(params)
    car_ov = tcp._carbon_overrides(params)
    assert set(sif_ov) and set(car_ov)
    assert set(sif_ov).isdisjoint(car_ov)
    assert set(car_ov) == set(tcp.SOM_FIELDS)


def test_sif_cli_flags_roundtrip():
    p = tcp.build_arg_parser()
    a = p.parse_args(["--dry-run-synthetic"])
    assert a.with_sif is False and a.sif_obs == "" and a.sif_weight == tcp.DEFAULT_SIF_WEIGHT
    b = p.parse_args(["--dry-run-synthetic", "--with-sif", "--sif-weight", "0.3",
                      "--sif-obs", "x.nc", "--sif-var", "SIF_740"])
    assert b.with_sif is True and b.sif_weight == 0.3
    assert b.sif_obs == "x.nc" and b.sif_var == "SIF_740"


def test_with_sif_real_path_requires_sif_obs():
    """--with-sif on the real path without --sif-obs -> a clear SystemExit; the dry-run
    fabricates the target instead (no error)."""
    with pytest.raises(SystemExit, match="requires --sif-obs"):
        tcp._finalize_args(tcp.build_arg_parser().parse_args(
            ["--rebuild", "--surf-path", "x.nc", "--with-sif"]))
    args = tcp._finalize_args(tcp.build_arg_parser().parse_args(
        ["--dry-run-synthetic", "--with-sif"]))
    assert args.with_sif is True


def test_with_sif_all_gap_sif_rejected(monkeypatch):
    """--with-sif but an all-missing (NaN) SIF product -> a clear SystemExit at build,
    never a dead SIF term reporting a fake 0 RMSE."""
    import legoesm.land.carbon.sif_observations as sifobs

    def _all_nan(table):
        return np.full(np.asarray(table.pft_id).shape[0], np.nan)

    monkeypatch.setattr(sifobs, "synthetic_observed_sif", _all_nan)
    args = tcp._finalize_args(tcp.build_arg_parser().parse_args(["--quick", "--with-sif"]))
    with pytest.raises(SystemExit, match="no archetype has a finite observed SIF"):
        tcp.build_target(args)


def test_quick_with_sif_sums_registry_and_writes_json(tmp_path):
    """One optimizer step on a tiny synthetic world with --with-sif: the registry sums
    soc + sif, the 1-step line-search never increases the summed loss, and the tuned JSON
    + scorecard carry the trained SIF params + a SIF RMSE (fast-analytic default mode)."""
    outdir = tmp_path / "carbon_sif"
    rc = tcp.main(["--quick", "--with-sif", "--output", str(outdir)])
    assert rc == 0

    tuned = json.loads((outdir / "tuned_carbon_parameters.json").read_text())
    training = tuned["training"]
    assert training["with_sif"] is True
    assert set(training["loss_terms"]) == {"soc", "sif"}
    assert training["sif_weight"] == tcp.DEFAULT_SIF_WEIGHT
    # At least one SIF fluorescence param survived preflight + is finite/in-bounds.
    sif_rows = [r for r in tuned["parameters"] if r["scheme_key"] == tcp.SIF_SCHEME_KEY]
    assert sif_rows, "no SIF params in the tuned JSON"
    for r in sif_rows:
        assert np.isfinite(r["tuned"]) and r["lower"] <= r["tuned"] <= r["upper"]
    hist = tuned["loss_history"]
    assert len(hist) >= 1 and all(np.isfinite(x) for x in hist)
    assert hist[-1] <= hist[0] + 1e-9      # summed soc+sif line-search is monotone

    scorecard = json.loads((outdir / "scorecard.json").read_text())
    sif_rmse = scorecard["cover_weighted_sif_rmse_umol_m2_s"]
    assert np.isfinite(sif_rmse["default"]) and np.isfinite(sif_rmse["tuned"])


# ---------------------------------------------------------------------------
# Biomass + LAI observation streams (--with-biomass / --with-lai)
# ---------------------------------------------------------------------------
def test_total_loss_sums_soc_biomass_lai_from_bundle():
    """The registry sums soc + biomass + lai from ONE predictions bundle without touching
    the loop: each extractor pulls its own key (preds["soc"]/["biomass"]/["lai"])."""
    import jax.numpy as jnp

    preds = {"soc": jnp.asarray([1.0, 2.0]), "biomass": jnp.asarray([3.0, 5.0]),
             "lai": jnp.asarray([1.0, 2.0])}
    cover = jnp.asarray([1.0, 1.0])
    losses = {
        "soc": tcp.LossTerm(target=jnp.asarray([1.5, 2.0]),
                            extractor=tcp._fast_soc_extractor, weight=1.0),
        "biomass": tcp.LossTerm(target=jnp.asarray([3.0, 4.0]),
                                extractor=tcp._biomass_extractor, weight=2.0),
        "lai": tcp.LossTerm(target=jnp.asarray([0.0, 2.0]),
                            extractor=tcp._lai_extractor, weight=0.5),
    }
    total = float(tcp._total_loss(preds, losses, cover))
    # soc MSE = mean(.25,0)=.125 ; biomass MSE = mean(0,1)=.5 ; lai MSE = mean(1,0)=.5
    # total = 1*.125 + 2*.5 + 0.5*.5 = 1.375
    npt.assert_allclose(total, 0.125 + 2.0 * 0.5 + 0.5 * 0.5, rtol=1e-12)
    assert np.isfinite(total)


def test_build_carbon_trainables_with_live_pool_streams_includes_fields():
    """--with-biomass / --with-lai add the seven live-pool allocation/residence/LCMA fields
    ALONGSIDE the SOM set; OFF (default) -> none of them."""
    for kw in ({"with_biomass": True}, {"with_lai": True}):
        params = tcp.build_carbon_trainables(som_only=True, **kw)
        names = {c.field for c in params.constraints}
        assert set(tcp.SOM_FIELDS) <= names
        assert set(tcp.LIVE_POOL_FIELDS) <= names, (kw, names)
    off = {c.field for c in tcp.build_carbon_trainables(som_only=True).constraints}
    assert not (set(tcp.LIVE_POOL_FIELDS) & off)   # no live-pool fields when both off
    assert set(off) == set(tcp.SOM_FIELDS)


def test_make_live_pool_forward_matches_module_forward():
    """The trainer's _make_live_pool_forward(precomputed)(carbon_overrides) equals the
    module compute_biomass_lai for the applied CarbonConfig (overrides traced INSIDE)."""
    import jax.numpy as jnp
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.live_pool_forward import compute_biomass_lai
    from legoesm.training.param_collector import apply_param_overrides

    fai, _real = _stub_precompute_result(n_arch=3)
    live_fwd = tcp._make_live_pool_forward(fai)
    params = tcp.build_carbon_trainables(som_only=True, with_biomass=True, with_lai=True)
    ov = tcp._carbon_overrides(params)
    b1, l1 = live_fwd(ov)
    cfg = apply_param_overrides(CarbonConfig(scheme="differland"), ov)
    b2, l2 = compute_biomass_lai(np.asarray(fai.npp_pos_annual),
                                 np.asarray(fai.is_woody),
                                 np.asarray(fai.is_evergreen), cfg)
    npt.assert_allclose(np.asarray(b1), np.asarray(b2), rtol=1e-10)
    npt.assert_allclose(np.asarray(l1), np.asarray(l2), rtol=1e-10)


def test_live_pool_cli_flags_roundtrip():
    p = tcp.build_arg_parser()
    a = p.parse_args(["--dry-run-synthetic"])
    assert a.with_biomass is False and a.with_lai is False
    assert a.biomass_weight == tcp.DEFAULT_BIOMASS_WEIGHT
    assert a.lai_weight == tcp.DEFAULT_LAI_WEIGHT
    assert a.biomass_obs == "" and a.lai_obs == "" and a.lai_var == "" and a.biomass_var == ""
    b = p.parse_args(["--dry-run-synthetic", "--with-biomass", "--biomass-weight", "0.4",
                      "--biomass-obs", "b.nc", "--biomass-var", "AGB",
                      "--with-lai", "--lai-weight", "0.7", "--lai-obs", "s.nc",
                      "--lai-var", "MONTHLY_LAI"])
    assert b.with_biomass is True and b.biomass_weight == 0.4 and b.biomass_obs == "b.nc"
    assert b.biomass_var == "AGB"
    assert b.with_lai is True and b.lai_weight == 0.7 and b.lai_obs == "s.nc"
    assert b.lai_var == "MONTHLY_LAI"


def test_live_pool_streams_require_fast_analytic():
    """--with-biomass / --with-lai need the FAST NPP precompute; --slow-spinup-grad -> a
    clear SystemExit (never a slow-path KeyError on the unpopulated biomass/lai pred)."""
    with pytest.raises(SystemExit, match="require --fast-analytic"):
        tcp._finalize_args(tcp.build_arg_parser().parse_args(
            ["--dry-run-synthetic", "--slow-spinup-grad", "--with-biomass"]))
    with pytest.raises(SystemExit, match="require --fast-analytic"):
        tcp._finalize_args(tcp.build_arg_parser().parse_args(
            ["--dry-run-synthetic", "--slow-spinup-grad", "--with-lai"]))


def test_with_biomass_real_path_requires_biomass_obs():
    """--with-biomass on the real path without --biomass-obs -> a clear SystemExit; the
    dry-run fabricates the target instead."""
    with pytest.raises(SystemExit, match="requires --biomass-obs"):
        tcp._finalize_args(tcp.build_arg_parser().parse_args(
            ["--rebuild", "--surf-path", "x.nc", "--with-biomass"]))
    args = tcp._finalize_args(tcp.build_arg_parser().parse_args(
        ["--dry-run-synthetic", "--with-biomass"]))
    assert args.with_biomass is True


def test_quick_with_lai_and_biomass_sums_registry_and_writes_json(tmp_path):
    """One optimizer step on a tiny synthetic world with --with-lai --with-biomass: the
    registry sums soc + biomass + lai, the line-search is monotone, and the tuned JSON +
    scorecard carry the trained live-pool params + biomass/LAI RMSEs (fast default mode)."""
    outdir = tmp_path / "carbon_live"
    rc = tcp.main(["--quick", "--with-lai", "--with-biomass", "--output", str(outdir)])
    assert rc == 0

    tuned = json.loads((outdir / "tuned_carbon_parameters.json").read_text())
    training = tuned["training"]
    assert training["with_biomass"] is True and training["with_lai"] is True
    assert set(training["loss_terms"]) == {"soc", "biomass", "lai"}
    # At least one live-pool allocation/residence/LCMA param survived preflight, finite,
    # in-bounds.
    live_fields = set(tcp.LIVE_POOL_FIELDS)
    live_rows = [r for r in tuned["parameters"] if r["field"] in live_fields]
    assert live_rows, "no live-pool params in the tuned JSON"
    for r in live_rows:
        assert np.isfinite(r["tuned"]) and r["lower"] <= r["tuned"] <= r["upper"]
    hist = tuned["loss_history"]
    assert len(hist) >= 1 and all(np.isfinite(x) for x in hist)
    assert hist[-1] <= hist[0] + 1e-9

    scorecard = json.loads((outdir / "scorecard.json").read_text())
    for key in ("cover_weighted_biomass_rmse_kgC_m2", "cover_weighted_lai_rmse_m2_m2"):
        assert key in scorecard, key
        assert np.isfinite(scorecard[key]["default"]) and np.isfinite(scorecard[key]["tuned"])
    # The live-pool closed-form-vs-spin-up fidelity is recorded in the tuned-JSON meta.
    match = training["analytic_vs_spinup_match"]
    assert np.isfinite(match["biomass_cover_weighted_rel_err"])
    assert np.isfinite(match["lai_cover_weighted_rel_err"])


# ---------------------------------------------------------------------------
# Leaf carbon-isotope discrimination stream (--with-d13c)
# ---------------------------------------------------------------------------
def test_total_loss_sums_soc_and_d13c_from_bundle():
    """The registry sums soc + d13c from ONE predictions bundle without touching the loop:
    each extractor pulls its own key (``preds["soc"]`` / ``preds["d13c"]``)."""
    import jax.numpy as jnp

    preds = {"soc": jnp.asarray([1.0, 2.0]), "d13c": jnp.asarray([-27.0, -29.0])}
    cover = jnp.asarray([1.0, 1.0])
    losses = {
        "soc": tcp.LossTerm(target=jnp.asarray([1.5, 2.0]),
                            extractor=tcp._fast_soc_extractor, weight=1.0),
        "d13c": tcp.LossTerm(target=jnp.asarray([-28.0, -29.0]),
                             extractor=tcp._d13c_extractor, weight=0.5),
    }
    total = float(tcp._total_loss(preds, losses, cover))
    # soc MSE = mean(.25, 0) = .125 ; d13c MSE = mean(1, 0) = 0.5
    # total = 1*.125 + 0.5*0.5 = 0.375
    npt.assert_allclose(total, 0.125 + 0.5 * 0.5, rtol=1e-12)
    assert np.isfinite(total)


def test_build_carbon_trainables_with_d13c_includes_stomata_wue_and_c4_leakiness():
    """--with-d13c adds the stomata WUE subset (g0/g1_bb; the C3 lever) AND the C4 leakiness phi
    ALONGSIDE the SOM set -- never the per-archetype-incompatible Vc_max25 / other stomata
    kinetics, and never the FIXED C4 Ci/Ca setpoint; OFF -> neither scheme."""
    params = tcp.build_carbon_trainables(som_only=True, with_d13c=True)
    names = {c.name for c in params.constraints}
    assert {f"{tcp.CARBON_SCHEME_KEY}.{f}" for f in tcp.SOM_FIELDS} <= names
    stomata = {n for n in names if n.startswith(tcp.STOMATA_SCHEME_KEY + ".")}
    assert stomata == {f"{tcp.STOMATA_SCHEME_KEY}.{f}" for f in tcp.STOMATA_WUE_FIELDS}
    # The full stomata set (Vc_max25 / J_max25 / kinetics) is NOT pulled in.
    assert f"{tcp.STOMATA_SCHEME_KEY}.Vc_max25" not in stomata
    # The C4 leakiness phi IS pulled in; the fixed Ci/Ca setpoint is spec-excluded (absent).
    d13c = {n for n in names if n.startswith(tcp.D13C_SCHEME_KEY + ".")}
    assert d13c == {f"{tcp.D13C_SCHEME_KEY}.{f}" for f in tcp.D13C_LEAKINESS_FIELDS}
    assert f"{tcp.D13C_SCHEME_KEY}.ci_ca_c4" not in d13c

    off = {c.name for c in tcp.build_carbon_trainables(som_only=True).constraints}
    assert not any(n.startswith(tcp.STOMATA_SCHEME_KEY + ".") for n in off)
    assert not any(n.startswith(tcp.D13C_SCHEME_KEY + ".") for n in off)


def test_stomata_and_d13c_overrides_select_only_their_scheme():
    """``_stomata_overrides`` returns the C3 stomata WUE slice; ``_d13c_config_overrides`` the
    C4 leakiness slice; ``_carbon_overrides`` the carbon slice (disjoint, each traceable into
    its own config)."""
    params = tcp.build_carbon_trainables(som_only=True, with_d13c=True)
    st_ov = tcp._stomata_overrides(params)
    d13c_ov = tcp._d13c_config_overrides(params)
    car_ov = tcp._carbon_overrides(params)
    assert set(st_ov) == set(tcp.STOMATA_WUE_FIELDS)
    assert set(d13c_ov) == set(tcp.D13C_LEAKINESS_FIELDS)
    assert set(car_ov) == set(tcp.SOM_FIELDS)
    assert set(st_ov).isdisjoint(car_ov)
    assert set(d13c_ov).isdisjoint(st_ov) and set(d13c_ov).isdisjoint(car_ov)


def test_d13c_cli_flags_roundtrip():
    p = tcp.build_arg_parser()
    a = p.parse_args(["--dry-run-synthetic"])
    assert a.with_d13c is False and a.d13c_obs == ""
    assert a.d13c_weight == tcp.DEFAULT_D13C_WEIGHT and a.d13c_var == ""
    b = p.parse_args(["--dry-run-synthetic", "--with-d13c", "--d13c-weight", "0.4",
                      "--d13c-obs", "x.nc", "--d13c-var", "delta13C"])
    assert b.with_d13c is True and b.d13c_weight == 0.4
    assert b.d13c_obs == "x.nc" and b.d13c_var == "delta13C"


def test_with_d13c_real_path_requires_d13c_obs():
    """--with-d13c on the real path without --d13c-obs -> a clear SystemExit; the dry-run
    fabricates the target instead (no error)."""
    with pytest.raises(SystemExit, match="requires --d13c-obs"):
        tcp._finalize_args(tcp.build_arg_parser().parse_args(
            ["--rebuild", "--surf-path", "x.nc", "--with-d13c"]))
    args = tcp._finalize_args(tcp.build_arg_parser().parse_args(
        ["--dry-run-synthetic", "--with-d13c"]))
    assert args.with_d13c is True


def test_build_observed_d13c_keeps_c4_archetypes():
    """C4 archetypes are NO LONGER NaN-masked: the model now has a FAITHFUL C4 Farquhar-Cerling
    discrimination, so a C4 archetype WITH an observation CONTRIBUTES (finite target).  The
    synthetic C4 target sits in the C4 band (distinctly less negative than the C3 targets)."""
    from legoesm.land.carbon.global_init import ArchetypeTable
    from legoesm.land.surface_params import is_c4_pft_id

    args = tcp._finalize_args(tcp.build_arg_parser().parse_args(
        ["--dry-run-synthetic", "--with-d13c"]))
    # A table mixing C3 trees/grass with the two C4 PFTs (14 c4_grass, 16 crop_c4).
    pft = np.array([4, 13, 14, 16])
    n = pft.shape[0]
    table = ArchetypeTable(
        pft_id=pft, mat_k=np.array([298.0, 288.0, 300.0, 299.0]),
        map_yr=np.array([1200.0] * n), t_seasonal_amp_k=np.array([3.0] * n),
        aridity=np.array([1.0] * n), sw_mean_w=np.array([220.0] * n),
        soil_class=np.array(["loam"] * n, dtype=object))
    obs = tcp._build_observed_d13c(args, table, None, None, None, n)
    assert np.all(np.isfinite(obs))               # NO archetype is NaN-masked now
    is_c4 = is_c4_pft_id(pft)
    # C4 targets are C4-banded, C3 targets C3-banded, cleanly separated -> C4 contributes
    assert np.all(obs[is_c4] > -16.0) and np.all(obs[is_c4] < -10.0), obs
    assert np.all(obs[~is_c4] < -20.0), obs
    assert obs[is_c4].min() > obs[~is_c4].max()


def test_with_d13c_all_gap_d13c_rejected(monkeypatch):
    """--with-d13c but an all-missing (NaN) delta13C target -> a clear SystemExit at build,
    never a dead d13c term reporting a fake 0 RMSE."""
    import legoesm.land.carbon.d13c_observations as d13cobs

    def _all_nan(table):
        return np.full(np.asarray(table.pft_id).shape[0], np.nan)

    monkeypatch.setattr(d13cobs, "synthetic_observed_d13c", _all_nan)
    args = tcp._finalize_args(tcp.build_arg_parser().parse_args(["--quick", "--with-d13c"]))
    with pytest.raises(SystemExit, match="no archetype has a finite observed D13C"):
        tcp.build_target(args)


def test_quick_with_d13c_sums_registry_and_writes_json(tmp_path):
    """One optimizer step on a tiny synthetic world with --with-d13c: the registry sums
    soc + d13c, the 1-step line-search never increases the summed loss, and the tuned JSON +
    scorecard carry the trained stomata WUE params + a delta13C RMSE (fast default mode)."""
    outdir = tmp_path / "carbon_d13c"
    rc = tcp.main(["--quick", "--with-d13c", "--output", str(outdir)])
    assert rc == 0

    tuned = json.loads((outdir / "tuned_carbon_parameters.json").read_text())
    training = tuned["training"]
    assert training["with_d13c"] is True
    assert set(training["loss_terms"]) == {"soc", "d13c"}
    assert training["d13c_weight"] == tcp.DEFAULT_D13C_WEIGHT
    # At least one stomata WUE param survived preflight + is finite/in-bounds.
    st_rows = [r for r in tuned["parameters"] if r["scheme_key"] == tcp.STOMATA_SCHEME_KEY]
    assert st_rows, "no stomata WUE params in the tuned JSON"
    assert {r["field"] for r in st_rows} <= set(tcp.STOMATA_WUE_FIELDS)
    for r in st_rows:
        assert np.isfinite(r["tuned"]) and r["lower"] <= r["tuned"] <= r["upper"]
    hist = tuned["loss_history"]
    assert len(hist) >= 1 and all(np.isfinite(x) for x in hist)
    assert hist[-1] <= hist[0] + 1e-9      # summed soc+d13c line-search is monotone

    scorecard = json.loads((outdir / "scorecard.json").read_text())
    d13c_rmse = scorecard["cover_weighted_d13c_rmse_permil"]
    assert np.isfinite(d13c_rmse["default"]) and np.isfinite(d13c_rmse["tuned"])
