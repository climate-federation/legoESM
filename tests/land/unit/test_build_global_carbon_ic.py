"""Unit tests for the global carbon-IC build's persistent caches
(``scripts/data/build_global_carbon_ic.py``).

Fast + hermetic: the EXPENSIVE ``equilibrate_archetypes`` (per-group coupled-land
compile + spin-up) is STUBBED with a counter, so these run with NO model compile
and prove the cache hit/miss/round-trip/self-heal contract plus the CLI flag
plumbing.  Mirrors the Stage-B trainer's cache tests
(``tests/land/unit/test_train_carbon_params.py``), whose policy this build shares.

Compute-node scale only in that importing the driver / building a ``CarbonState``
touches JAX; run via the sbatch/srun wrapper, NOT the login node.
``JAX_ENABLE_X64=1``; forced CPU here.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import numpy.testing as npt
import pytest

from scripts.data import build_global_carbon_ic as bgc


# ---------------------------------------------------------------------------
# Part A -- compilation-cache CLI flags
# ---------------------------------------------------------------------------
def test_compilation_cache_cli_flags_roundtrip(monkeypatch):
    """--compilation-cache-dir / --cache-min-compile-secs parse and default
    correctly: the dir defaults to $JAX_COMPILATION_CACHE_DIR (empty when unset),
    the threshold default is the small carbon floor, and both round-trip from
    explicit CLI values (an explicit dir overrides the env default)."""
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    a = bgc.build_arg_parser().parse_args(["--dry-run-synthetic"])
    assert a.compilation_cache_dir == ""
    assert a.cache_min_compile_secs == bgc._CACHE_MIN_COMPILE_SECS_DEFAULT
    # env-driven default for the dir
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/shared/jaxcache")
    b = bgc.build_arg_parser().parse_args(["--dry-run-synthetic"])
    assert b.compilation_cache_dir == "/shared/jaxcache"
    # explicit values round-trip and override the env default
    c = bgc.build_arg_parser().parse_args(
        ["--dry-run-synthetic", "--compilation-cache-dir", "/x/y",
         "--cache-min-compile-secs", "7.5"])
    assert c.compilation_cache_dir == "/x/y"
    assert c.cache_min_compile_secs == 7.5


def test_equilibrium_cache_cli_flags_roundtrip(monkeypatch):
    """--equilibrium-cache-dir defaults to $CARBON_EQUILIBRIUM_CACHE_DIR (empty
    when unset -> disabled, unchanged behavior), and --rebuild-equilibrium is a
    store_true default-False; both round-trip from explicit CLI values."""
    monkeypatch.delenv("CARBON_EQUILIBRIUM_CACHE_DIR", raising=False)
    a = bgc.build_arg_parser().parse_args(["--dry-run-synthetic"])
    assert a.equilibrium_cache_dir == ""      # empty -> disabled by default
    assert a.rebuild_equilibrium is False
    monkeypatch.setenv("CARBON_EQUILIBRIUM_CACHE_DIR", "/shared/eqcache")
    b = bgc.build_arg_parser().parse_args(["--dry-run-synthetic"])
    assert b.equilibrium_cache_dir == "/shared/eqcache"
    c = bgc.build_arg_parser().parse_args(
        ["--dry-run-synthetic", "--equilibrium-cache-dir", "/x/eq",
         "--rebuild-equilibrium"])
    assert c.equilibrium_cache_dir == "/x/eq"
    assert c.rebuild_equilibrium is True


# ---------------------------------------------------------------------------
# Part B -- deterministic equilibrium RESULT cache
# (_load_or_equilibrate / _equilibrium_cache_key). The expensive
# equilibrate_archetypes is STUBBED with a counter -> no model compile.
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


def _stub_equilibrium_result(n_arch: int = 3):
    """Fixed ``(CarbonState eq, qc dict)`` the stub returns.

    ``eq`` is a real CarbonState of per-archetype pools (jnp arrays, matching the
    compute's return type) and ``qc`` the exact five-key QC bundle
    ``equilibrate_archetypes`` produces, each ``(n_arch,)``.
    """
    import jax.numpy as jnp
    from legoesm.land.carbon.config import CarbonState

    eq = CarbonState(**{
        f: jnp.asarray(np.linspace(1.0 + i, 3.0 + i, n_arch))
        for i, f in enumerate(CarbonState._fields)
    })
    qc = {
        "gpp": np.linspace(300.0, 900.0, n_arch),
        "npp": np.linspace(150.0, 450.0, n_arch),
        "som_kgC": np.linspace(5.0, 15.0, n_arch),
        "biomass_kgC": np.linspace(1.0, 10.0, n_arch),
        "drift_frac_per_yr": np.linspace(-1e-3, 1e-3, n_arch),
    }
    return eq, qc


def _counting_stub(eq, qc, calls):
    def stub(table, *, n_spinup, n_verify, dt, n_layers, soil_depth):
        calls["n"] += 1
        return eq, qc
    return stub


_SPIN = {"n_spinup": 20, "n_verify": 4, "dt": 7200.0, "n_layers": 8,
         "soil_depth": 3.0}


def _cache_path(cache_dir: str, key: str) -> str:
    """The backend-namespaced cache path _load_or_equilibrate writes/reads."""
    import jax
    return os.path.join(cache_dir, jax.default_backend(), key + ".npz")


def _patch_equilibrate(monkeypatch, calls):
    eq, qc = _stub_equilibrium_result()
    monkeypatch.setattr(bgc, "equilibrate_archetypes",
                        _counting_stub(eq, qc, calls))
    return eq, qc


def test_load_or_equilibrate_hit_miss_and_roundtrip(tmp_path, monkeypatch):
    """Call #1 computes + writes the npz; call #2 (same table+spin) is a CACHE HIT
    (equilibrate_archetypes NOT called again) reconstructing the SAME eq pools + qc
    bundle EXACTLY (values + dtypes); a changed spin and a bumped
    _EQUILIBRIUM_CACHE_VERSION both MISS (recompute)."""
    calls = {"n": 0}
    eq0, qc0 = _patch_equilibrate(monkeypatch, calls)
    table = _tiny_archetype_table()
    cache_dir = str(tmp_path / "eq")

    # call #1 -> MISS: computes and writes the npz
    eq1, qc1 = bgc._load_or_equilibrate(
        table, _SPIN, cache_dir=cache_dir, rebuild=False)
    assert calls["n"] == 1
    key = bgc._equilibrium_cache_key(table, _SPIN)
    assert os.path.exists(_cache_path(cache_dir, key))

    # call #2 -> CACHE HIT: stub NOT called again; arrays reconstruct exactly
    eq2, qc2 = bgc._load_or_equilibrate(
        table, _SPIN, cache_dir=cache_dir, rebuild=False)
    assert calls["n"] == 1  # unchanged -> served from disk

    # every pool round-trips value + dtype exactly (byte-identical to a compute)
    for f in eq0._fields:
        a1, a2 = np.asarray(getattr(eq1, f)), np.asarray(getattr(eq2, f))
        npt.assert_array_equal(a1, a2)
        assert a1.dtype == a2.dtype, f
        npt.assert_array_equal(a2, np.asarray(getattr(eq0, f)))
    # qc keys + values reconstruct exactly (same set, same values)
    assert set(qc2) == set(qc0)
    for k in qc0:
        npt.assert_array_equal(np.asarray(qc1[k]), np.asarray(qc2[k]))
        npt.assert_array_equal(np.asarray(qc2[k]), np.asarray(qc0[k]))

    # a changed spin param -> MISS (recompute)
    bgc._load_or_equilibrate(table, dict(_SPIN, n_spinup=40),
                             cache_dir=cache_dir, rebuild=False)
    assert calls["n"] == 2

    # bump the stale-physics guard -> MISS even with identical table+spin
    monkeypatch.setattr(bgc, "_EQUILIBRIUM_CACHE_VERSION", "v2-test")
    bgc._load_or_equilibrate(table, _SPIN, cache_dir=cache_dir, rebuild=False)
    assert calls["n"] == 3


def test_load_or_equilibrate_rebuild_bypasses_cache(tmp_path, monkeypatch):
    """rebuild=True (--rebuild-equilibrium) recomputes + overwrites even on a hit."""
    calls = {"n": 0}
    _patch_equilibrate(monkeypatch, calls)
    table = _tiny_archetype_table()
    cache_dir = str(tmp_path / "eq")
    bgc._load_or_equilibrate(table, _SPIN, cache_dir=cache_dir, rebuild=False)  # writes
    assert calls["n"] == 1
    bgc._load_or_equilibrate(table, _SPIN, cache_dir=cache_dir, rebuild=True)   # forced
    assert calls["n"] == 2


def test_load_or_equilibrate_empty_dir_disables_cache(tmp_path, monkeypatch):
    """cache_dir='' always recomputes and writes NOTHING to disk (unchanged
    behavior when the cache is not opted into)."""
    calls = {"n": 0}
    _patch_equilibrate(monkeypatch, calls)
    table = _tiny_archetype_table()
    bgc._load_or_equilibrate(table, _SPIN, cache_dir="", rebuild=False)
    bgc._load_or_equilibrate(table, _SPIN, cache_dir="", rebuild=False)
    assert calls["n"] == 2  # never cached
    # nothing was written under tmp_path
    assert not list(tmp_path.glob("*.npz"))


def test_load_or_equilibrate_recovers_from_corrupt_npz(tmp_path, monkeypatch):
    """A corrupt/truncated npz at the cache path is dropped + recomputed (never a
    hard crash on every future run), then re-saved as a clean cache."""
    calls = {"n": 0}
    _patch_equilibrate(monkeypatch, calls)
    table = _tiny_archetype_table()
    cache_dir = str(tmp_path / "eq")
    key = bgc._equilibrium_cache_key(table, _SPIN)
    bad = _cache_path(cache_dir, key)
    os.makedirs(os.path.dirname(bad), exist_ok=True)
    with open(bad, "wb") as fh:
        fh.write(b"not a real npz zip archive")  # garbage -> BadZipFile on load
    # must NOT raise: drops the bad file and recomputes
    bgc._load_or_equilibrate(table, _SPIN, cache_dir=cache_dir, rebuild=False)
    assert calls["n"] == 1
    # a GOOD file was re-saved atomically -> the next call is a clean CACHE HIT
    bgc._load_or_equilibrate(table, _SPIN, cache_dir=cache_dir, rebuild=False)
    assert calls["n"] == 1


def test_load_or_equilibrate_incomplete_npz_self_heals(tmp_path, monkeypatch):
    """A LOADABLE npz that is STRUCTURALLY INCOMPLETE (missing one QC member) is
    NOT served as a partial hit -- it is dropped + recomputed, then re-saved
    complete so the next call is a clean hit.  Guards codex Medium-2."""
    calls = {"n": 0}
    eq0, qc0 = _stub_equilibrium_result()
    monkeypatch.setattr(bgc, "equilibrate_archetypes",
                        _counting_stub(eq0, qc0, calls))
    table = _tiny_archetype_table()
    cache_dir = str(tmp_path / "eq")
    p = _cache_path(cache_dir, bgc._equilibrium_cache_key(table, _SPIN))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    # write a VALID npz missing exactly one QC member (qc_npp)
    arrays = {f"eq_{f}": np.asarray(getattr(eq0, f)) for f in eq0._fields}
    arrays.update({f"qc_{k}": np.asarray(qc0[k])
                   for k in bgc._QC_KEYS if k != "npp"})
    np.savez(p, **arrays)
    # must NOT crash / partial-hit: drop + recompute
    bgc._load_or_equilibrate(table, _SPIN, cache_dir=cache_dir, rebuild=False)
    assert calls["n"] == 1
    # a complete file was re-saved -> next call is a clean CACHE HIT
    bgc._load_or_equilibrate(table, _SPIN, cache_dir=cache_dir, rebuild=False)
    assert calls["n"] == 1


def test_check_cached_array_validates_dtype_and_shape():
    """_check_cached_array accepts a numeric (n_arch,) array and rejects a wrong
    shape or a non-numeric dtype (so a jnp.asarray TypeError can't slip through as
    an uncaught crash on a cache read)."""
    npt.assert_array_equal(
        bgc._check_cached_array(np.linspace(0.0, 1.0, 3), "eq_C_lab", 3),
        np.linspace(0.0, 1.0, 3))
    # integer dtype is numeric and allowed (kind in 'fiu')
    bgc._check_cached_array(np.arange(3, dtype=np.int64), "qc_x", 3)
    for bad in (np.zeros((3, 2)), np.zeros(4),
                np.array(["a", "b", "c"], dtype=object)):
        with pytest.raises(ValueError):
            bgc._check_cached_array(bad, "eq_x", 3)


def test_equilibrium_cache_key_deterministic_and_input_sensitive():
    """Same inputs -> same key (deterministic); any table/spin change -> a new key.

    (Guards collision-safety: distinct inputs must not alias to one digest.)"""
    t = _tiny_archetype_table()
    k = bgc._equilibrium_cache_key(t, _SPIN)
    assert k == bgc._equilibrium_cache_key(t, _SPIN)          # deterministic
    assert len(k) == 64                                        # sha256 hexdigest
    # a numeric-field change misses
    assert bgc._equilibrium_cache_key(t._replace(mat_k=t.mat_k + 1.0), _SPIN) != k
    assert bgc._equilibrium_cache_key(
        t._replace(pft_id=t.pft_id + 1), _SPIN) != k
    # a soil_class (string field) change misses
    assert bgc._equilibrium_cache_key(
        t._replace(soil_class=np.array(["sand", "sand", "clay"], dtype=object)),
        _SPIN) != k
    # every spin-config field is in the key
    for f, v in [("n_spinup", 21), ("n_verify", 5), ("dt", 3600.0),
                 ("n_layers", 6), ("soil_depth", 2.0)]:
        assert bgc._equilibrium_cache_key(t, dict(_SPIN, **{f: v})) != k, f


def test_equilibrium_cache_key_version_sensitive(monkeypatch):
    """Bumping _EQUILIBRIUM_CACHE_VERSION changes the key (the stale-physics
    guard: a physics/default-param change with UNCHANGED inputs MUST miss)."""
    t = _tiny_archetype_table()
    k = bgc._equilibrium_cache_key(t, _SPIN)
    monkeypatch.setattr(bgc, "_EQUILIBRIUM_CACHE_VERSION", "v999-test")
    assert bgc._equilibrium_cache_key(t, _SPIN) != k
