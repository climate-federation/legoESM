"""End-to-end guard for the multilayer-land checkpoint round-trip (#769/#770).

The companion ``test_land_ml_checkpoint_roundtrip.py`` exercises the two halves
(``_checkpoint_carry_aux`` pack + ``_restore_land_ml_from_carry_aux`` rebuild) on
``SimpleNamespace`` mocks.  That leaves the *contract that actually fixes #769*
untested: a real ``ModelDriver`` must build ``_land_ml_state`` during ``setup()``
**before** ``load_checkpoint`` runs, and ``load_checkpoint`` must populate
``_carry_aux`` from the on-disk npz **before** calling the restore — so a future
reorder (or a dropped restore call site) silently reintroduces the cold-start
that #769 reported, with the mock tests still green.

This test drives the production ``save_checkpoint`` -> ``load_checkpoint`` path on
a real (single-column, synthetic-loader) multilayer driver and asserts the soil
columns come back, not the cold-start seed.  Loaders are monkeypatched to
synthetic data so it stays offline + portable (mirrors
``test_multilayer_land_driver.py``).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.driver.config import (
    ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver
from legoesm.land.state import MultiLayerLandState


def _fake_surface_map(path, lat_deg, lon_deg):
    """A uniform-loam, all-bare-soil CLM map sized to the requested columns."""
    n = int(np.asarray(lat_deg).size)
    pft = np.zeros((n, 17)); pft[:, 0] = 1.0          # 100% bare soil
    o = np.ones(n)
    return dict(
        pft_fractions=jnp.asarray(pft),
        theta_wp=jnp.asarray(0.12 * o), theta_fc=jnp.asarray(0.30 * o),
        glacier_frac=jnp.asarray(np.zeros(n)),
        pct_sand=jnp.asarray(40.0 * o), pct_clay=jnp.asarray(20.0 * o),
        theta_r=jnp.asarray(0.05 * o), theta_sat=jnp.asarray(0.45 * o),
        alpha_vg=jnp.asarray(2.0 * o), n_vg=jnp.asarray(1.4 * o),
        K_sat=jnp.asarray(1.0e-5 * o),
    )


def _patch_land_loaders(monkeypatch):
    import legoesm.land.clm_surface_map as clm
    import legoesm.grids.topography as topo
    monkeypatch.setattr(clm, "download_clm_surfdata", lambda *a, **k: "synthetic")
    monkeypatch.setattr(clm, "load_clm_surface", _fake_surface_map)
    monkeypatch.setattr(
        topo, "load_land_fraction",
        lambda grid, path, *a, **k: jnp.full(grid.lat.shape, 0.5))


def _small_cfg(**over):
    base = dict(
        grid=GridConfig(resolution=8, nlev=8),
        dycore=DycoreConfig(dt=600.0),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical",
        radiation="gray",
        land_mask_path="synthetic.nc",
        use_multilayer_land=True,
        multilayer_n_layers=6, multilayer_soil_depth=2.5,
    )
    base.update(over)
    return ExperimentConfig(**base)


def test_multilayer_land_survives_real_driver_checkpoint(monkeypatch, tmp_path):
    """save_checkpoint -> load_checkpoint on a real driver resumes the evolved
    soil, not the cold-start seed — the end-to-end #769 contract."""
    _patch_land_loaders(monkeypatch)

    src = ModelDriver(_small_cfg(), output_dir=tmp_path / "run_a")
    src.setup()
    seed = src._land_ml_state
    assert isinstance(seed, MultiLayerLandState)

    # Perturb the prognostic columns off the deterministic cold-start seed so a
    # restore is distinguishable from a re-seed.  A physical, per-field-distinct
    # offset (a dropped/swapped field is caught, not masked).
    evolved = seed._replace(
        T_soil=seed.T_soil + 4.0,
        psi_soil=seed.psi_soil - 0.5,
        theta_soil=jnp.clip(seed.theta_soil * 0.9, 0.0, 1.0),
        snow_depth=seed.snow_depth + 0.02,
    )
    src._land_ml_state = evolved

    day = float(src.config.start_day) + 1.0
    src.save_checkpoint(step=144, day=day)
    ckpt = tmp_path / "run_a" / f"checkpoint_day_{int(round(day)):04d}.npz"
    assert ckpt.exists(), f"checkpoint not written at {ckpt}"

    # A fresh driver lands on the cold-start seed; load must overwrite it.
    dst = ModelDriver(_small_cfg(), output_dir=tmp_path / "run_b")
    dst.setup()
    assert not np.allclose(
        np.asarray(dst._land_ml_state.T_soil), np.asarray(evolved.T_soil)
    ), "cold-start seed already equals the evolved state; test is vacuous"

    dst.load_checkpoint(ckpt)

    for f in MultiLayerLandState._fields:
        want = getattr(evolved, f)
        got = getattr(dst._land_ml_state, f)
        if want is None:
            assert got is None, f"optional field {f} should restore to None"
        else:
            np.testing.assert_allclose(
                np.asarray(got), np.asarray(want), rtol=1e-6, atol=1e-6,
                err_msg=f"multilayer land field {f} did not survive the "
                        f"real save/load round-trip (cold-start leak, #769)")


def test_slab_driver_checkpoint_has_no_land_ml(monkeypatch, tmp_path):
    """A slab (non-multilayer) driver never writes land_ml_* and loads clean —
    the fix must not perturb the byte-identical slab restart."""
    _patch_land_loaders(monkeypatch)

    src = ModelDriver(_small_cfg(use_multilayer_land=False), output_dir=tmp_path / "s_a")
    src.setup()
    assert src._land_ml_state is None

    day = float(src.config.start_day) + 1.0
    src.save_checkpoint(step=144, day=day)
    ckpt = tmp_path / "s_a" / f"checkpoint_day_{int(round(day)):04d}.npz"
    assert ckpt.exists()

    # save_restart serializes carry_aux fields with a ``carry_`` prefix
    # (``carry_land_ml_*``), so match the substring, not a bare prefix.
    with np.load(ckpt, allow_pickle=False) as z:
        assert not any("land_ml_" in k for k in z.files), \
            "slab checkpoint must not contain land_ml_* carry keys"

    dst = ModelDriver(_small_cfg(use_multilayer_land=False), output_dir=tmp_path / "s_b")
    dst.setup()
    dst.load_checkpoint(ckpt)                 # must not raise
    assert dst._land_ml_state is None


def test_band_mpi_land_ml_guard_predicate():
    """Directly exercise the band-MPI fail-fast predicate: the guard itself
    only fires on the distributed restart path the single-process tests above
    never reach, so without this a prefix rename would silently disable it."""
    f = ModelDriver._carry_has_unscatterable_land_ml
    assert f({"land_ml_T_soil": 1}) is True            # multilayer -> must block
    assert f({}) is False                              # clean restart -> allow
    assert f(None) is False                            # no carry_aux -> allow
    assert f({"T_land": 1, "dmtr_q_v": 2}) is False    # slab/other carry -> allow
    # The guard runs on the POST-bcast in-memory carry_aux, whose keys are bare
    # (load strips the on-disk ``carry_``/``diag_`` prefix), so an on-disk-style
    # key must NOT match — matching semantics are startswith, not substring.
    assert f({"carry_land_ml_T_soil": 1}) is False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
