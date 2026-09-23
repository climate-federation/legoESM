"""scripts/experiment/remap_mpas_checkpoint.py: a sigma-lane checkpoint moved to
a tropopause-refined grid keeps every column inventory (T, u, tracers, tke) to
round-off, stays within the source range, rewrites meta_vgrid to the target
coordinate exactly as the driver builds it, copies non-column payloads
unchanged, and the identity grid is a bitwise copy.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "experiment" / "remap_mpas_checkpoint.py")
_spec = importlib.util.spec_from_file_location("remap_mpas_checkpoint", _SCRIPT)
remap = importlib.util.module_from_spec(_spec)
sys.modules["remap_mpas_checkpoint"] = remap
_spec.loader.exec_module(remap)

NLEV, NCOL, NEDGE = 30, 5, 7
# Enumerated independently of the tool: every level-resolved field a sigma-lane
# checkpoint carries, by remap class.
CONSERVED = ("T", "u", "trc_q_v", "trc_q_c", "trc_q_i", "trc_q_r", "trc_q_s", "trc_q_g",
             "trc_N_c", "trc_N_i", "trc_N_r", "physstate_aerosol_number",
             "physstate_tke", "physstate_qke")
BOUNDED = ("physstate_cloud_fraction",)
PROFILES = ("physstate_conv_prog_profile", "physstate_rad_heating")


def _fake_checkpoint(path, sigma_half):
    rng = np.random.default_rng(7)
    sf = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    T = 300.0 - 60.0 * (1.0 - sf)[None, :] + rng.normal(size=(NCOL, NLEV))
    T[:, 0] = 205.0                                      # a cold top layer
    d = {
        "T": T, "u": rng.normal(size=(NEDGE, NLEV)) * 20.0,
        "p_s": np.full(NCOL, 1.0e5), "phis": np.zeros(NCOL),
        "meta_vgrid": np.stack([np.zeros(NLEV + 1), sigma_half]),
        "day": np.float64(90.0), "step": np.int64(69120),
        "land_ml_T_soil": rng.normal(size=(NCOL, 10)) + 280.0,
        "physstate_clubb_moments": np.zeros((NCOL, 1, 1)),
        "physstate_prng_key": np.array([1, 2], dtype=np.uint32),
        "tracer_names": np.array(["q_v", "q_c"]),
    }
    for n in CONSERVED[2:]:
        d[n] = np.abs(rng.normal(size=(NCOL, NLEV))) * (1e-3 if n.startswith("trc_q") else 1.0)
    d["physstate_cloud_fraction"] = np.clip(rng.uniform(size=(NCOL, NLEV)), 0.0, 1.0)
    d["physstate_conv_prog_profile"] = np.abs(rng.normal(size=(NCOL, NLEV)))
    d["physstate_rad_heating"] = (rng.normal(size=(NCOL, NLEV)) * 1e-5).astype(np.float32)
    np.savez(path, **d)
    return d


@pytest.fixture
def uniform_src(tmp_path):
    src = tmp_path / "src.npz"
    d = _fake_checkpoint(src, np.linspace(0.01, 1.0, NLEV + 1))
    return src, d


def test_identity_grid_is_a_bitwise_copy(uniform_src, tmp_path):
    src, d = uniform_src
    dst = tmp_path / "dst.npz"
    assert remap.main(["--src", str(src), "--dst", str(dst), "--tropopause-refine", "1.0"]) == 0
    out = np.load(dst, allow_pickle=True)
    for k, v in d.items():
        assert np.array_equal(out[k], np.asarray(v)), k
    assert "remap_meta" in out.files


def test_refined_grid_conserves_and_stays_bounded(uniform_src, tmp_path):
    from legoesm.grids.vertical import create_sigma_coordinate
    import jax.numpy as jnp
    src, d = uniform_src
    dst = tmp_path / "dst.npz"
    assert remap.main(["--src", str(src), "--dst", str(dst), "--tropopause-refine", "3.0"]) == 0
    out = np.load(dst, allow_pickle=True)
    s_old = d["meta_vgrid"][1]
    coord = create_sigma_coordinate(NLEV, tropopause_refine=3.0, dtype=jnp.float64)
    s_new = np.asarray(coord.sigma_half)
    assert np.array_equal(out["meta_vgrid"], np.stack([np.zeros(NLEV + 1), s_new]))
    assert not np.allclose(s_new, s_old)
    for k in CONSERVED:
        ib = np.sum(d[k] * np.diff(s_old)[None, :], axis=1)
        ia = np.sum(out[k] * np.diff(s_new)[None, :], axis=1)
        np.testing.assert_allclose(ia, ib, rtol=1e-12, atol=1e-30, err_msg=k)
        assert not np.array_equal(out[k], d[k]), k          # actually remapped
        span = 1e-2 * (d[k].max() - d[k].min())     # PPM overshoot on noise is ~1e-3 of the range
        assert out[k].min() >= d[k].min() - span and out[k].max() <= d[k].max() + span, k
        if k.startswith("trc_") or k.endswith("tke"):
            assert out[k].min() >= 0.0, k
    assert out["physstate_cloud_fraction"].min() >= 0.0
    assert out["physstate_cloud_fraction"].max() <= 1.0
    assert out["physstate_rad_heating"].dtype == np.float32
    sf_old = 0.5 * (s_old[:-1] + s_old[1:])
    sf_new = np.asarray(coord.sigma_full)
    for k in PROFILES:                                   # independent reference
        ref = np.stack([np.interp(sf_new, sf_old, d[k][i].astype(np.float64))
                        for i in range(NCOL)])
        np.testing.assert_allclose(out[k].astype(np.float64), ref,
                                   rtol=1e-6 if k.endswith("heating") else 1e-12, err_msg=k)
        assert not np.array_equal(out[k], d[k]), k
    for k in ("p_s", "phis", "land_ml_T_soil", "physstate_prng_key", "step", "day"):
        assert np.array_equal(out[k], np.asarray(d[k])), k
    meta = json.loads(str(out["remap_meta"]))
    assert meta["tropopause_refine"] == 3.0 and len(meta["sha256"]) == 64


def test_refuses_hybrid_or_existing_destination(uniform_src, tmp_path):
    src, d = uniform_src
    hyb = tmp_path / "hyb.npz"
    dd = dict(d)
    dd["meta_vgrid"] = np.stack([np.linspace(0.0, 1.0, NLEV + 1), d["meta_vgrid"][1]])
    np.savez(hyb, **dd)
    assert remap.main(["--src", str(hyb), "--dst", str(tmp_path / "x.npz"),
                       "--tropopause-refine", "3.0"]) == 1
    dst = tmp_path / "exists.npz"
    dst.write_bytes(b"")
    assert remap.main(["--src", str(src), "--dst", str(dst), "--tropopause-refine", "3.0"]) == 1


def test_refuses_nan_and_level_resolved_carries(uniform_src, tmp_path):
    src, d = uniform_src
    bad = tmp_path / "nan.npz"
    dd = dict(d)
    dd["trc_q_v"] = d["trc_q_v"].copy()
    dd["trc_q_v"][2, 11] = np.nan                         # one poisoned cell
    np.savez(bad, **dd)
    out = tmp_path / "nan_out.npz"
    assert remap.main(["--src", str(bad), "--dst", str(out), "--tropopause-refine", "3.0"]) == 1
    assert not out.exists()
    for nan_field, refine in (("physstate_cloud_fraction", "3.0"), ("trc_q_c", "1.0")):
        bad = tmp_path / f"nan_{nan_field}_{refine}.npz"
        dd = dict(d)
        dd[nan_field] = d[nan_field].copy()
        dd[nan_field][1, 3] = np.nan
        np.savez(bad, **dd)
        out = tmp_path / f"nan_{nan_field}_{refine}_out.npz"
        assert remap.main(["--src", str(bad), "--dst", str(out), "--tropopause-refine", refine]) == 1
        assert not out.exists(), (nan_field, refine)     # identity path and bounded field too
    clubb = tmp_path / "clubb.npz"
    dd = dict(d)
    dd["physstate_clubb_moments"] = np.zeros((NCOL, 15, NLEV + 1))   # prognostic CLUBB
    np.savez(clubb, **dd)
    out = tmp_path / "clubb_out.npz"
    assert remap.main(["--src", str(clubb), "--dst", str(out), "--tropopause-refine", "3.0"]) == 1
    assert not out.exists()


def test_remapping_twice_keeps_one_provenance_key(uniform_src, tmp_path):
    src, _d = uniform_src
    mid = tmp_path / "mid.npz"
    assert remap.main(["--src", str(src), "--dst", str(mid), "--tropopause-refine", "3.0"]) == 0
    again = tmp_path / "again.npz"
    assert remap.main(["--src", str(mid), "--dst", str(again), "--tropopause-refine", "3.0"]) == 0
    assert np.load(again, allow_pickle=True).files.count("remap_meta") == 1
