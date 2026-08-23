"""Direct unit tests for the eta wave-field twin instrument
(``scripts/validate/ocean_fidelity/dino_1226/eta_wave_twin.py``).

The instrument decides what we believe about the wave comparison, so its
masking, its locus partition and its spectral estimator are tested here on
synthetic data with KNOWN answers -- including the synthetic-violation checks
that prove the mask control is not vacuous.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@pytest.fixture(scope="module")
def ewt():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.dino_1226.eta_wave_twin as m
        return m
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def _basin(ny=9, nx=7):
    """A tiny closed basin: land ring, wet interior."""
    wet = np.zeros((ny, nx), dtype=bool)
    wet[1:-1, 1:-1] = True
    lat = np.linspace(-4.0, 4.0, ny)[:, None] * np.ones((1, nx))
    return wet, lat


def test_masked_stats_ignores_dry_cells(ewt):
    wet, _ = _basin()
    diff = np.zeros(wet.shape)
    diff[2, 2] = 0.5           # wet
    diff[0, 0] = 1000.0        # dry
    st = ewt.masked_stats(diff, wet)
    assert st["max_abs_m"] == pytest.approx(0.5)
    assert (st["argmax_j"], st["argmax_i"]) == (2, 2)


def test_plant_dry_violation_is_inert_on_masked_stats(ewt):
    wet, _ = _basin()
    rng = np.random.default_rng(0)
    eta = rng.normal(size=wet.shape)
    poisoned = ewt.plant_dry_violation(eta, wet)
    assert ewt.masked_stats(eta, wet) == ewt.masked_stats(poisoned, wet)
    # ...and NOT inert on the UNMASKED statistic, which is what makes the
    # control non-vacuous.  (Asserting that the poison itself is large would
    # be a tautology -- the function writes the value.)
    assert (ewt.masked_stats(poisoned, np.ones_like(wet))["max_abs_m"]
            > 1e3 * ewt.masked_stats(poisoned, wet)["max_abs_m"])


def test_plant_dry_violation_refuses_an_all_wet_frame(ewt):
    """A control that cannot fire must abort, not silently pass."""
    wet = np.ones((4, 4), dtype=bool)
    with pytest.raises(SystemExit, match="cannot fire"):
        ewt.plant_dry_violation(np.zeros((4, 4)), wet)


def test_masked_stats_would_move_without_masking(ewt):
    """Synthetic violation: remove the mask and the statistic MUST change.

    This is the non-vacuity proof for every masked number the probe reports.
    """
    wet, _ = _basin()
    diff = np.zeros(wet.shape)
    diff[2, 2] = 0.5
    diff[0, 0] = 1000.0
    masked = ewt.masked_stats(diff, wet)["max_abs_m"]
    unmasked = ewt.masked_stats(diff, np.ones_like(wet))["max_abs_m"]
    assert masked == pytest.approx(0.5)
    assert unmasked == pytest.approx(1000.0)


def test_locus_partition_is_a_disjoint_cover_of_the_wet_domain(ewt):
    wet, lat = _basin(ny=21, nx=11)
    r = ewt.locus_partition(wet, lat)
    stack = np.stack([r["wall"], r["equator"], r["interior"]])
    assert np.array_equal(stack.any(axis=0), wet)          # covers
    assert stack.sum(axis=0).max() <= 1                     # disjoint
    # every wet cell touching land is in the wall band
    assert r["wall"][1, 1] and r["wall"][1, 5]
    # the equator band is the |lat| <= 2 strip away from the walls
    assert r["equator"].any()
    assert np.all(np.abs(lat[r["equator"]]) <= 2.0)


def test_locus_shares_sum_to_one_and_localise(ewt):
    wet, lat = _basin(ny=21, nx=11)
    r = ewt.locus_partition(wet, lat)
    diff = np.zeros(wet.shape)
    diff[r["wall"]] = 1.0
    sh = ewt.locus_shares(diff, r)
    assert sh["wall"]["share"] == pytest.approx(1.0)
    assert sum(v["share"] for v in sh.values()) == pytest.approx(1.0)
    # the enrichment is the number that can name a locus: the wall holds all
    # the signal while occupying a fraction of the domain
    assert sh["wall"]["enrichment"] > 1.0
    assert sh["wall"]["enrichment"] == pytest.approx(
        1.0 / sh["wall"]["area_fraction"])


def test_locus_shares_of_an_exactly_zero_field_is_zero_not_nan(ewt):
    wet, lat = _basin()
    r = ewt.locus_partition(wet, lat)
    sh = ewt.locus_shares(np.zeros(wet.shape), r)
    assert all(v["share"] == 0.0 for v in sh.values())


def test_spectra_recovers_a_known_sinusoid(ewt):
    """Known-answer control for the estimator, before it is used on model data."""
    dt = 2700.0
    n = 512
    period_h = 8.0
    t = np.arange(n) * dt
    amp = 0.37
    x = amp * np.sin(2 * np.pi * t / (period_h * 3600.0)) + 5.0
    f, a = ewt.spectra(x, dt)
    peak = np.argmax(a[1:]) + 1
    assert 1.0 / (f[peak] * 3600.0) == pytest.approx(period_h, rel=0.05)
    assert a[peak] == pytest.approx(amp, rel=0.05)
    # The constant offset must not appear at zero frequency.  The bound is
    # 1e-3 of the amplitude, not machine zero: the series is de-meaned
    # exactly, so the DC bin holds only Hann leakage from the sinusoid not
    # spanning a whole number of periods (measured ~5e-7 relative here).
    assert a[0] < 1e-3 * amp


def test_load_side_rejects_nan(ewt, tmp_path):
    p = tmp_path / "bad.npz"
    eta = np.zeros((3, 4, 5))
    eta[1, 1, 1] = np.nan
    np.savez(p, eta=eta, t_seconds=np.arange(3.0))
    with pytest.raises(SystemExit, match="NaN"):
        ewt.load_side(str(p), "x")


def test_load_side_rejects_a_mismatched_time_axis(ewt, tmp_path):
    p = tmp_path / "bad2.npz"
    np.savez(p, eta=np.zeros((3, 4, 5)), t_seconds=np.arange(2.0))
    with pytest.raises(SystemExit, match="t_seconds"):
        ewt.load_side(str(p), "x")


def test_provenance_stamps_sha_clock_and_dtype(ewt):
    prov = ewt.provenance({"source": "test"})
    for key in ("git_sha", "git_dirty", "utc", "dtype", "numpy", "argv"):
        assert key in prov
    assert prov["source"] == "test"
    # NB: prov["dtype"] describes the ANALYSIS and is a literal, so asserting
    # it here would be a tautology. The dtype each ARTIFACT was stored at is
    # checked where it matters, in test_compare_refuses_a_float32_lego_artifact.


def test_variance_bands_puts_a_known_sinusoid_in_the_right_band(ewt):
    """Known-answer control: a pure 4 h oscillation must land in '<6h'."""
    wet, _ = _basin()
    dt = 2700.0
    n = 256
    t = np.arange(n) * dt
    eta = np.zeros((n,) + wet.shape)
    eta[:, wet] = np.sin(2 * np.pi * t / (4.0 * 3600.0))[:, None]
    b = ewt.variance_bands(eta, wet, dt)
    assert b["lt_6h"] > 0.98
    assert sum(b.values()) == pytest.approx(1.0)


def test_variance_bands_puts_a_slow_signal_in_the_slow_band(ewt):
    wet, _ = _basin()
    dt = 2700.0
    n = 256
    t = np.arange(n) * dt
    eta = np.zeros((n,) + wet.shape)
    eta[:, wet] = np.sin(2 * np.pi * t / (72.0 * 3600.0))[:, None]
    b = ewt.variance_bands(eta, wet, dt)
    assert b["gt_24h"] > 0.98


def test_variance_bands_refuses_a_static_field(ewt):
    wet, _ = _basin()
    with pytest.raises(SystemExit, match="no temporal variance"):
        ewt.variance_bands(np.ones((8,) + wet.shape), wet, 2700.0)


def test_step_noise_ratio_is_one_for_identical_fields(ewt):
    wet, lat = _basin(ny=21, nx=11)
    r = ewt.locus_partition(wet, lat)
    rng = np.random.default_rng(1)
    a = rng.normal(size=(20,) + wet.shape)
    out = ewt.step_noise_ratio(a, a, wet, r)
    assert out["median"] == pytest.approx(1.0)
    assert out["max"] == pytest.approx(1.0)
    assert all(v["share"] == 0.0
               for v in out["excess_share_by_locus"].values())


def test_step_noise_ratio_finds_a_planted_wall_mode(ewt):
    """Synthetic violation: plant a 2-step oscillation on the wall band only.

    The ratio must blow up AND the excess must be attributed to 'wall'.
    """
    wet, lat = _basin(ny=21, nx=11)
    r = ewt.locus_partition(wet, lat)
    rng = np.random.default_rng(2)
    nemo = rng.normal(size=(20,) + wet.shape) * 1e-3
    lego = nemo.copy()
    flip = ((-1.0) ** np.arange(20))[:, None]
    lego[:, r["wall"]] += flip * 1.0
    out = ewt.step_noise_ratio(lego, nemo, wet, r)
    assert out["max"] > 100.0
    assert out["excess_share_by_locus"]["wall"]["share"] > 0.99
    assert out["excess_share_by_locus"]["wall"]["enrichment"] > 1.0
    assert r["wall"][out["argmax_j"], out["argmax_i"]]


def test_step_noise_ratio_excludes_a_motionless_nemo_cell(ewt):
    """A cell where the oracle never moves must not become an infinite ratio."""
    wet, lat = _basin(ny=21, nx=11)
    r = ewt.locus_partition(wet, lat)
    rng = np.random.default_rng(3)
    nemo = rng.normal(size=(20,) + wet.shape)
    nemo[:, 10, 5] = 7.0          # exactly static
    lego = rng.normal(size=(20,) + wet.shape)
    out = ewt.step_noise_ratio(lego, nemo, wet, r)
    assert np.isfinite(out["max"])
    assert out["n_live_cells"] == int(wet.sum()) - 1


def test_time_level_discriminator_picks_the_matching_level(ewt):
    """Known-answer control: build a series where lego == sshn exactly."""
    wet, _ = _basin()
    rng = np.random.default_rng(4)
    sshn = rng.normal(size=(12,) + wet.shape)
    sshb = sshn + 0.5 * rng.normal(size=sshn.shape)
    d = ewt.time_level_discriminator(sshn.copy(), sshn, sshb, wet)
    assert d["registered_lego_N_vs_sshn_N"] == pytest.approx(0.0)
    assert d["alt_lego_N_vs_sshb_N"] > 0.0
    assert d["nemo_own_sshn_minus_sshb"] > 0.0


def test_time_level_discriminator_would_expose_a_wrong_pairing(ewt):
    """Synthetic violation: if lego actually equals sshb, the check must say so."""
    wet, _ = _basin()
    rng = np.random.default_rng(5)
    sshn = rng.normal(size=(12,) + wet.shape)
    sshb = sshn + 0.5 * rng.normal(size=sshn.shape)
    d = ewt.time_level_discriminator(sshb.copy(), sshn, sshb, wet)
    assert d["alt_lego_N_vs_sshb_N"] < d["registered_lego_N_vs_sshn_N"]


def test_highpass_removes_the_slow_component(ewt):
    dt = 2700.0
    n = 256
    t = np.arange(n) * dt
    slow = 3.0 * np.sin(2 * np.pi * t / (96.0 * 3600.0))
    fast = 0.5 * np.sin(2 * np.pi * t / (4.0 * 3600.0))
    out = ewt.highpass((slow + fast)[:, None], dt)[:, 0]
    assert np.abs(out - fast).max() < 1e-8 * 0.5 + 1e-9
    assert out.std() == pytest.approx(fast.std(), rel=1e-6)


def test_propagation_lag_recovers_a_planted_travelling_wave(ewt):
    """Known-answer control: a signal shifted by exactly 2 steps per cell."""
    dt = 2700.0
    dx = 60_000.0
    n, nx, ny = 256, 12, 3
    wet = np.ones((ny, nx), dtype=bool)
    rng = np.random.default_rng(7)
    base = ewt.highpass(rng.normal(size=(n, 1)), dt)[:, 0]
    eta = np.zeros((n, ny, nx))
    for i in range(nx):
        eta[:, 1, i] = np.roll(base, 2 * i)
    out = ewt.propagation_lag(eta, wet, 1, dx, dt)
    assert out["median_lag_steps"] == pytest.approx(2.0)
    assert out["phase_speed_m_s"] == pytest.approx(dx / (2 * dt))
    assert out["n_pairs"] == nx - 1


def test_propagation_lag_reports_standing_when_there_is_no_lag(ewt):
    """A signal identical in every column must read as zero lag, not a speed."""
    dt = 2700.0
    n, nx, ny = 128, 10, 3
    wet = np.ones((ny, nx), dtype=bool)
    rng = np.random.default_rng(8)
    base = ewt.highpass(rng.normal(size=(n, 1)), dt)[:, 0]
    eta = np.zeros((n, ny, nx))
    eta[:, 1, :] = base[:, None]
    out = ewt.propagation_lag(eta, wet, 1, 60_000.0, dt)
    assert out["median_lag_steps"] == 0.0
    assert out["phase_speed_m_s"] is None
    assert out["frac_pairs_within_one_step"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# fixes demanded by the two adversarial reviews of this instrument
# ---------------------------------------------------------------------------
def test_locus_partition_does_not_call_a_periodic_seam_a_wall(ewt):
    """DINO is zonally periodic; the first and last columns are the ACC
    channel's seam, not a wall. This test FAILS if the wrap is removed."""
    ny, nx = 11, 8
    wet = np.zeros((ny, nx), dtype=bool)
    wet[1:-1, :] = True                 # open all the way round in x
    lat = np.linspace(-40.0, 40.0, ny)[:, None] * np.ones((1, nx))
    r = ewt.locus_partition(wet, lat, periodic_i=True)
    # an interior row's seam columns have wet neighbours all round
    assert not r["wall"][5, 0]
    assert not r["wall"][5, -1]
    # ...and with the wrap switched off they WOULD be walls
    r2 = ewt.locus_partition(wet, lat, periodic_i=False)
    assert r2["wall"][5, 0] and r2["wall"][5, -1]


def test_locus_partition_wall_width_grows(ewt):
    wet, lat = _basin(ny=21, nx=13)
    one = ewt.locus_partition(wet, lat, wall_cells=1)["wall"].sum()
    two = ewt.locus_partition(wet, lat, wall_cells=2)["wall"].sum()
    assert two > one


def test_locus_partition_equator_width_is_a_parameter(ewt):
    wet, lat = _basin(ny=41, nx=9)
    narrow = ewt.locus_partition(wet, lat, equator_half_width_deg=1.0)
    wide = ewt.locus_partition(wet, lat, equator_half_width_deg=3.0)
    assert wide["equator"].sum() > narrow["equator"].sum()


def test_masked_stats_rms_is_area_weighted(ewt):
    """An unweighted rms would ignore the area argument entirely."""
    wet = np.ones((2, 2), dtype=bool)
    diff = np.array([[1.0, 0.0], [0.0, 0.0]])
    flat = ewt.masked_stats(diff, wet, None)["rms_m"]
    heavy = ewt.masked_stats(diff, wet, np.array([[9.0, 1.0], [1.0, 1.0]]))
    assert flat == pytest.approx(0.5)
    assert heavy["rms_m"] == pytest.approx(np.sqrt(9.0 / 12.0))
    assert heavy["rms_m"] != pytest.approx(flat)


def test_spectra_does_not_double_count_the_nyquist_bin(ewt):
    """A pure 2-step oscillation sits exactly on Nyquist, where the coherent-
    gain factor is 1/sum(w), not 2/sum(w)."""
    dt = 2700.0
    n = 256
    amp = 0.3
    x = amp * ((-1.0) ** np.arange(n))
    f, a = ewt.spectra(x, dt)
    assert a[-1] == pytest.approx(amp, rel=0.05)


def test_step_noise_ratio_floors_a_near_static_denominator(ewt):
    """A nearly-motionless oracle cell must not dominate the headline max."""
    wet, lat = _basin(ny=21, nx=11)
    r = ewt.locus_partition(wet, lat)
    rng = np.random.default_rng(9)
    nemo = rng.normal(size=(20,) + wet.shape)
    nemo[:, 10, 5] *= 1e-12          # near-static, NOT exactly static
    lego = rng.normal(size=(20,) + wet.shape)
    out = ewt.step_noise_ratio(lego, nemo, wet, r)
    assert out["denominator_floor_m2"] > 0.0
    assert out["max"] < 1e6          # unfloored this would be ~1e24


def test_time_level_discriminator_and_gate_shape(ewt):
    wet, _ = _basin()
    rng = np.random.default_rng(10)
    sshn = rng.normal(size=(12,) + wet.shape)
    sshb = sshn + 0.5 * rng.normal(size=sshn.shape)
    d = ewt.time_level_discriminator(sshn.copy(), sshn, sshb, wet)
    assert set(d) >= {"registered_lego_N_vs_sshn_N", "alt_lego_N_vs_sshb_N",
                      "alt_lego_N_vs_sshn_Nminus1",
                      "alt_lego_Nminus1_vs_sshn_N",
                      "nemo_own_sshn_minus_sshb"}


def _synthetic_pair(tmp_path, nt=24, ny=9, nx=7, lego_offset=0.0):
    """A minimal mesh_mask + two artifacts, enough to run compare() for real."""
    import netCDF4
    mesh = tmp_path / "mesh.nc"
    wet = np.zeros((ny, nx), dtype=bool)
    wet[1:-1, 1:-1] = True
    lat = np.linspace(-60.0, 60.0, ny)[:, None] * np.ones((1, nx))
    lon = np.linspace(0.0, 30.0, nx)[None, :] * np.ones((ny, 1))
    with netCDF4.Dataset(mesh, "w") as ds:
        ds.createDimension("x", nx)
        ds.createDimension("y", ny)
        ds.createDimension("z", 2)
        ds.createDimension("t", 1)
        ds.createVariable("tmask", "f8", ("t", "z", "y", "x"))[:] = np.broadcast_to(
            wet, (1, 2, ny, nx))
        ds.createVariable("nav_lat", "f8", ("y", "x"))[:] = lat
        ds.createVariable("nav_lon", "f8", ("y", "x"))[:] = lon
        ds.createVariable("e1t", "f8", ("t", "y", "x"))[:] = np.full((1, ny, nx), 6e4)
        ds.createVariable("e2t", "f8", ("t", "y", "x"))[:] = np.full((1, ny, nx), 6e4)
    dt = 2700.0
    t = np.arange(1, nt + 1) * dt
    rng = np.random.default_rng(11)
    base = rng.normal(size=(nt, ny, nx)) * 1e-3
    base += np.linspace(0.0, 0.05, nt)[:, None, None]
    npz_n = tmp_path / "nemo.npz"
    npz_l = tmp_path / "lego.npz"
    np.savez(npz_n, eta=base, eta_before=base * 0.999, t_seconds=t,
             nav_lat=lat, nav_lon=lon)
    np.savez(npz_l, eta=base + lego_offset, t_seconds=t)
    return str(mesh), str(npz_n), str(npz_l)


def test_compare_runs_end_to_end_and_reports_the_registered_fields(ewt, tmp_path):
    """The function that produces the scientific number, exercised for real."""
    mesh, npz_n, npz_l = _synthetic_pair(tmp_path, lego_offset=1e-6)
    out = tmp_path / "out"
    res = ewt.compare(npz_n, npz_l, str(out), mesh)
    for key in ("propagation", "time_levels_rms_m", "variance_bands",
                "step_noise_ratio", "one_step_floor_max_abs_m", "linear_bar_m",
                "growth_over_linear_bar", "snapshots", "growth", "spectra",
                "locus_sensitivity", "provenance"):
        assert key in res, key
    assert (out / "eta_wave_twin.json").exists()
    for fig in ("fig_diff_maps.png", "fig_hovmoller.png",
                "fig_growth_locus.png", "fig_spectra.png"):
        assert (out / fig).exists(), fig
    # the pre-registered spectral bar is EVALUATED, not merely stamped
    for v in res["spectra"].values():
        assert isinstance(v["exceeds_spectral_bar"], bool)
    # every retained snapshot lies inside the run
    for s in res["snapshots"]:
        assert s["t_hours"] <= res["growth"][-1]["t_hours"] + 1e-9


def test_compare_refuses_a_float32_lego_artifact(ewt, tmp_path):
    mesh, npz_n, npz_l = _synthetic_pair(tmp_path)
    d = dict(np.load(npz_l))
    d["eta"] = d["eta"].astype(np.float32)
    np.savez(npz_l, **d)
    with pytest.raises(SystemExit, match="float32"):
        ewt.compare(npz_n, npz_l, str(tmp_path / "o2"), mesh)


def test_compare_refuses_a_mismatched_mesh(ewt, tmp_path):
    import netCDF4
    mesh, npz_n, npz_l = _synthetic_pair(tmp_path)
    with netCDF4.Dataset(mesh, "r+") as ds:
        ds.variables["nav_lat"][:] = ds.variables["nav_lat"][:] + 5.0
    with pytest.raises(SystemExit, match="does not match the mesh"):
        ewt.compare(npz_n, npz_l, str(tmp_path / "o3"), mesh)


def _fake_nemo_run(tmp_path, nsteps=6, ny=5, nx=4, offset=0):
    """Write per-step NEMO-style restart tiles obeying the Asselin identity.

    ``offset`` shifts the sshb series by one dump, which is exactly the
    mapping error the identity exists to catch.
    """
    import netCDF4
    rng = np.random.default_rng(12)
    sshn = np.cumsum(rng.normal(size=(nsteps + 2, ny, nx)) * 1e-3, axis=0)
    atfp = 0.1
    sshb = np.zeros_like(sshn)
    sshb[0] = sshn[0]
    for n in range(1, sshn.shape[0]):
        sshb[n] = sshn[n - 1] + atfp * (sshb[n - 1] - 2 * sshn[n - 1] + sshn[n])
    run = tmp_path / "run"
    run.mkdir()
    for n in range(1, nsteps + 1):
        kt = 5760 + n
        f = run / f"DINO_{kt:08d}_restart_0000.nc"
        with netCDF4.Dataset(f, "w") as ds:
            ds.createDimension("x", nx)
            ds.createDimension("y", ny)
            ds.createDimension("t", 1)
            for name, arr in (("sshn", sshn[n]),
                              ("sshb", sshb[n + offset]),
                              ("nav_lat", None), ("nav_lon", None)):
                if name.startswith("nav"):
                    v = ds.createVariable(name, "f8", ("y", "x"))
                    v[:] = np.zeros((ny, nx))
                else:
                    v = ds.createVariable(name, "f8", ("t", "y", "x"))
                    v[:] = arr[None]
            ds.createVariable("rdt", "f8")[...] = 2700.0
            ds.DOMAIN_size_global = np.array([nx, ny])
            ds.DOMAIN_position_first = np.array([1, 1])
            ds.DOMAIN_position_last = np.array([nx, ny])
            ds.DOMAIN_halo_size_start = np.array([0, 0])
            ds.DOMAIN_halo_size_end = np.array([0, 0])
    return str(run)


def test_extract_nemo_accepts_a_correctly_mapped_run(ewt, tmp_path):
    run = _fake_nemo_run(tmp_path)
    out = tmp_path / "n.npz"
    got = ewt.extract_nemo(run, 5760, 6, str(out))
    assert got["eta"].shape == (6, 5, 4)


def test_extract_nemo_rejects_a_one_dump_time_level_offset(ewt, tmp_path):
    """The synthetic violation the identity exists to catch."""
    run = _fake_nemo_run(tmp_path, offset=1)
    with pytest.raises(SystemExit, match="TIME-LEVEL IDENTITY FAILED"):
        ewt.extract_nemo(run, 5760, 6, str(tmp_path / "n2.npz"))


def test_compare_impulse_lane_differences_each_side_against_its_own_free_run(
        ewt, tmp_path):
    """The response lane must subtract like from like, and it must actually
    change the answer relative to the free lane."""
    mesh, nf, lf = _synthetic_pair(tmp_path, lego_offset=1e-6)
    # Give each side the SAME extra response on top of a DIFFERENT background,
    # so a lane that differenced against the wrong run would see the
    # background difference and a correct one sees only the (identical)
    # response.
    d_n, d_l = dict(np.load(nf)), dict(np.load(lf))
    nt, ny, nx = d_n["eta"].shape
    resp = np.zeros((nt, ny, nx))
    for n in range(nt):
        # interior columns only: a response planted on the dry rim would carry
        # no energy over wet cells and is not a response at all
        resp[n, 1:-1, 1 + (n % (nx - 2))] = 0.02
    pn, pl = tmp_path / "nemo_imp.npz", tmp_path / "lego_imp.npz"
    np.savez(pn, **{**d_n, "eta": d_n["eta"] + resp,
                    "eta_before": d_n["eta_before"] + resp})
    np.savez(pl, **{**d_l, "eta": d_l["eta"] + resp})
    res = ewt.compare(str(pn), str(pl), str(tmp_path / "imp"), mesh,
                      nf, lf)
    assert res["lane"] == "impulse_response"
    # both responses are the SAME planted field, so they agree exactly, even
    # though the two backgrounds differ by 1e-6 m
    assert res["final_max_abs_m"] < 1e-12
    free = ewt.compare(str(pn), str(pl), str(tmp_path / "fre"), mesh)
    assert free["final_max_abs_m"] > 1e-9   # the free lane sees the background


def test_compare_refuses_a_one_sided_impulse_lane(ewt, tmp_path):
    mesh, npz_n, npz_l = _synthetic_pair(tmp_path)
    with pytest.raises(SystemExit, match="BOTH free runs or neither"):
        ewt.compare(npz_n, npz_l, str(tmp_path / "o4"), mesh, npz_n, None)


def test_two_dt_mode_annihilates_a_smooth_series(ewt):
    wet, _ = _basin()
    n = 20
    smooth = np.linspace(0.0, 1.0, n)[:, None, None] * np.ones((1,) + wet.shape)
    out = ewt.two_dt_mode(smooth, wet)
    assert max(out["amplitude_by_sample_m"]) < 1e-12


def test_two_dt_mode_recovers_a_planted_alternation(ewt):
    """A field flipping sign every step must return its own amplitude."""
    wet, _ = _basin()
    n = 20
    amp = 0.004
    alt = amp * ((-1.0) ** np.arange(n))[:, None, None] * np.ones((1,) + wet.shape)
    out = ewt.two_dt_mode(alt, wet)
    # the reported number is the amplitude A of x[n] = A*(-1)**n, NOT 2A: the
    # raw second-difference operator returns -2A and is halved in the function
    assert out["first_8_mean_m"] == pytest.approx(amp)


def test_two_dt_mode_is_not_fooled_by_a_slow_oscillation(ewt):
    wet, _ = _basin()
    n = 64
    t = np.arange(n)
    slow = 0.01 * np.sin(2 * np.pi * t / 32.0)[:, None, None] * np.ones((1,) + wet.shape)
    out = ewt.two_dt_mode(slow, wet)
    assert out["first_8_mean_m"] < 0.01 * 0.05


def test_alternation_ratio_is_one_for_a_flat_series(ewt):
    assert ewt.alternation_ratio([1.0] * 8) == pytest.approx(1.0)


def test_alternation_ratio_detects_ringing(ewt):
    assert ewt.alternation_ratio([10.0, 1.0] * 4) == pytest.approx(10.0)


def test_radial_spread_tracks_a_ring_moving_outward(ewt):
    """Known-answer control: a ring at a known radius must be reported there."""
    ny, nx = 41, 41
    wet = np.ones((ny, nx), dtype=bool)
    area = np.ones((ny, nx))
    dx = dy = 10_000.0
    field = np.zeros((3, ny, nx))
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    r = np.sqrt((jj - 20) ** 2 + (ii - 20) ** 2)
    for n, want in enumerate((5.0, 10.0, 15.0)):
        field[n] = (np.abs(r - want) < 0.5).astype(float)
    got = ewt.radial_spread(field, wet, area, 20, 20, dx, dy)
    for g, want in zip(got, (5.0, 10.0, 15.0)):
        assert g == pytest.approx(want * dx / 1e3, rel=0.05)


def test_radial_spread_refuses_an_empty_sample(ewt):
    wet = np.ones((5, 5), dtype=bool)
    with pytest.raises(SystemExit, match="no energy"):
        ewt.radial_spread(np.zeros((2, 5, 5)), wet, np.ones((5, 5)),
                          2, 2, 1e4, 1e4)
