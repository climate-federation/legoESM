"""Volcanic stratospheric LONGWAVE aerosol (#9) + the B1/B2/B3 correctness fix.

The optional per-layer ``aerosol_lw_od`` is OFF by default (byte-identical
no-op) and, when supplied, materialises as a JIT-stable (ncol, nlev) field.

Fix coverage (Codex-reviewed):
- ``TestUSStdAtmPressure`` / ``TestPlanckBandWeights`` -- the z->p (geopotential-
  corrected) and Planck-weighted gray-collapse helpers.
- ``TestLoaderAbsorptionScaling`` -- B1: loader feeds ext*(1-omega), not raw
  extinction (synthetic-violation guard).
- ``TestStratosphericPlacement`` -- B2: conservative log-pressure overlap remap
  puts >80% of OD above 100 hPa vs <15% for the old pressure-mass helper.
- ``TestPlacementDifferentiability`` -- grad matches FD, jit/vmap match eager.
- ``TestExternalForcingGate`` -- the ``_ext_forcing`` gate includes LW aerosol.
- ``TestLwAerosolOlrSign`` -- solve_columns: a stratospheric LW absorption bump
  lowers OLR (correct warming sign) by a small amount.
- ``TestRealFileEndToEnd`` -- the real 1979 file lands in the stratosphere.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


class TestVolcanicLwDefaultOff(unittest.TestCase):
    """The loader is OFF by default => no LW aerosol => no-op."""

    def test_get_aerosol_lw_disabled_returns_none(self):
        from legoesm.forcing.external import AerosolConfig, get_aerosol_lw_at_time
        cfg = AerosolConfig()  # volcanic_lw_enabled defaults False
        self.assertFalse(cfg.volcanic_lw_enabled)
        out = get_aerosol_lw_at_time(cfg, day=0.0, lat_grid=np.zeros(4))
        self.assertIsNone(out)

    def test_get_aerosol_lw_enabled_without_path_returns_none(self):
        from legoesm.forcing.external import AerosolConfig, get_aerosol_lw_at_time
        cfg = AerosolConfig(volcanic_lw_enabled=True)  # no volcanic_path
        out = get_aerosol_lw_at_time(cfg, day=0.0, lat_grid=np.zeros(4))
        self.assertIsNone(out)


class TestPackForcingLwField(unittest.TestCase):
    """pack_forcing materialises aerosol_lw_od as a JIT-stable concrete array."""

    def _pack(self, aerosol_lw_od):
        from legoesm.driver.compiled_segments import pack_forcing
        shp = (6, 4, 4)
        aer = jnp.zeros((6 * 4 * 4, 5))  # (ncol, nlev) per-layer SW aerosol od
        return pack_forcing(
            sst=jnp.full(shp, 290.0), sic=jnp.zeros(shp),
            day_of_year=1, seconds_of_day=0.0,
            solar_weights=jnp.ones(1), s_0=1361.0,
            o3_vmr=jnp.zeros((6 * 4 * 4, 5)), aerosol_od=aer,
            aerosol_lw_od=aerosol_lw_od,
        )

    def test_none_materialises_zeros_same_shape(self):
        f = self._pack(None)
        self.assertEqual(f.aerosol_lw_od.shape, f.aerosol_od.shape)
        self.assertTrue(jnp.all(f.aerosol_lw_od == 0.0))

    def test_supplied_value_preserved(self):
        lw = jnp.full((6 * 4 * 4, 5), 0.1)
        f = self._pack(lw)
        self.assertEqual(f.aerosol_lw_od.shape, (6 * 4 * 4, 5))
        self.assertTrue(jnp.allclose(f.aerosol_lw_od, 0.1))


# ---------------------------------------------------------------------------
# Fix B1/B2/B3: extinction->absorption, stratospheric placement, band collapse
# ---------------------------------------------------------------------------

_REAL_VOLC_LW = Path(
    "/pool/data/ICON/grids/public/mpim/common/aerosol_volcanic_cmip6/"
    "bc_aeropt_cmip6_volc_lw_b16_sw_b14_1979.nc"
)


def _make_synth_volc_lw(path, omega=0.3, ext_val=1.0e-3, nband=16,
                        nlat=4, nalt=6, nmonth=12, alt_lo=15.0, alt_hi=25.0):
    """Write a minimal CMIP6-shaped volcanic LW file for loader tests."""
    import xarray as xr
    alt = np.linspace(alt_lo, alt_hi, nalt)          # km, stratospheric
    lat = np.linspace(-60.0, 60.0, nlat)
    month = np.arange(1, nmonth + 1)
    ev = np.asarray(ext_val, dtype=np.float64)
    if ev.ndim == 0:                                 # scalar -> band-uniform
        ext = np.full((nband, nlat, nalt, nmonth), float(ev))
    else:                                            # per-band profile
        ext = np.broadcast_to(
            ev[:, None, None, None], (nband, nlat, nalt, nmonth)).copy()
    om = np.full_like(ext, omega)
    wl2 = np.linspace(250.0, 4.0, nband)             # upper wavelength [um]
    wl1 = wl2 * 0.8
    ds = xr.Dataset(
        {
            "ext_earth": (("terrestrial_bands", "latitude", "altitude",
                           "month"), ext, {"units": "1/km"}),
            "omega_earth": (("terrestrial_bands", "latitude", "altitude",
                             "month"), om),
            "wl1_earth": (("terrestrial_bands",), wl1),
            "wl2_earth": (("terrestrial_bands",), wl2),
        },
        coords={"terrestrial_bands": np.arange(nband), "latitude": lat,
                "altitude": alt, "month": month},
    )
    ds["altitude"].attrs["units"] = "km"
    ds.to_netcdf(path)


import functools


@functools.lru_cache(maxsize=1)
def _lw_solver():
    """RRTMGP solver with default optics tables (built once for the OLR tests)."""
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    return RRTMGP.from_legoesm_config(RRTMGPConfig())


@functools.lru_cache(maxsize=1)
def _tropical_column(nlev=40, p_sfc=101300.0, sfc_T=300.0):
    """Solve-columns kwargs for a single standard tropical column."""
    sig = np.linspace(0.0, 1.0, nlev + 1) ** 1.1
    p_half = 100.0 + (p_sfc - 100.0) * sig
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    z = -7000.0 * np.log(p_full / p_sfc)
    t_iso = sfc_T - 6.5e-3 * (-7000.0 * np.log(1.0e4 / p_sfc))
    temp = np.clip(np.where(p_full > 1.0e4, sfc_T - 6.5e-3 * z, t_iso),
                   195.0, 305.0)
    q = np.clip(0.018 * (p_full / p_sfc) ** 3, 1e-6, None)
    return dict(
        T=jnp.asarray(temp[None, :]), p_full=jnp.asarray(p_full[None, :]),
        p_half=jnp.asarray(p_half[None, :]),
        sfc_temperature=jnp.asarray([sfc_T]),
        q_v=jnp.asarray(q[None, :]), cos_zenith=jnp.asarray([0.5]),
    ), p_full


def _olr(solver, kw, aer_lw=None):
    out = solver.solve_columns(aerosol_absorption_optical_depth_lw=aer_lw, **kw)
    return float(np.asarray(out.lw_flux_up)[0, 0])   # TOA upward LW [W/m^2]


class TestUSStdAtmPressure(unittest.TestCase):
    """US Std Atm 1976 z->p is monotone and matches published base pressures.

    The helper takes GEOMETRIC altitude and converts to geopotential internally,
    so the published GEOPOTENTIAL base pressures are recovered by passing the
    geometric heights that map to the geopotential bases.
    """

    def test_known_values_and_monotonic(self):
        from legoesm import constants
        from legoesm.forcing.external import _ussa1976_pressure
        r_e = constants.R_earth
        h_bases = np.array([0.0, 11000.0, 20000.0, 32000.0])   # geopotential [m]
        z_geom = r_e * h_bases / (r_e - h_bases)               # -> geometric [m]
        p = _ussa1976_pressure(z_geom)
        # Published US Std Atm 1976 base pressures [Pa].
        for got, want in zip(p, (101325.0, 22632.0, 5474.9, 868.0)):
            self.assertAlmostEqual(got, want, delta=0.01 * want)
        zc = np.linspace(0.0, 40000.0, 50)
        self.assertTrue(np.all(np.diff(_ussa1976_pressure(zc)) < 0.0))
        # The geopotential correction is applied: at geometric 32 km the mapped
        # pressure exceeds the uncorrected power-law value by ~2%.
        self.assertGreater(float(_ussa1976_pressure(np.array([32000.0]))[0]),
                           880.0)


class TestPlanckBandWeights(unittest.TestCase):
    """B3: the gray collapse weights emission-bearing LW bands, not a flat mean."""

    def test_far_ir_dominates_near_ir_at_strat_temp(self):
        from legoesm.forcing.external import _planck_band_weights
        # Two bands: far-IR (15-50 um) and near-IR (3-4 um) at 220 K.
        w = _planck_band_weights(np.array([15.0, 3.0]),
                                 np.array([50.0, 4.0]), 220.0)
        self.assertGreater(w[0], 100.0 * w[1])   # near-IR ~ 0 at 220 K
        # A flat mean would give equal weights; Planck must not.
        self.assertNotAlmostEqual(w[0] / w.sum(), 0.5, places=2)


class TestLoaderAbsorptionScaling(unittest.TestCase):
    """B1: the loader feeds ABSORPTION = ext*(1-omega), never raw extinction."""

    def test_one_minus_omega_applied(self):
        import tempfile
        from legoesm.forcing.external import _load_volcanic_lw_absorption_profile
        with tempfile.TemporaryDirectory() as d:
            p0 = str(Path(d) / "omega0.nc")
            p3 = str(Path(d) / "omega03.nc")
            _make_synth_volc_lw(p0, omega=0.0)
            _make_synth_volc_lw(p3, omega=0.3)
            _, _, _, prof0, _ = _load_volcanic_lw_absorption_profile(p0)
            _, _, _, prof3, _ = _load_volcanic_lw_absorption_profile(p3)
            col0 = prof0.sum(axis=-1)          # extinction column (omega=0)
            col3 = prof3.sum(axis=-1)          # absorption column (omega=0.3)
            # Synthetic-violation guard: feeding raw extinction (dropping the
            # (1-omega) scale) would make col3 == col0.  It must be 0.7*col0.
            self.assertTrue(np.allclose(col3, 0.7 * col0, rtol=1e-6))
            self.assertTrue(np.all(col3 < col0 - 1e-12))

    def test_loader_uses_planck_not_flat_band_mean(self):
        """B3 regression: with band-VARYING extinction the loader's gray column
        must equal the Planck-weighted collapse, not the flat band mean.  Swaps
        would be caught (band-uniform fixtures cannot distinguish them)."""
        import tempfile
        from legoesm.forcing.external import (
            _load_volcanic_lw_absorption_profile, _planck_band_weights,
            _VOLC_LW_PLANCK_TEMP_K,
        )
        nband = 16
        ext = np.zeros(nband)
        ext[0] = 2.0e-3                       # only far-IR band 0 carries OD
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "bandvary.nc")
            # nalt=6 over 15-25 km => 2 km spacing => 12 km total thickness.
            _make_synth_volc_lw(p, omega=0.0, ext_val=ext, nband=nband,
                                nalt=6, alt_lo=15.0, alt_hi=25.0)
            _, _, _, prof, _ = _load_volcanic_lw_absorption_profile(p)
        wl2 = np.linspace(250.0, 4.0, nband)  # must match the builder
        wl1 = wl2 * 0.8
        w = _planck_band_weights(wl1, wl2, _VOLC_LW_PLANCK_TEMP_K)
        thickness_km = 12.0
        col_planck = w[0] * 2.0e-3 / w.sum() * thickness_km
        col_flat = (2.0e-3 / nband) * thickness_km
        col = float(prof.sum(-1)[0, 0])       # (time0, lat0) column OD
        self.assertAlmostEqual(col, col_planck, delta=1e-4 * col_planck)
        self.assertGreater(abs(col - col_flat), 0.2 * col_flat)  # clearly != flat

    def test_p_edges_ascending_and_aligned(self):
        import tempfile
        from legoesm.forcing.external import _load_volcanic_lw_absorption_profile
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "v.nc")
            _make_synth_volc_lw(p, omega=0.1, nalt=6)
            _, _, _, prof, p_edges = _load_volcanic_lw_absorption_profile(p)
            self.assertEqual(p_edges.shape[0], prof.shape[-1] + 1)
            self.assertTrue(np.all(np.diff(p_edges) > 0.0))   # ascending Pa


class TestStratosphericPlacement(unittest.TestCase):
    """B2: profile is placed at its true pressure, conserving column OD."""

    @staticmethod
    def _standard_column(ncol=3, nlev=30, p_top=100.0, p_sfc=101325.0):
        sig = np.linspace(0.0, 1.0, nlev + 1) ** 1.2
        p_half = p_top + (p_sfc - p_top) * sig      # (nlev+1,), ascending
        return jnp.tile(jnp.asarray(p_half)[None, :], (ncol, 1))

    def test_mass_above_100hpa_and_conserved(self):
        from legoesm.forcing.surface_utils import (
            place_stratospheric_aod_profile_to_layers,
        )
        # Aerosol between 10 hPa (1000 Pa) and 90 hPa (9000 Pa): entirely
        # stratospheric (above 100 hPa).
        p_edges = jnp.asarray(np.linspace(1000.0, 9000.0, 7))
        total = 0.01
        aod_profile = jnp.full((3, 6), total / 6.0)
        p_half = self._standard_column()
        placed = place_stratospheric_aod_profile_to_layers(
            aod_profile, p_edges, p_half,
        )
        # Conservation: the column spans the aerosol range fully.
        self.assertTrue(np.allclose(np.asarray(placed.sum(axis=1)), total,
                                    rtol=1e-6))
        # >80% of OD mass sits above 100 hPa (p < 1e4 Pa).
        p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        frac_strat = np.asarray(
            (placed * (p_mid < 1.0e4)).sum(axis=1) / placed.sum(axis=1)
        )
        self.assertTrue(np.all(frac_strat > 0.80))

    def test_log_pressure_within_layer_split(self):
        """A model interface cutting a source layer at its LOG-pressure midpoint
        splits the OD 50/50 (source OD uniform in geometric height ~ log-p).
        A reverted LINEAR-pressure CDF would give [0.4142, 0.5858] and fail."""
        from legoesm.forcing.surface_utils import (
            place_stratospheric_aod_profile_to_layers,
        )
        p_edges = jnp.asarray([1000.0, 2000.0])          # one source layer
        aod = jnp.asarray([[1.0]])
        p_cut = float(np.sqrt(1000.0 * 2000.0))          # log-p midpoint
        p_half = jnp.asarray([[1000.0, p_cut, 2000.0]])  # two model layers
        placed = np.asarray(
            place_stratospheric_aod_profile_to_layers(aod, p_edges, p_half)
        )
        self.assertTrue(np.allclose(placed[0], [0.5, 0.5], atol=1e-6),
                        f"log-p split expected [0.5,0.5], got {placed[0]}")

    def test_versus_pressure_mass_spreading(self):
        """The OLD helper dumps the same column AOD mostly in the troposphere."""
        from legoesm.forcing.surface_utils import (
            distribute_column_aod_to_layers,
        )
        p_half = self._standard_column()
        placed_old = distribute_column_aod_to_layers(
            jnp.full((3,), 0.01), p_half,
        )
        p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        frac_strat_old = np.asarray(
            (placed_old * (p_mid < 1.0e4)).sum(axis=1) / placed_old.sum(axis=1)
        )
        # Confirms the defect the new helper fixes: <15% stays stratospheric.
        self.assertTrue(np.all(frac_strat_old < 0.15))


class TestPlacementDifferentiability(unittest.TestCase):
    """Placement is differentiable and JIT/vmap-consistent (LegoESM AD goal)."""

    def _setup(self):
        p_edges = jnp.asarray(np.linspace(1000.0, 9000.0, 7))
        aod = jnp.asarray(np.linspace(0.001, 0.003, 6))[None, :]
        p_half = TestStratosphericPlacement._standard_column(ncol=1, nlev=24)
        return p_edges, aod, p_half

    def test_grad_matches_finite_difference(self):
        from legoesm.forcing.surface_utils import (
            place_stratospheric_aod_profile_to_layers,
        )
        p_edges, aod, p_half = self._setup()

        def scalar(ph):
            placed = place_stratospheric_aod_profile_to_layers(aod, p_edges, ph)
            # weight by layer index so gradient w.r.t. edges is non-trivial
            wts = jnp.arange(placed.shape[1], dtype=placed.dtype)
            return jnp.sum(placed * wts)

        g = jax.grad(scalar)(p_half)
        self.assertFalse(bool(jnp.any(jnp.isnan(g))))
        # Centered FD on a mid interface (perturb one half-level).
        eps = 1.0
        k = 12
        ph_p = p_half.at[0, k].add(eps)
        ph_m = p_half.at[0, k].add(-eps)
        fd = (scalar(ph_p) - scalar(ph_m)) / (2 * eps)
        self.assertAlmostEqual(float(g[0, k]), float(fd), delta=1e-4)

    def test_jit_and_vmap_match(self):
        from legoesm.forcing.surface_utils import (
            place_stratospheric_aod_profile_to_layers,
        )
        p_edges, aod, p_half = self._setup()
        eager = place_stratospheric_aod_profile_to_layers(aod, p_edges, p_half)
        jitted = jax.jit(place_stratospheric_aod_profile_to_layers)(
            aod, p_edges, p_half,
        )
        self.assertTrue(np.allclose(np.asarray(eager), np.asarray(jitted)))
        # vmap over an ensemble axis on the two per-column args.
        aod_e = jnp.stack([aod, 2.0 * aod])
        ph_e = jnp.stack([p_half, p_half])
        vm = jax.vmap(
            lambda a, ph: place_stratospheric_aod_profile_to_layers(
                a, p_edges, ph),
        )(aod_e, ph_e)
        self.assertTrue(np.allclose(np.asarray(vm[0]), np.asarray(eager)))
        self.assertTrue(np.allclose(np.asarray(vm[1]), 2.0 * np.asarray(eager)))


@unittest.skipUnless(_REAL_VOLC_LW.exists(), "CMIP6 volcanic LW file absent")
class TestRealFileEndToEnd(unittest.TestCase):
    """End-to-end on the real 1979 file: the LW aerosol lands in the
    stratosphere.  On a deep-TROPICAL column (100 hPa ~ tropopause) >80% of
    the OD is above 100 hPa; at every latitude the OD-weighted centroid is
    stratospheric (the extratropical aerosol tracks the lower tropopause, so a
    fixed 100 hPa line -- a tropical proxy -- correctly lets some 30N OD sit
    between 100-180 hPa, still above the local tropopause)."""

    def _load(self, lats_deg):
        from legoesm.forcing.external import (
            AerosolConfig, get_aerosol_lw_at_time,
        )
        cfg = AerosolConfig(
            volcanic_lw_enabled=True, volcanic_path=str(_REAL_VOLC_LW),
            volcanic_scale=1.0,
        )
        out = get_aerosol_lw_at_time(
            cfg, day=196.0, lat_grid=jnp.asarray(np.radians(lats_deg)),
        )
        self.assertIsNotNone(out)
        return out

    def test_tropical_column_above_100hpa(self):
        from legoesm.forcing.surface_utils import (
            place_stratospheric_aod_profile_to_layers,
        )
        prof_col, p_edges = self._load([0.0, 5.0, -5.0])
        self.assertGreater(float(prof_col.sum()), 0.0)
        p_half = TestStratosphericPlacement._standard_column(ncol=3, nlev=40)
        placed = place_stratospheric_aod_profile_to_layers(
            prof_col, p_edges, p_half,
        )
        p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        frac = np.asarray(
            (placed * (p_mid < 1.0e4)).sum(axis=1) / placed.sum(axis=1)
        )
        self.assertTrue(np.all(frac > 0.80), f"tropical frac>100hPa={frac}")

    def test_all_latitude_centroid_is_stratospheric(self):
        prof_col, p_edges = self._load(np.linspace(-80.0, 80.0, 9))
        pe = np.asarray(p_edges)
        p_mid = 0.5 * (pe[:-1] + pe[1:])
        prof = np.asarray(prof_col)
        centroid = (prof * p_mid[None, :]).sum(1) / prof.sum(1)   # [Pa]
        # Stratospheric centroid at every latitude (< 150 hPa).
        self.assertTrue(np.all(centroid < 1.5e4), f"centroid[Pa]={centroid}")

    def test_loader_to_solver_lowers_olr(self):
        """Full chain: file -> get_aerosol_lw_at_time -> remap -> solve_columns.
        The correctly-placed 1979 absorption lowers TOA OLR (warming) by a
        small but non-zero amount (background year)."""
        from legoesm.forcing.surface_utils import (
            place_stratospheric_aod_profile_to_layers,
        )
        prof_col, p_edges = self._load([0.0])          # equatorial column
        solver = _lw_solver()
        kw, _ = _tropical_column()
        placed = place_stratospheric_aod_profile_to_layers(
            prof_col, p_edges, kw["p_half"],
        ).astype(kw["T"].dtype)
        olr_base = _olr(solver, kw)
        olr_pert = _olr(solver, kw, placed)
        self.assertLess(olr_pert, olr_base)            # correct warming sign
        self.assertGreater(olr_base - olr_pert, 0.005)  # small but resolved


class TestExternalForcingGate(unittest.TestCase):
    """Regression: the _ext_forcing gate opens for a volcanic-LW-only run."""

    def test_lw_only_activates_and_scheme_gated(self):
        from legoesm.driver.model_driver import _external_forcing_active
        # Only LW aerosol active (no ozone / SW aerosol / GHG / experiment).
        self.assertTrue(
            _external_forcing_active(True, False, False, True, False, False))
        # No gas-radiation scheme => no forcing dict regardless.
        self.assertFalse(
            _external_forcing_active(False, False, False, True, False, False))
        # Nothing active => off.
        self.assertFalse(
            _external_forcing_active(True, False, False, False, False, False))


class TestLwAerosolOlrSign(unittest.TestCase):
    """solve_columns: a stratospheric LW ABSORPTION bump lowers OLR (warming).

    Pins the absorption-slot sign convention end-to-end -- the physics the B1
    (ext->abs) fix relies on: an LW absorber aloft re-emits at a colder
    temperature than it intercepts, so TOA OLR drops."""

    def test_stratospheric_absorption_lowers_olr(self):
        solver = _lw_solver()
        kw, p_full = _tropical_column()
        aer = np.where(p_full < 1.0e4, 0.02, 0.0)[None, :]
        aer = jnp.asarray(aer).astype(kw["T"].dtype)
        olr_base = _olr(solver, kw)
        olr_pert = _olr(solver, kw, aer)
        self.assertLess(olr_pert, olr_base)          # absorber aloft -> warming
        self.assertGreater(olr_base - olr_pert, 0.1)  # non-trivial at OD 0.02


if __name__ == "__main__":
    unittest.main()
