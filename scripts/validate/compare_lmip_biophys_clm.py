"""LMIP biophys (2 deg, calibrated) vs CLM5.1-SP — manuscript figures.

Compares the calibrated 2-deg CRU-JRA biophysics LMIP run against the
CTSM5.1-dev land-only simulation driven by the same CRU-JRA forcing
(ctsm51d142_f19_CRUjra_hist_flds).  Produces two manuscript figures:

* ``fig_lmip_lh_gpp``         — annual-mean latent-heat maps side by side
                                 (legoESM, CLM) + zonal SUM of GPP
                                 (PgC yr-1 deg-1; area under the curve =
                                 global total) with each obs product as its
                                 own line
* ``fig_lmip_seasonal_biome`` — biome-mean seasonal cycles of latent heat and
                                 GPP, composited by local season (SH cells
                                 shifted 6 months; model area-weighted IQR
                                 shading; one line per obs product)

NO regridding anywhere: every dataset (legoESM 2 deg, CLM f19, obs 0.5 deg)
is analysed on its NATIVE grid with its own land-area weights (cell area x
land fraction; obs weighted by cell area where data exist).  Zonal sums in
PgC yr-1 per degree latitude and area-weighted biome means are resolution-
independent, so native-grid curves are directly comparable.  The shared
biome classification (17 CLM5 PFTs -> 8 classes, >50% dominance,
glacier/lake excluded) is built once on the surfdata grid and nearest-
neighbour sampled to each dataset's grid — categorical, so no flux
regridding is involved.

Observations are ILAMB copies of FLUXCOM and GBAF; each product's
climatology is taken over its record intersected with the analysis period.
A physical-validity guard drops any numerically-diverged legoESM cells
(unphysical fluxes) from the legoESM fields.

Monthly climatologies are cached as netcdf in ``<outdir>/cache`` so re-runs
are fast; use ``--force`` to rebuild.

Requires xarray + netCDF4 + matplotlib + legoesm (for the shared physical
constants — any legoESM conda env has all four).  On derecho/casper::

  /glade/work/linnia/conda-envs/legoesm-lmip/bin/python \
      scripts/validate/compare_lmip_biophys_clm.py
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import xarray as xr

# --------------------------------------------------------------------------
# Defaults (override on the command line)
# --------------------------------------------------------------------------
LEGO_DIR = '/glade/derecho/scratch/linnia/lmip/lmip_biophys_1985_calib'
CLM_DIR = ('/glade/campaign/cgd/tss/common/Land_Only_Simulations/CTSM51_DEV/'
           'ctsm51d142_f19_CRUjra_hist_flds/lnd/hist')
SURFDATA = '/glade/work/linnia/legoESM/data/legoesm_surfdata_c260716.nc'
ILAMB = '/glade/campaign/cesm/community/lmwg/diag/ILAMB/DATA'
YEAR0, YEAR1 = 2005, 2014        # obs-covered decade

# quantity -> [(product, path, varname, units), ...]
OBS_SOURCES = {
    'GPP': [('FLUXCOM', f'{ILAMB}/gpp/FLUXCOM/gpp.nc',      'gpp', 'g/m2/day'),
            ('GBAF',    f'{ILAMB}/gpp/GBAF/gpp_0.5x0.5.nc', 'gpp', 'kg/m2/s')],
    'LH':  [('FLUXCOM', f'{ILAMB}/le/FLUXCOM/le.nc',        'le',  'W/m2'),
            ('GBAF',    f'{ILAMB}/le/GBAF/le_0.5x0.5.nc',   'le',  'W/m2')],
}

# --------------------------------------------------------------------------
# Constants / unit conversions (physical constants from legoesm.constants —
# same values the model ran with; note R_earth = 6371.229 km, the model's
# CESM-convention radius, not the generic 6371 km)
# --------------------------------------------------------------------------
from legoesm import constants  # noqa: E402

MW_C = 12.011                            # g C per mol (molar mass, unit conv)
UMOL2G_DAY = MW_C * 1e-6 * 86400.0       # umol/m2/s -> gC/m2/day (FPSN -> GPP)
LV = constants.L_v                       # J/kg latent heat of vaporisation
WM2_TO_MMDAY = 86400.0 / LV              # 1 W/m2 -> mm/day (~0.03455)
R_EARTH = constants.R_earth              # m
DAYS_YR = 365.0                          # noleap calendar year length

# CLM5 PFT indices (17 incl. bare) aggregated into 8 biome classes —
# identical to the LEAP/lego_analysis mask so figures are comparable.
BIOME_PFTS = {
    'tropical_broadleaf': [4, 6],
    'temperate_forest':   [1, 5, 7],
    'boreal_needleleaf':  [2, 3, 8],
    'grassland':          [13, 14],
    'savanna_shrub':      [9, 10, 11],
    'cropland':           [15, 16],
    'tundra':             [12],
    'arid':               [0],
}
BIOMES = list(BIOME_PFTS)

# Peer-model styling (colourblind-safe; also distinct in grayscale)
C_LEGO, C_CLM = '#0072B2', '#D55E00'
LS_LEGO, LS_CLM = '-', (0, (4, 2))
# obs products: one grey line each, distinct linestyles
OBS_STYLE = {'FLUXCOM': ('0.35', '-'), 'GBAF': ('0.55', (0, (1, 1.5)))}
MM = 1 / 25.4
COL1, COL2 = 89 * MM, 183 * MM           # single/double column widths (in)


# ==========================================================================
# Grid helpers
# ==========================================================================
def _bounds_1d(centers):
    """Cell edges (n+1) from monotonic 1-D centers, lat clipped to +/-90."""
    c = np.asarray(centers, dtype=float)
    mid = 0.5 * (c[:-1] + c[1:])
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def cell_area(lat, lon):
    """Area (m2) of each cell on a regular lat/lon grid."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    lat_b = np.deg2rad(np.clip(_bounds_1d(lat), -90.0, 90.0))
    dlon = np.deg2rad(np.abs(lon[1] - lon[0]))
    dsin = np.abs(np.sin(lat_b[1:]) - np.sin(lat_b[:-1]))   # abs: handles
    area = (R_EARTH ** 2) * dlon * dsin                     # descending lat
    return xr.DataArray(np.repeat(area[:, None], lon.size, axis=1),
                        coords={'lat': lat, 'lon': lon}, dims=('lat', 'lon'))


def wmean(da, w, dims=('lat', 'lon')):
    return da.weighted(w.fillna(0)).mean(dim=dims)


def open_concat(files, keep, **kw):
    """Eagerly open+concat monthly files along time (no dask needed)."""
    return xr.concat([xr.open_dataset(f, **kw)[keep] for f in files],
                     dim='time', join='override', coords='minimal',
                     compat='override')


def to_0360(ds):
    """Roll a -180..180 longitude dataset to 0..360, sorted."""
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')
    return ds


def zonal_gpp(gpp_clim, w):
    """(month,lat,lon) GPP in gC/m2/day + weights (m2) -> PgC/yr per degree
    latitude on the dataset's own latitude axis.  Area under the curve =
    global total, so curves from different grids are directly comparable."""
    dlat = float(abs(gpp_clim.lat[1] - gpp_clim.lat[0]))
    ann = gpp_clim.mean('month')                       # gC/m2/day
    per_cell = ann.fillna(0) * w.fillna(0) * DAYS_YR / 1e15   # PgC/yr per cell
    return per_cell.sum('lon') / dlat                  # PgC/yr/deg


def global_total(zonal_per_deg):
    dlat = float(abs(zonal_per_deg.lat[1] - zonal_per_deg.lat[0]))
    return float((zonal_per_deg * dlat).sum())


# ==========================================================================
# Climatologies (cached) — each dataset on its NATIVE grid
# ==========================================================================
class Data:
    """Loads/caches everything the figures need.  No regridding: each
    dataset keeps its native grid and carries its own land-area weights."""

    def __init__(self, lego_dir, clm_dir, surfdata, y0, y1, outdir, force):
        self.lego_dir, self.clm_dir, self.surfdata = lego_dir, clm_dir, surfdata
        self.y0, self.y1 = y0, y1
        self.nmon = (y1 - y0 + 1) * 12
        self.cache = os.path.join(outdir, 'cache')
        os.makedirs(self.cache, exist_ok=True)
        self.force = force
        self.tag = f'{y0}-{y1}'         # cache-file suffix

        # -- legoESM: native 2 deg; weights = cell area x land_fraction
        first = os.path.join(lego_dir, f'lmip_biophys.monthly.{y0}.nc')
        lf = xr.open_dataset(first, decode_timedelta=True)['land_fraction']
        if 'time' in lf.dims:
            lf = lf.isel(time=0, drop=True)
        self.lego = self._lego_clim()
        self.lego_lf = lf
        self.lego_w = (cell_area(lf.lat.values, lf.lon.values) * lf
                       ).rename('lego_weight')

        # physical-validity guard: drop numerically-diverged lego cells
        phys = ((np.abs(self.lego['LH']) < 1e3) &
                (self.lego['GPP'] >= -0.1) &
                (self.lego['GPP'] < 30)).all('month')
        n_bad = int(((~phys) & (self.lego_w > 0)
                     & self.lego['LH'].notnull().all('month')).sum())
        if n_bad:
            print(f'  [excluded {n_bad} numerically-diverged lego cells]')
        self.lego = self.lego.where(phys)
        self.lego_w = self.lego_w.where(phys, 0.0)

        # -- CLM: native f19; weights = cell area x landfrac (from history)
        clm0 = sorted(glob.glob(os.path.join(clm_dir, f'*.h0.{y0}-*.nc')))[0]
        ds0 = xr.open_dataset(clm0)
        clm_lf = ds0['landfrac'].fillna(0.0)
        self.clm_lf = clm_lf
        self.clm_w = (ds0['area'].fillna(0.0) * 1e6 * clm_lf
                      ).rename('clm_weight')            # km2 -> m2
        self.clm = self._clm_clim().where(clm_lf > 0)

        self._biome_src = None          # surfdata-grid biome map (lazy)

    # -- legoESM ----------------------------------------------------------
    def _lego_clim(self):
        fn = os.path.join(self.cache, f'lego_clim_{self.tag}.nc')
        if os.path.exists(fn) and not self.force:
            return xr.open_dataset(fn)
        print(f'[building legoESM climatology {self.y0}-{self.y1} ...]')
        files = [os.path.join(self.lego_dir, f'lmip_biophys.monthly.{y}.nc')
                 for y in range(self.y0, self.y1 + 1)]
        ds = open_concat(files, ['GPP', 'lhflx'], decode_timedelta=True)
        ds['time'] = xr.cftime_range(str(self.y0), periods=self.nmon,
                                     freq='MS', calendar='noleap')
        out = ds[['GPP', 'lhflx']].groupby('time.month').mean('time')
        out = out.rename({'lhflx': 'LH'}).load()        # GPP gC/m2/day; LH W/m2
        out.attrs['desc'] = (f'legoESM lmip_biophys monthly climatology '
                             f'{self.y0}-{self.y1} (native 2 deg)')
        out.to_netcdf(fn)
        return out

    # -- CLM --------------------------------------------------------------
    def _clm_clim(self):
        fn = os.path.join(self.cache, f'clm_clim_native_{self.tag}.nc')
        if os.path.exists(fn) and not self.force:
            return xr.open_dataset(fn)
        print(f'[building CLM climatology {self.y0}-{self.y1} (native f19) ...]')
        files = []
        for y in range(self.y0, self.y1 + 1):
            files += sorted(glob.glob(os.path.join(self.clm_dir,
                                                   f'*.h0.{y}-*.nc')))
        assert len(files) == self.nmon, \
            f'expected {self.nmon} CLM files, found {len(files)}'
        ds = open_concat(files, ['FPSN', 'EFLX_LH_TOT'],
                         decode_timedelta=True)
        # CLM h0 stamps end-of-month; replace with a clean monthly index
        ds['time'] = xr.cftime_range(str(self.y0), periods=self.nmon,
                                     freq='MS', calendar='noleap')
        clim = ds.groupby('time.month').mean('time').load()
        out = xr.Dataset()
        out['GPP'] = clim['FPSN'] * UMOL2G_DAY          # gC/m2/day
        out['LH'] = clim['EFLX_LH_TOT']                 # W/m2
        out.attrs['desc'] = (f'CLM5.1-SP (ctsm51d142_f19_CRUjra) monthly '
                             f'climatology {self.y0}-{self.y1} (native f19)')
        out.to_netcdf(fn)
        return out

    # -- observations -----------------------------------------------------
    def obs_clim(self, var):
        """{product: DataArray(month,lat,lon)} on each product's NATIVE
        grid.  GPP in gC/m2/day; LH in W/m2.  Climatology over each
        product's record intersected with the analysis period."""
        # products live on different grids -> one cache file per product
        def cfn(prod):
            return os.path.join(self.cache,
                                f'obs_{var}_{prod}_clim_native_{self.tag}.nc')
        if all(os.path.exists(cfn(p)) for p, *_ in OBS_SOURCES[var]) \
                and not self.force:
            return {p: xr.open_dataset(cfn(p))[p]
                    for p, *_ in OBS_SOURCES[var]}
        print(f'[building {var} obs climatologies ...]')
        out = {}
        for prod, path, vname, unit in OBS_SOURCES[var]:
            ds = to_0360(xr.open_dataset(path, decode_times=True)[[vname]])
            ds = ds.sel(time=slice(f'{self.y0}-01-01', f'{self.y1}-12-31'))
            da = ds[vname]
            # conflicting _FillValue/missing_value metadata in these files
            # stops xarray from masking fills -> mask them explicitly
            mv = da.attrs.get('missing_value', -9999.0)
            da = da.where((da != mv) & (np.abs(da) < 1e4))
            clim = da.groupby('time.month').mean('time').load()
            clim = clim.sortby('lat')      # some products store lat N->S
            if unit == 'kg/m2/s':
                clim = clim * 1000.0 * 86400.0     # kgC/m2/s -> gC/m2/day
            out[prod] = clim.rename(prod)
            save = out[prod].to_dataset(name=prod)
            for v in save.variables:   # source files carry conflicting
                save[v].encoding = {}  # _FillValue/missing_value metadata
                save[v].attrs.pop('missing_value', None)
            save.to_netcdf(cfn(prod))
        return out

    def obs_weight(self, da):
        """Cell area (m2) where the product has data in every month."""
        a = cell_area(da.lat.values, da.lon.values)
        return a.where(da.notnull().all('month'), 0.0)

    # -- biome classification --------------------------------------------
    def biome_on(self, lat, lon):
        """Dominant-biome class nearest-sampled onto an arbitrary grid.
        -1 = unclassified (mixed / glacier / lake / no land)."""
        if self._biome_src is None:
            self._biome_src = self._build_biome_src()
        b = self._biome_src.astype('float32').interp(
            lat=lat, lon=lon, method='nearest',
            kwargs={'fill_value': -1.0})
        return b.fillna(-1).astype('int8')

    def _build_biome_src(self):
        fn = os.path.join(self.cache, 'biome_surfgrid.nc')
        if os.path.exists(fn) and not self.force:
            return xr.open_dataset(fn)['biome']
        print('[building biome classification on the surfdata grid ...]')
        sd = to_0360(xr.open_dataset(self.surfdata))
        pft = sd['pft_frac'].isel(year=0)
        total = pft.sum('npft')
        cls = xr.concat(
            [100.0 * pft.isel(npft=i).sum('npft') / total.where(total > 0)
             for i in BIOME_PFTS.values()], dim='biome')
        cls = cls.assign_coords(biome=np.arange(len(BIOMES)))
        dom_frac = cls.max('biome')
        dom_idx = cls.fillna(-1).argmax('biome')
        valid = ((dom_frac > 50.0) & (sd['f_land'].isel(year=0) > 0) &
                 (sd['f_glacier'].isel(year=0) < 0.5) &
                 (sd['f_lake'].isel(year=0) < 0.5))
        biome = xr.where(valid, dom_idx, -1).astype('int8').rename('biome')
        biome.attrs['classes'] = ';'.join(f'{i}:{b}'
                                          for i, b in enumerate(BIOMES))
        biome.to_dataset(name='biome').to_netcdf(fn)
        return biome


# ==========================================================================
# Figure styling
# ==========================================================================
def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'savefig.dpi': 300, 'figure.dpi': 120,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def panel_letter(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=8,
            fontweight='bold', ha='right', va='bottom')


def savefig(fig, outdir, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'{name}.{ext}'), bbox_inches='tight')
    print(f'  saved {name}.pdf/.png -> {outdir}')


# ==========================================================================
# Figures
# ==========================================================================
def fig_lh_gpp(D, outdir):
    """1x3: annual-mean latent-heat maps (legoESM, CLM; each on its native
    grid, shared 'Blues' scale, darker = more LH) + zonal SUM of GPP
    (PgC/yr per degree latitude; area under curve = global total) with one
    line per obs product."""
    import matplotlib.pyplot as plt

    lh_l = D.lego['LH'].mean('month')
    lh_c = D.clm['LH'].mean('month')

    zl = zonal_gpp(D.lego['GPP'], D.lego_w)
    zc = zonal_gpp(D.clm['GPP'], D.clm_w)
    tl, tc = global_total(zl), global_total(zc)
    obz = {p: zonal_gpp(o, D.obs_weight(o))
           for p, o in D.obs_clim('GPP').items()}

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 0.40 * COL2),
                             constrained_layout=True)

    # -- (a,b) latent-heat maps on native grids, shared colour scale --
    # display-only: hide mostly-ocean coastal fringe cells & tiny islands
    # (stats/weights are untouched)
    MAP_LF_MIN = 0.1
    maps = [(lh_l, D.lego_lf,
             f'legoESM  ({float(wmean(lh_l, D.lego_w)):.1f} W m$^{{-2}}$)'),
            (lh_c, D.clm_lf,
             f'CLM5.1-SP  ({float(wmean(lh_c, D.clm_w)):.1f} W m$^{{-2}}$)')]
    for ax, letter, (da, lf, title) in zip(axes[:2], 'ab', maps):
        lon2, lat2 = np.meshgrid(da.lon, da.lat)
        pm = ax.pcolormesh(lon2, lat2, da.where(lf >= MAP_LF_MIN),
                           cmap='Blues',
                           vmin=0, vmax=120, shading='auto', rasterized=True)
        ax.set_title(title, fontsize=7.5)
        ax.set_ylim(-60, 90)
        ax.set_yticks([-60, -30, 0, 30, 60, 90])
        ax.set_xticks([0, 90, 180, 270, 360])
        panel_letter(ax, letter)
    axes[1].set_yticklabels([])
    cb = fig.colorbar(pm, ax=axes[:2], location='bottom', shrink=0.6,
                      pad=0.04, aspect=40)
    cb.set_label('latent heat (W m$^{-2}$)', fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # -- (c) zonal GPP sum, one line per dataset --
    ax = axes[2]
    for p, z in obz.items():
        col, ls = OBS_STYLE[p]
        ax.plot(z, z.lat, color=col, ls=ls, lw=0.9,
                label=f'{p} ({global_total(z):.0f})')
    ax.plot(zl, zl.lat, color=C_LEGO, ls=LS_LEGO,
            label=f'legoESM ({tl:.0f})')
    ax.plot(zc, zc.lat, color=C_CLM, ls=LS_CLM,
            label=f'CLM5.1-SP ({tc:.0f})')
    for y in (-23.5, 0, 23.5):
        ax.axhline(y, color='0.6', lw=0.5, ls=':')
    ax.grid(axis='x', color='0.85', lw=0.4)
    ax.set_ylim(-60, 90)
    ax.set_yticks([-60, -30, 0, 30, 60, 90])
    # x-headroom so the legend sits clear of the curves
    ax.set_xlim(0, 1.45 * float(max(zl.max(), zc.max())))
    ax.set_ylabel('latitude ($^\\circ$)')
    ax.set_xlabel('GPP (PgC yr$^{-1}$ deg$^{-1}$)')
    ax.legend(frameon=False, loc='upper right', fontsize=6)
    ax.set_title('Zonal GPP', fontsize=7.5)
    panel_letter(ax, 'c')

    fig.suptitle(f'Annual-mean latent heat & zonal GPP ({D.y0}-{D.y1})',
                 fontsize=8)

    print('\n=== Latent heat, annual mean (land, W/m2) ===')
    print(f'  legoESM : {float(wmean(lh_l, D.lego_w)):6.2f}')
    print(f'  CLM5.1  : {float(wmean(lh_c, D.clm_w)):6.2f}')
    print('\n=== Global total GPP (PgC/yr) ===')
    print(f'  legoESM : {tl:.1f}')
    print(f'  CLM5.1  : {tc:.1f}')
    for p, z in obz.items():
        print(f'  obs {p:8s}: {global_total(z):.1f}')
    savefig(fig, outdir, 'fig_lmip_lh_gpp')


def local_season(clim):
    """Composite a (month,lat,lon) climatology by LOCAL season: SH cells are
    shifted 6 months so month 1 = mid-winter / month 7 = mid-summer for
    every cell.  Removes the bimodality that mixing hemispheres in opposite
    seasons puts into biome statistics (e.g. savanna/shrub in DJF)."""
    shifted = clim.roll(month=-6, roll_coords=False)
    return clim.where(clim.lat >= 0, shifted)


def fig_seasonal_biome(D, outdir):
    """Biome-mean seasonal cycles of LH and GPP for 4 forest/woody biomes,
    composited by local season (SH cells shifted 6 months).  Models with
    area-weighted IQR shading across cells; one line per obs product.
    Biome masks are nearest-sampled onto each dataset's native grid."""
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    mon = np.arange(1, 13)
    cols = ['tropical_broadleaf', 'savanna_shrub',
            'temperate_forest', 'boreal_needleleaf']
    obs = {v: {p: local_season(o) for p, o in D.obs_clim(v).items()}
           for v in ('LH', 'GPP')}
    lego = {v: local_season(D.lego[v]) for v in ('LH', 'GPP')}
    clm = {v: local_season(D.clm[v]) for v in ('LH', 'GPP')}

    # per-dataset (field-independent) biome masks on native grids
    bm_lego = D.biome_on(D.lego.lat, D.lego.lon)
    bm_clm = D.biome_on(D.clm.lat, D.clm.lon)

    rows = [('LH', 'Latent heat (W m$^{-2}$)'),
            ('GPP', 'GPP (gC m$^{-2}$ d$^{-1}$)')]
    letters = iter('abcdefgh')
    fig, axes = plt.subplots(2, 4, figsize=(COL2, 0.52 * COL2),
                             sharex=True, constrained_layout=True)
    for r, (var, ylab) in enumerate(rows):
        for c, bname in enumerate(cols):
            ax = axes[r, c]
            i = BIOMES.index(bname)
            # obs products: one line each
            for p, oda in obs[var].items():
                ow = D.obs_weight(oda)
                m = D.biome_on(oda.lat, oda.lon) == i
                line = wmean(oda.where(m), ow.where(m, 0.0))
                col, ls = OBS_STYLE[p]
                ax.plot(mon, line, color=col, ls=ls, lw=0.9)
            # models: line + area-weighted IQR shading across cells (same
            # weighting as the mean, so the band always brackets the line)
            for da, w, bm, col, ls in [
                    (lego[var], D.lego_w, bm_lego, C_LEGO, LS_LEGO),
                    (clm[var], D.clm_w, bm_clm, C_CLM, LS_CLM)]:
                m = bm == i
                sub = da.where(m & (w > 0))
                line = wmean(sub, w.where(m, 0.0))
                q = sub.weighted(w.where(m, 0.0).fillna(0)).quantile(
                    [0.25, 0.75], dim=('lat', 'lon'))
                ax.fill_between(mon, q.sel(quantile=0.25),
                                q.sel(quantile=0.75), color=col,
                                alpha=0.18, lw=0)
                ax.plot(mon, line, color=col, ls=ls)
            if r == 0:
                ax.set_title(bname.replace('_', '\n'), fontsize=7)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 1:
                ax.set_xlabel('month (local season)')
            ax.set_xticks([1, 4, 7, 10])
            ax.margins(y=0.10)
            panel_letter(ax, next(letters))
    handles = ([mlines.Line2D([], [], color=C_LEGO, ls=LS_LEGO,
                              label='legoESM'),
                mlines.Line2D([], [], color=C_CLM, ls=LS_CLM,
                              label='CLM5.1-SP')] +
               [mlines.Line2D([], [], color=c, ls=ls, lw=0.9, label=p)
                for p, (c, ls) in OBS_STYLE.items()])
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f'Latent heat & GPP seasonality by biome ({D.y0}-{D.y1}; '
                 'local-season composite, SH shifted 6 months; '
                 'shading = model area-weighted IQR across cells)',
                 fontsize=7.5)
    savefig(fig, outdir, 'fig_lmip_seasonal_biome')


# ==========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--lego-dir', default=LEGO_DIR)
    p.add_argument('--clm-dir', default=CLM_DIR)
    p.add_argument('--surfdata', default=SURFDATA)
    p.add_argument('--year0', type=int, default=YEAR0)
    p.add_argument('--year1', type=int, default=YEAR1)
    p.add_argument('--outdir', default=None,
                   help='figure/cache dir (default <lego-dir>/analysis)')
    p.add_argument('--force', action='store_true',
                   help='rebuild cached climatologies')
    args = p.parse_args()

    outdir = args.outdir or os.path.join(args.lego_dir, 'analysis')
    os.makedirs(outdir, exist_ok=True)
    set_style()

    D = Data(args.lego_dir, args.clm_dir, args.surfdata,
             args.year0, args.year1, outdir, args.force)
    fig_lh_gpp(D, outdir)
    fig_seasonal_biome(D, outdir)


if __name__ == '__main__':
    main()
