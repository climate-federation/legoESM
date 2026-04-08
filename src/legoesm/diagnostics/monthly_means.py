"""Monthly-mean diagnostic accumulation for long climate runs.

Accumulates 2D fields and zonal-mean profiles over calendar months,
then outputs monthly averages. Designed for 10-year AMIP-class runs
where storing every timestep is impractical.

Usage
-----
    accum = MonthlyAccumulator(nlev, n_lat_bins=90)
    # Each diagnostic step:
    accum.add(day_of_year, year, fields_2d, fields_3d, lat_2d)
    # At end:
    monthly = accum.finalize()
"""

from __future__ import annotations

import numpy as np


class MonthlyAccumulator:
    """Accumulate 2D and zonal-mean fields into monthly bins.

    Parameters
    ----------
    nlev : int
        Number of vertical levels (for zonal-mean profiles).
    n_lat_bins : int
        Number of latitude bins for zonal means.
    """

    # Days in each month (non-leap year, 365-day calendar)
    MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    def __init__(self, nlev: int, n_lat_bins: int = 90):
        self.nlev = nlev
        self.n_lat_bins = n_lat_bins
        self.lat_edges = np.linspace(-90.0, 90.0, n_lat_bins + 1)
        self.lat_centers = 0.5 * (self.lat_edges[:-1] + self.lat_edges[1:])

        # Storage: dict of {(year, month): {field_name: (sum, count)}}
        self._data: dict[tuple[int, int], dict] = {}

        # Scalar global-mean storage
        self._scalars: dict[tuple[int, int], dict] = {}

    @staticmethod
    def day_to_month(day_of_year: float) -> int:
        """Convert day-of-year (1-based) to month (1-12)."""
        day = int(day_of_year) - 1  # 0-based
        day = max(0, min(day, 364))
        cum = 0
        for m, d in enumerate(MonthlyAccumulator.MONTH_DAYS, 1):
            cum += d
            if day < cum:
                return m
        return 12

    def _get_key(self, day_of_year: float, year: int) -> tuple[int, int]:
        month = self.day_to_month(day_of_year)
        return (year, month)

    def _ensure_bucket(self, key: tuple[int, int]):
        if key not in self._data:
            self._data[key] = {}
            self._scalars[key] = {}

    def add_2d(
        self,
        day_of_year: float,
        year: int,
        fields: dict[str, np.ndarray],
        lat_deg: np.ndarray,
    ) -> None:
        """Add 2D fields (on cubed-sphere) to the monthly accumulator.

        Fields are binned into zonal-mean latitude bands and also
        accumulated as global means.

        Parameters
        ----------
        day_of_year : float
            Day of year (1-based).
        year : int
            Year number (for multi-year runs).
        fields : dict
            Mapping of field names to 2D arrays, shape (6, n, n) or flat.
        lat_deg : np.ndarray
            Latitude in degrees for each grid point, same shape as fields.
        """
        key = self._get_key(day_of_year, year)
        self._ensure_bucket(key)
        bucket = self._data[key]
        scalars = self._scalars[key]

        lat_flat = np.asarray(lat_deg).ravel()
        bin_idx = np.digitize(lat_flat, self.lat_edges) - 1
        bin_idx = np.clip(bin_idx, 0, self.n_lat_bins - 1)

        for name, field in fields.items():
            vals = np.asarray(field).ravel()

            # Global mean
            if name not in scalars:
                scalars[name] = (0.0, 0)
            s, c = scalars[name]
            scalars[name] = (s + float(np.mean(vals)), c + 1)

            # Zonal mean
            zonal_key = f"zonal_{name}"
            if zonal_key not in bucket:
                bucket[zonal_key] = (
                    np.zeros(self.n_lat_bins),
                    np.zeros(self.n_lat_bins, dtype=int),
                )
            zsum, zcount = bucket[zonal_key]
            np.add.at(zsum, bin_idx, vals)
            np.add.at(zcount, bin_idx, 1)

    def add_3d(
        self,
        day_of_year: float,
        year: int,
        fields: dict[str, np.ndarray],
        lat_deg: np.ndarray,
    ) -> None:
        """Add 3D fields for zonal-mean profile accumulation.

        Parameters
        ----------
        fields : dict
            Mapping of field names to 3D arrays, shape (..., nlev).
        lat_deg : np.ndarray
            Latitude in degrees, shape (...) matching spatial dims.
        """
        key = self._get_key(day_of_year, year)
        self._ensure_bucket(key)
        bucket = self._data[key]

        lat_flat = np.asarray(lat_deg).ravel()
        bin_idx = np.digitize(lat_flat, self.lat_edges) - 1
        bin_idx = np.clip(bin_idx, 0, self.n_lat_bins - 1)

        for name, field in fields.items():
            arr = np.asarray(field)
            # Reshape to (ncol, nlev)
            ncol = lat_flat.shape[0]
            nlev = arr.shape[-1]
            vals = arr.reshape(ncol, nlev)

            profile_key = f"profile_{name}"
            if profile_key not in bucket:
                bucket[profile_key] = (
                    np.zeros((self.n_lat_bins, nlev)),
                    np.zeros(self.n_lat_bins, dtype=int),
                )
            psum, pcount = bucket[profile_key]
            for k in range(nlev):
                np.add.at(psum[:, k], bin_idx, vals[:, k])
            np.add.at(pcount, bin_idx, 1)

    def add_scalar(
        self,
        day_of_year: float,
        year: int,
        scalars: dict[str, float],
    ) -> None:
        """Add global-mean scalar diagnostics."""
        key = self._get_key(day_of_year, year)
        self._ensure_bucket(key)
        bucket = self._scalars[key]
        for name, val in scalars.items():
            if name not in bucket:
                bucket[name] = (0.0, 0)
            s, c = bucket[name]
            bucket[name] = (s + val, c + 1)

    def finalize(self) -> dict:
        """Compute monthly means and return structured output.

        Returns
        -------
        dict with keys:
            'months' : list of (year, month) tuples
            'zonal_{field}' : array (n_months, n_lat_bins) — zonal means
            'profile_{field}' : array (n_months, n_lat_bins, nlev) — zonal-mean profiles
            'scalar_{field}' : array (n_months,) — global-mean scalars
            'lat' : array (n_lat_bins,) — latitude bin centers
        """
        months = sorted(self._data.keys())
        result: dict = {
            'months': months,
            'lat': self.lat_centers,
        }

        if not months:
            return result

        # Collect all field names
        zonal_fields: set[str] = set()
        profile_fields: set[str] = set()
        scalar_fields: set[str] = set()

        for key in months:
            for k in self._data[key]:
                if k.startswith("zonal_"):
                    zonal_fields.add(k)
                elif k.startswith("profile_"):
                    profile_fields.add(k)
            for k in self._scalars.get(key, {}):
                scalar_fields.add(k)

        n_months = len(months)

        # Zonal means
        for fname in sorted(zonal_fields):
            arr = np.full((n_months, self.n_lat_bins), np.nan)
            for i, key in enumerate(months):
                if fname in self._data[key]:
                    zsum, zcount = self._data[key][fname]
                    mask = zcount > 0
                    arr[i, mask] = zsum[mask] / zcount[mask]
            result[fname] = arr

        # Profiles
        for fname in sorted(profile_fields):
            nlev = self.nlev
            # Get actual nlev from data
            for key in months:
                if fname in self._data[key]:
                    nlev = self._data[key][fname][0].shape[1]
                    break
            arr = np.full((n_months, self.n_lat_bins, nlev), np.nan)
            for i, key in enumerate(months):
                if fname in self._data[key]:
                    psum, pcount = self._data[key][fname]
                    mask = pcount > 0
                    for k in range(nlev):
                        arr[i, mask, k] = psum[mask, k] / pcount[mask]
            result[fname] = arr

        # Scalars
        for fname in sorted(scalar_fields):
            arr = np.full(n_months, np.nan)
            for i, key in enumerate(months):
                if fname in self._scalars.get(key, {}):
                    s, c = self._scalars[key][fname]
                    if c > 0:
                        arr[i] = s / c
            result[f"scalar_{fname}"] = arr

        return result

    def save(self, path, **extra_arrays) -> None:
        """Finalize and save to npz file."""
        data = self.finalize()
        save_dict = {}

        # Convert months to arrays
        if data['months']:
            years = np.array([m[0] for m in data['months']])
            month_nums = np.array([m[1] for m in data['months']])
            save_dict['years'] = years
            save_dict['month_nums'] = month_nums
        save_dict['lat'] = data['lat']

        for k, v in data.items():
            if k in ('months', 'lat'):
                continue
            if isinstance(v, np.ndarray):
                save_dict[k] = v

        save_dict.update(extra_arrays)
        np.savez(str(path), **save_dict)


class SpatialMonthlyAccumulator:
    """Accumulate full 2-D and 3-D spatial fields into monthly bins.

    Unlike :class:`MonthlyAccumulator` (which stores only zonal means),
    this class stores complete ``(nlat, nlon)`` and ``(nlat, nlon, nlev)``
    fields, producing scientifically valid monthly means for CMIP output.

    For cubed-sphere source grids the caller should regrid to lat-lon
    *before* calling :meth:`add_2d` / :meth:`add_3d`.

    Parameters
    ----------
    nlat, nlon : int
        Target lat-lon grid dimensions.
    nlev : int
        Number of vertical levels (for 3-D fields).
    """

    MONTH_DAYS = MonthlyAccumulator.MONTH_DAYS

    def __init__(self, nlat: int, nlon: int, nlev: int = 0):
        self.nlat = nlat
        self.nlon = nlon
        self.nlev = nlev
        # Storage: {(year, month): {field_name: (sum_array, count)}}
        self._data_2d: dict[tuple[int, int], dict] = {}
        self._data_3d: dict[tuple[int, int], dict] = {}

    @staticmethod
    def day_to_month(day_of_year: float) -> int:
        return MonthlyAccumulator.day_to_month(day_of_year)

    def _key(self, day_of_year: float, year: int) -> tuple[int, int]:
        return (year, self.day_to_month(day_of_year))

    def add_2d(
        self,
        day_of_year: float,
        year: int,
        fields: dict[str, np.ndarray],
    ) -> None:
        """Add regridded 2-D fields, shape ``(nlat, nlon)``."""
        key = self._key(day_of_year, year)
        bucket = self._data_2d.setdefault(key, {})
        for name, field in fields.items():
            arr = np.asarray(field)
            if arr.shape != (self.nlat, self.nlon):
                raise ValueError(
                    f"SpatialMonthlyAccumulator.add_2d: expected "
                    f"({self.nlat}, {self.nlon}), got {arr.shape} "
                    f"for field {name!r}"
                )
            if name not in bucket:
                bucket[name] = (np.zeros_like(arr, dtype=np.float64), 0)
            s, c = bucket[name]
            s += arr.astype(np.float64)
            bucket[name] = (s, c + 1)

    def add_3d(
        self,
        day_of_year: float,
        year: int,
        fields: dict[str, np.ndarray],
    ) -> None:
        """Add regridded 3-D fields, shape ``(nlat, nlon, nlev)``."""
        key = self._key(day_of_year, year)
        bucket = self._data_3d.setdefault(key, {})
        for name, field in fields.items():
            arr = np.asarray(field)
            if arr.ndim != 3 or arr.shape[0] != self.nlat or arr.shape[1] != self.nlon:
                raise ValueError(
                    f"SpatialMonthlyAccumulator.add_3d: expected "
                    f"({self.nlat}, {self.nlon}, nlev), got {arr.shape} "
                    f"for field {name!r}"
                )
            if name not in bucket:
                bucket[name] = (np.zeros_like(arr, dtype=np.float64), 0)
            s, c = bucket[name]
            s += arr.astype(np.float64)
            bucket[name] = (s, c + 1)

    def finalize(self) -> dict:
        """Compute monthly means and return structured output.

        Returns
        -------
        dict with keys:
            'months' : sorted list of (year, month)
            'field_2d_{name}' : (n_months, nlat, nlon) — 2-D means
            'field_3d_{name}' : (n_months, nlat, nlon, nlev) — 3-D means
        """
        months_2d = set(self._data_2d.keys())
        months_3d = set(self._data_3d.keys())
        months = sorted(months_2d | months_3d)
        result: dict = {'months': months}
        if not months:
            return result

        n_months = len(months)

        # 2-D fields
        all_2d_names: set[str] = set()
        for bucket in self._data_2d.values():
            all_2d_names.update(bucket.keys())
        for fname in sorted(all_2d_names):
            arr = np.full((n_months, self.nlat, self.nlon), np.nan)
            for i, key in enumerate(months):
                bucket = self._data_2d.get(key, {})
                if fname in bucket:
                    s, c = bucket[fname]
                    if c > 0:
                        arr[i] = s / c
            result[f"field_2d_{fname}"] = arr

        # 3-D fields
        all_3d_names: set[str] = set()
        for bucket in self._data_3d.values():
            all_3d_names.update(bucket.keys())
        for fname in sorted(all_3d_names):
            sample = None
            for bucket in self._data_3d.values():
                if fname in bucket:
                    sample = bucket[fname][0]
                    break
            if sample is None:
                continue
            nlev = sample.shape[2]
            arr = np.full((n_months, self.nlat, self.nlon, nlev), np.nan)
            for i, key in enumerate(months):
                bucket = self._data_3d.get(key, {})
                if fname in bucket:
                    s, c = bucket[fname]
                    if c > 0:
                        arr[i] = s / c
            result[f"field_3d_{fname}"] = arr

        return result
