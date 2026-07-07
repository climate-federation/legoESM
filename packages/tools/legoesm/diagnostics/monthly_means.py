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

import json

import numpy as np

# Sidecar-state schema version for the accumulator ``get_state`` /
# ``set_state`` round-trip (bump only on an incompatible manifest-layout
# change).  A restore rejects any other version so a stale sidecar fails
# loudly rather than corrupting the restored means.
_STATE_VERSION = 1


def _encode_manifest(manifest: dict) -> np.ndarray:
    """Serialize a state manifest dict to a 0-d numpy string array so it can
    ride inside the same ``np.savez`` sidecar as the payload arrays."""
    return np.asarray(json.dumps(manifest))


def _decode_manifest(manifest_obj) -> dict:
    """Inverse of :func:`_encode_manifest`; accepts the 0-d string array (as
    reloaded by ``np.load``) or a raw JSON string."""
    if isinstance(manifest_obj, np.ndarray):
        manifest_obj = manifest_obj.item()
    return json.loads(str(manifest_obj))


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

        # Number of ``add_2d`` / ``add_3d`` / ``add_scalar`` calls per bucket.
        # Used by ``finalize`` to drop partial-month buckets (e.g. a final
        # sample from a 90-day run that spills into day-of-year 91 → April).
        self._call_counts: dict[tuple[int, int], int] = {}
        self._max_count_ever: int = 0

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
            self._call_counts[key] = 0

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
        self._call_counts[key] += 1
        if self._call_counts[key] > self._max_count_ever:
            self._max_count_ever = self._call_counts[key]

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
        self._call_counts[key] += 1
        if self._call_counts[key] > self._max_count_ever:
            self._max_count_ever = self._call_counts[key]

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
        self._call_counts[key] += 1
        if self._call_counts[key] > self._max_count_ever:
            self._max_count_ever = self._call_counts[key]
        for name, val in scalars.items():
            if name not in bucket:
                bucket[name] = (0.0, 0)
            s, c = bucket[name]
            bucket[name] = (s + val, c + 1)

    def finalize(self, min_sample_fraction: float = 0.5) -> dict:
        """Compute monthly means and return structured output.

        Parameters
        ----------
        min_sample_fraction : float, default 0.5
            Partial months (buckets with fewer than
            ``min_sample_fraction * max(sample_count)`` sampling events)
            are dropped. A 90-day run starting Jan 1 ends at
            day-of-year 91 (April 1), which would otherwise produce a
            1-sample "April mean" alongside three full months. Set to 0
            to disable.

        Returns
        -------
        dict with keys:
            'months' : list of (year, month) tuples
            'zonal_{field}' : array (n_months, n_lat_bins) — zonal means
            'profile_{field}' : array (n_months, n_lat_bins, nlev) — zonal-mean profiles
            'scalar_{field}' : array (n_months,) — global-mean scalars
            'lat' : array (n_lat_bins,) — latitude bin centers
        """
        all_months = sorted(self._data.keys())
        if not all_months:
            return {'months': [], 'lat': self.lat_centers}

        # Drop partial-month buckets (prevents e.g. a single April-1
        # sample from a 90-day run contaminating the output). Uses
        # ``_max_count_ever`` so the guard also fires when only one
        # bucket remains (e.g. after incremental flushing).
        if min_sample_fraction > 0:
            live_max = max(
                (self._call_counts.get(k, 0) for k in all_months),
                default=0,
            )
            max_count = max(live_max, self._max_count_ever)
            if max_count > 0:
                threshold = min_sample_fraction * max_count
                months = [
                    k for k in all_months
                    if self._call_counts.get(k, 0) >= threshold
                ]
            else:
                months = all_months
        else:
            months = all_months

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

    def get_state(self) -> dict[str, np.ndarray]:
        """Serialize all mutable accumulator state to a flat, ``np.savez``-able
        dict for restart-chain persistence.

        The nested ``{(year, month): {field: (sum, count)}}`` structure is
        encoded as a JSON manifest (a 0-d string array under ``"__manifest__"``)
        plus one payload array per ``(bucket, field)`` sum/count, keyed by
        index.  Reversed exactly by :meth:`set_state` (float64 sums + integer
        counts preserved bit-for-bit).  Pure numpy — no JAX.
        """
        arrays: dict[str, np.ndarray] = {}

        def _put(value) -> str:
            akey = f"arr_{len(arrays)}"
            arrays[akey] = np.asarray(value)
            return akey

        data_entries: list = []
        for (yr, mo), bucket in self._data.items():
            for fkey, (fsum, fcount) in bucket.items():
                data_entries.append(
                    [int(yr), int(mo), fkey, _put(fsum), _put(fcount)]
                )
        scalar_entries: list = []
        for (yr, mo), bucket in self._scalars.items():
            for name, (ssum, scount) in bucket.items():
                scalar_entries.append(
                    [int(yr), int(mo), name, _put(ssum), int(scount)]
                )
        manifest = {
            "version": _STATE_VERSION,
            "type": "MonthlyAccumulator",
            "nlev": int(self.nlev),
            "n_lat_bins": int(self.n_lat_bins),
            "max_count_ever": int(self._max_count_ever),
            "call_counts": [
                [int(yr), int(mo), int(c)]
                for (yr, mo), c in self._call_counts.items()
            ],
            "data": data_entries,
            "scalars": scalar_entries,
        }
        out: dict[str, np.ndarray] = {"__manifest__": _encode_manifest(manifest)}
        out.update(arrays)
        return out

    def set_state(self, state: dict) -> None:
        """Restore state written by :meth:`get_state` (exact round-trip).

        Raises ``ValueError`` on an unknown schema version or a
        construction-dimension mismatch (``nlev`` / ``n_lat_bins``).
        """
        manifest = _decode_manifest(state["__manifest__"])
        if manifest.get("version") != _STATE_VERSION:
            raise ValueError(
                f"MonthlyAccumulator.set_state: unknown state version "
                f"{manifest.get('version')!r} (expected {_STATE_VERSION})"
            )
        saved = (int(manifest["nlev"]), int(manifest["n_lat_bins"]))
        if saved != (self.nlev, self.n_lat_bins):
            raise ValueError(
                f"MonthlyAccumulator.set_state: dims mismatch — saved "
                f"(nlev, n_lat_bins)={saved} vs current "
                f"{(self.nlev, self.n_lat_bins)}"
            )
        self._max_count_ever = int(manifest["max_count_ever"])
        self._call_counts = {
            (int(yr), int(mo)): int(c) for yr, mo, c in manifest["call_counts"]
        }
        # Recreate every bucket (including any empty ones) so the
        # ``_ensure_bucket`` invariant — identical key sets across
        # ``_data`` / ``_scalars`` / ``_call_counts`` — round-trips exactly.
        self._data = {key: {} for key in self._call_counts}
        self._scalars = {key: {} for key in self._call_counts}
        for yr, mo, fkey, skey, ckey in manifest["data"]:
            self._data.setdefault((int(yr), int(mo)), {})[fkey] = (
                np.array(state[skey]), np.array(state[ckey]),
            )
        for yr, mo, name, skey, scount in manifest["scalars"]:
            self._scalars.setdefault((int(yr), int(mo)), {})[name] = (
                float(np.asarray(state[skey]).item()), int(scount),
            )


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
        # Per-bucket add_* call count; used to drop partial months.
        self._call_counts: dict[tuple[int, int], int] = {}
        # Max add_* count observed on any bucket over the lifetime of
        # this accumulator, including buckets already popped by
        # ``pop_completed_months``. This lets ``finalize`` apply the
        # partial-month guard when only the in-progress bucket remains.
        self._max_count_ever: int = 0

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
        new_count = self._call_counts.get(key, 0) + 1
        self._call_counts[key] = new_count
        if new_count > self._max_count_ever:
            self._max_count_ever = new_count
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
        new_count = self._call_counts.get(key, 0) + 1
        self._call_counts[key] = new_count
        if new_count > self._max_count_ever:
            self._max_count_ever = new_count
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

    def finalize(self, min_sample_fraction: float = 0.5) -> dict:
        """Compute monthly means and return structured output.

        Partial-month buckets (< ``min_sample_fraction`` of the maximum
        sample count) are dropped so that e.g. a single April-1 sample
        from a 90-day run does not appear as a full monthly mean.

        Returns
        -------
        dict with keys:
            'months' : sorted list of (year, month)
            'field_2d_{name}' : (n_months, nlat, nlon) — 2-D means
            'field_3d_{name}' : (n_months, nlat, nlon, nlev) — 3-D means
        """
        months_2d = set(self._data_2d.keys())
        months_3d = set(self._data_3d.keys())
        all_months = sorted(months_2d | months_3d)
        if not all_months:
            return {'months': []}

        if min_sample_fraction > 0:
            # Use the lifetime max (including buckets already popped by
            # ``pop_completed_months``) so the guard still fires when
            # only the in-progress month remains at end of run.
            live_max = max(
                (self._call_counts.get(k, 0) for k in all_months),
                default=0,
            )
            max_count = max(live_max, self._max_count_ever)
            if max_count > 0:
                threshold = min_sample_fraction * max_count
                months = [
                    k for k in all_months
                    if self._call_counts.get(k, 0) >= threshold
                ]
            else:
                months = all_months
        else:
            months = all_months
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

    def pop_completed_months(self, current_year: int, current_month: int) -> dict:
        """Finalize and remove months that are strictly before the current month.

        This allows incremental flushing: completed months are returned
        and freed from memory, keeping only the in-progress month.

        Parameters
        ----------
        current_year, current_month : int
            The month currently being accumulated (will NOT be popped).

        Returns
        -------
        dict
            Same structure as :meth:`finalize` but only for completed months.
            Empty ``{'months': []}`` if no months are ready.
        """
        current_key = (current_year, current_month)
        completed_2d = {k for k in self._data_2d if k < current_key}
        completed_3d = {k for k in self._data_3d if k < current_key}
        completed = sorted(completed_2d | completed_3d)
        if not completed:
            return {'months': []}

        result: dict = {'months': completed}
        n = len(completed)

        # 2-D
        all_2d_names: set[str] = set()
        for key in completed:
            bucket = self._data_2d.get(key, {})
            all_2d_names.update(bucket.keys())
        for fname in sorted(all_2d_names):
            arr = np.full((n, self.nlat, self.nlon), np.nan)
            for i, key in enumerate(completed):
                bucket = self._data_2d.get(key, {})
                if fname in bucket:
                    s, c = bucket[fname]
                    if c > 0:
                        arr[i] = s / c
            result[f"field_2d_{fname}"] = arr

        # 3-D
        all_3d_names: set[str] = set()
        for key in completed:
            bucket = self._data_3d.get(key, {})
            all_3d_names.update(bucket.keys())
        for fname in sorted(all_3d_names):
            sample = None
            for key in completed:
                bucket = self._data_3d.get(key, {})
                if fname in bucket:
                    sample = bucket[fname][0]
                    break
            if sample is None:
                continue
            nlev = sample.shape[2]
            arr = np.full((n, self.nlat, self.nlon, nlev), np.nan)
            for i, key in enumerate(completed):
                bucket = self._data_3d.get(key, {})
                if fname in bucket:
                    s, c = bucket[fname]
                    if c > 0:
                        arr[i] = s / c
            result[f"field_3d_{fname}"] = arr

        # Free memory for completed months
        for key in completed:
            self._data_2d.pop(key, None)
            self._data_3d.pop(key, None)

        return result

    def get_state(self) -> dict[str, np.ndarray]:
        """Serialize all mutable state to a flat, ``np.savez``-able dict
        (see :meth:`MonthlyAccumulator.get_state`).

        Reversed exactly by :meth:`set_state`; float64 sums and integer
        counts are preserved bit-for-bit.  Pure numpy — no JAX.
        """
        arrays: dict[str, np.ndarray] = {}

        def _put(value) -> str:
            akey = f"arr_{len(arrays)}"
            arrays[akey] = np.asarray(value)
            return akey

        data_2d: list = []
        for (yr, mo), bucket in self._data_2d.items():
            for name, (fsum, fcount) in bucket.items():
                data_2d.append([int(yr), int(mo), name, int(fcount), _put(fsum)])
        data_3d: list = []
        for (yr, mo), bucket in self._data_3d.items():
            for name, (fsum, fcount) in bucket.items():
                data_3d.append([int(yr), int(mo), name, int(fcount), _put(fsum)])
        manifest = {
            "version": _STATE_VERSION,
            "type": "SpatialMonthlyAccumulator",
            "nlat": int(self.nlat),
            "nlon": int(self.nlon),
            "nlev": int(self.nlev),
            "max_count_ever": int(self._max_count_ever),
            "call_counts": [
                [int(yr), int(mo), int(c)]
                for (yr, mo), c in self._call_counts.items()
            ],
            # 2-D and 3-D bucket key sets are tracked separately (``_call_counts``
            # is their union), so any empty bucket round-trips to the right store.
            "buckets_2d": [[int(yr), int(mo)] for (yr, mo) in self._data_2d],
            "buckets_3d": [[int(yr), int(mo)] for (yr, mo) in self._data_3d],
            "data_2d": data_2d,
            "data_3d": data_3d,
        }
        out: dict[str, np.ndarray] = {"__manifest__": _encode_manifest(manifest)}
        out.update(arrays)
        return out

    def set_state(self, state: dict) -> None:
        """Restore state written by :meth:`get_state` (exact round-trip).

        Raises ``ValueError`` on an unknown schema version or a
        construction-dimension mismatch (``nlat`` / ``nlon`` / ``nlev``).
        """
        manifest = _decode_manifest(state["__manifest__"])
        if manifest.get("version") != _STATE_VERSION:
            raise ValueError(
                f"SpatialMonthlyAccumulator.set_state: unknown state version "
                f"{manifest.get('version')!r} (expected {_STATE_VERSION})"
            )
        saved = (int(manifest["nlat"]), int(manifest["nlon"]), int(manifest["nlev"]))
        if saved != (self.nlat, self.nlon, self.nlev):
            raise ValueError(
                f"SpatialMonthlyAccumulator.set_state: dims mismatch — saved "
                f"(nlat, nlon, nlev)={saved} vs current "
                f"{(self.nlat, self.nlon, self.nlev)}"
            )
        self._max_count_ever = int(manifest["max_count_ever"])
        self._call_counts = {
            (int(yr), int(mo)): int(c) for yr, mo, c in manifest["call_counts"]
        }
        # Pre-create every bucket (including any empty ones) so the 2-D / 3-D
        # key sets round-trip exactly.
        self._data_2d = {(int(yr), int(mo)): {} for yr, mo in manifest["buckets_2d"]}
        self._data_3d = {(int(yr), int(mo)): {} for yr, mo in manifest["buckets_3d"]}
        for yr, mo, name, fcount, akey in manifest["data_2d"]:
            self._data_2d.setdefault((int(yr), int(mo)), {})[name] = (
                np.array(state[akey], dtype=np.float64), int(fcount),
            )
        for yr, mo, name, fcount, akey in manifest["data_3d"]:
            self._data_3d.setdefault((int(yr), int(mo)), {})[name] = (
                np.array(state[akey], dtype=np.float64), int(fcount),
            )


class SpatialDailyAccumulator:
    """Accumulate daily-resolved 2-D spatial fields for CMIP6 ``day`` table.

    Bucketed by (year, day_of_year). Tracks running mean for all fields
    and, for fields in ``track_extremes``, running daily min and max so
    that tasmin/tasmax can be emitted alongside tas.

    Parameters
    ----------
    nlat, nlon : int
        Target lat-lon grid dimensions.
    track_extremes : iterable of str, optional
        Field names for which daily min *and* max are tracked in addition
        to the mean. Default: ``{"tas"}``. These produce ``tasmin`` /
        ``tasmax`` (or ``<name>min`` / ``<name>max``) on finalize.
    """

    def __init__(
        self, nlat: int, nlon: int,
        track_extremes: set[str] | None = None,
    ) -> None:
        self.nlat = nlat
        self.nlon = nlon
        self.track_extremes: set[str] = (
            set(track_extremes) if track_extremes is not None else {"tas"}
        )
        # Per-day buckets keyed by (year, doy_int).
        # bucket[name] = [sum, count, min_or_None, max_or_None]
        self._data: dict[tuple[int, int], dict] = {}
        self._call_counts: dict[tuple[int, int], int] = {}
        self._max_count_ever: int = 0

    @staticmethod
    def _key(day_of_year: float, year: int) -> tuple[int, int]:
        return (year, int(day_of_year))

    def add_2d(
        self,
        day_of_year: float,
        year: int,
        fields: dict[str, np.ndarray],
    ) -> None:
        """Add regridded 2-D fields, shape ``(nlat, nlon)``."""
        key = self._key(day_of_year, year)
        bucket = self._data.setdefault(key, {})
        new_count = self._call_counts.get(key, 0) + 1
        self._call_counts[key] = new_count
        if new_count > self._max_count_ever:
            self._max_count_ever = new_count
        for name, field in fields.items():
            arr = np.asarray(field, dtype=np.float64)
            if arr.shape != (self.nlat, self.nlon):
                raise ValueError(
                    f"SpatialDailyAccumulator.add_2d: expected "
                    f"({self.nlat}, {self.nlon}), got {arr.shape} "
                    f"for field {name!r}"
                )
            track = name in self.track_extremes
            if name not in bucket:
                sum_arr = np.zeros_like(arr)
                min_arr = np.copy(arr) if track else None
                max_arr = np.copy(arr) if track else None
                bucket[name] = [sum_arr, 0, min_arr, max_arr]
            entry = bucket[name]
            entry[0] += arr
            entry[1] += 1
            if track:
                np.minimum(entry[2], arr, out=entry[2])
                np.maximum(entry[3], arr, out=entry[3])

    def finalize(self, min_sample_fraction: float = 0.5) -> dict:
        """Compute daily means (and extremes where tracked).

        Partial-day buckets (< ``min_sample_fraction`` of the maximum
        sample count across all days) are dropped so the tail of a run
        ending mid-day does not produce a spurious daily mean with too
        few samples. Setting ``min_sample_fraction=0`` disables the
        guard.

        Returns
        -------
        dict with keys:
            'days' : sorted list of (year, doy)
            'field_2d_{name}' : (n_days, nlat, nlon) — daily mean
            'field_2d_{name}_min' : (n_days, nlat, nlon) — for tracked fields
            'field_2d_{name}_max' : (n_days, nlat, nlon) — for tracked fields
        """
        all_days = sorted(self._data.keys())
        if not all_days:
            return {'days': []}

        if min_sample_fraction > 0:
            live_max = max(
                (self._call_counts.get(k, 0) for k in all_days),
                default=0,
            )
            max_count = max(live_max, self._max_count_ever)
            if max_count > 0:
                threshold = min_sample_fraction * max_count
                days = [
                    k for k in all_days
                    if self._call_counts.get(k, 0) >= threshold
                ]
            else:
                days = all_days
        else:
            days = all_days

        result: dict = {'days': days}
        if not days:
            return result

        n_days = len(days)
        all_names: set[str] = set()
        for bucket in self._data.values():
            all_names.update(bucket.keys())

        for name in sorted(all_names):
            mean_arr = np.full((n_days, self.nlat, self.nlon), np.nan)
            has_extrema = name in self.track_extremes
            if has_extrema:
                min_out = np.full((n_days, self.nlat, self.nlon), np.nan)
                max_out = np.full((n_days, self.nlat, self.nlon), np.nan)
            for i, key in enumerate(days):
                bucket = self._data.get(key, {})
                if name in bucket:
                    s, c, mn, mx = bucket[name]
                    if c > 0:
                        mean_arr[i] = s / c
                    if has_extrema and mn is not None:
                        min_out[i] = mn
                        max_out[i] = mx
            result[f"field_2d_{name}"] = mean_arr
            if has_extrema:
                result[f"field_2d_{name}_min"] = min_out
                result[f"field_2d_{name}_max"] = max_out

        return result

    def pop_completed_days(self, current_year: int, current_doy: int) -> dict:
        """Finalize and remove days strictly before the in-progress day.

        Daily analogue of
        :meth:`SpatialMonthlyAccumulator.pop_completed_months`.  A graceful
        wallclock exit uses this to emit (and free) the segment's COMPLETED
        daily means while leaving the in-progress day un-written: writing the
        partial current day here and then again from the restart segment —
        which resumes inside the same ``(year, doy)`` — would produce
        duplicate ``time`` coordinates and split-day means, because
        ``CFWriter.write_field`` appends blindly to an existing file.  Unlike
        :meth:`finalize`, no ``min_sample_fraction`` guard is applied (parity
        with ``pop_completed_months``); every strictly-past day is complete by
        construction.

        Parameters
        ----------
        current_year, current_doy : int
            The day currently being accumulated (will NOT be popped).

        Returns
        -------
        dict
            Same structure as :meth:`finalize` but only for completed days.
            Empty ``{'days': []}`` if none are ready.
        """
        current_key = (current_year, int(current_doy))
        completed = sorted(k for k in self._data if k < current_key)
        if not completed:
            return {'days': []}

        result: dict = {'days': completed}
        n = len(completed)
        all_names: set[str] = set()
        for key in completed:
            all_names.update(self._data.get(key, {}).keys())

        for name in sorted(all_names):
            has_extrema = name in self.track_extremes
            mean_arr = np.full((n, self.nlat, self.nlon), np.nan)
            if has_extrema:
                min_out = np.full((n, self.nlat, self.nlon), np.nan)
                max_out = np.full((n, self.nlat, self.nlon), np.nan)
            for i, key in enumerate(completed):
                bucket = self._data.get(key, {})
                if name in bucket:
                    s, c, mn, mx = bucket[name]
                    if c > 0:
                        mean_arr[i] = s / c
                    if has_extrema and mn is not None:
                        min_out[i] = mn
                        max_out[i] = mx
            result[f"field_2d_{name}"] = mean_arr
            if has_extrema:
                result[f"field_2d_{name}_min"] = min_out
                result[f"field_2d_{name}_max"] = max_out

        # Free memory for completed days.
        for key in completed:
            self._data.pop(key, None)
            self._call_counts.pop(key, None)

        return result

    def get_state(self) -> dict[str, np.ndarray]:
        """Serialize all mutable state to a flat, ``np.savez``-able dict
        (see :meth:`MonthlyAccumulator.get_state`).

        Preserves the running sums, integer counts, and the per-day min/max
        extremes exactly (a ``None`` extreme — untracked field — round-trips as
        an absent array key).  Pure numpy — no JAX.
        """
        arrays: dict[str, np.ndarray] = {}

        def _put(value) -> str:
            akey = f"arr_{len(arrays)}"
            arrays[akey] = np.asarray(value)
            return akey

        entries: list = []
        for (yr, doy), bucket in self._data.items():
            for name, entry in bucket.items():
                fsum, fcount, fmin, fmax = entry
                entries.append([
                    int(yr), int(doy), name, int(fcount),
                    _put(fsum),
                    None if fmin is None else _put(fmin),
                    None if fmax is None else _put(fmax),
                ])
        manifest = {
            "version": _STATE_VERSION,
            "type": "SpatialDailyAccumulator",
            "nlat": int(self.nlat),
            "nlon": int(self.nlon),
            "track_extremes": sorted(self.track_extremes),
            "max_count_ever": int(self._max_count_ever),
            "call_counts": [
                [int(yr), int(doy), int(c)]
                for (yr, doy), c in self._call_counts.items()
            ],
            "data": entries,
        }
        out: dict[str, np.ndarray] = {"__manifest__": _encode_manifest(manifest)}
        out.update(arrays)
        return out

    def set_state(self, state: dict) -> None:
        """Restore state written by :meth:`get_state` (exact round-trip).

        Raises ``ValueError`` on an unknown schema version or a
        construction-dimension mismatch (``nlat`` / ``nlon``).
        """
        manifest = _decode_manifest(state["__manifest__"])
        if manifest.get("version") != _STATE_VERSION:
            raise ValueError(
                f"SpatialDailyAccumulator.set_state: unknown state version "
                f"{manifest.get('version')!r} (expected {_STATE_VERSION})"
            )
        saved = (int(manifest["nlat"]), int(manifest["nlon"]))
        if saved != (self.nlat, self.nlon):
            raise ValueError(
                f"SpatialDailyAccumulator.set_state: dims mismatch — saved "
                f"(nlat, nlon)={saved} vs current {(self.nlat, self.nlon)}"
            )
        self.track_extremes = set(manifest["track_extremes"])
        self._max_count_ever = int(manifest["max_count_ever"])
        self._call_counts = {
            (int(yr), int(doy)): int(c) for yr, doy, c in manifest["call_counts"]
        }
        # Pre-create every bucket (incl. empties) so the day key set round-trips
        # (``_data`` and ``_call_counts`` share their key set by construction).
        self._data = {key: {} for key in self._call_counts}
        for yr, doy, name, fcount, skey, mnkey, mxkey in manifest["data"]:
            fmin = None if mnkey is None else np.array(state[mnkey], dtype=np.float64)
            fmax = None if mxkey is None else np.array(state[mxkey], dtype=np.float64)
            self._data.setdefault((int(yr), int(doy)), {})[name] = [
                np.array(state[skey], dtype=np.float64), int(fcount), fmin, fmax,
            ]
