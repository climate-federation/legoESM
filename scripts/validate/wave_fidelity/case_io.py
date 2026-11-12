"""Read the four archived WW3 propagation experiments, without a model dependency."""

import json
from pathlib import Path

import netCDF4 as nc
import numpy as np



def read_config(path):
    return json.loads(Path(path).read_text())


def times_from_variable(variable):
    """Decode CF timestamps; never use a frame number as physical time."""
    dates = nc.num2date(variable[:], variable.units,
                       calendar=getattr(variable, "calendar", "standard"))
    return np.array([np.datetime64(d.strftime("%Y-%m-%dT%H:%M:%S"), "s") for d in dates])


def frequency_widths(freq):
    """waden's midpoint finite-volume widths, in Hz (including endpoint cells)."""
    freq = np.asarray(freq, dtype=np.float64)
    edges = np.r_[freq[0] - (freq[1] - freq[0]) / 2,
                  (freq[1:] + freq[:-1]) / 2,
                  freq[-1] + (freq[-1] - freq[-2]) / 2]
    return np.diff(edges)


def moments(spectra, theta_deg, freq):
    """Common finite-band moments; MWD is a going-to compass angle."""
    spectra = np.asarray(spectra, dtype=np.float64)
    weighted = spectra * frequency_widths(freq) * (2 * np.pi / len(theta_deg))
    m0 = weighted.sum(axis=(-2, -1))
    m1 = (weighted * freq).sum(axis=(-2, -1))
    angle = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))[:, None]
    east = (weighted * np.sin(angle)).sum(axis=(-2, -1))
    north = (weighted * np.cos(angle)).sum(axis=(-2, -1))
    period = np.divide(m0, m1, out=np.zeros_like(m0), where=m1 > 0)
    direction = np.rad2deg(np.arctan2(east, north)) % 360
    return dict(swh=4 * np.sqrt(m0), mwp=period, mwd=direction,
                directional_resultant=np.hypot(east, north), m0=m0)


def load_case(config, data_root):
    """Load reference data; only t=0 is used to reconstruct the separable initial field.

    Full-grid hs is [time,y,x]; waden uses [batch,x,y,direction,frequency].
    The archived cases have a common initial spectral shape wherever hs>0.
    The constant prescribed spectra are from this same initial condition.
    No future interior reference is fed to the simulation.
    """
    folder = Path(data_root) / config["data_directory"] / "input"
    grid_path = folder / config["grid_file"]
    with nc.Dataset(grid_path) as ds:
        xname, yname = ("longitude", "latitude") if config["coordinate_type"] == "Spherical" else ("x", "y")
        x, y = np.asarray(ds[xname][:]), np.asarray(ds[yname][:])
        times = times_from_variable(ds["time"])
        mapsta = np.asarray(ds["MAPSTA"][:]).T
        wet, boundary_mask = np.isin(mapsta, (1, 2)), mapsta == 2
        fields = {key: np.ma.filled(ds[name][:], np.nan).transpose(0, 2, 1)
                  for key, name in (("swh", "hs"), ("mwd_from", "dir"), ("mwp", "t01"))
                  if name in ds.variables}
        observed_depth = np.ma.filled(ds["dpt"][0], np.nan).T
        provenance = dict(ww3_version=ds.WAVEWATCH_III_version_number,
                          ww3_compiled_switches=ds.WAVEWATCH_III_switches,
                          grid_file=str(grid_path.resolve()))

    extension = config["bathymetry"]
    if "x_profile_m" in extension:
        depth = np.broadcast_to(np.asarray(extension["x_profile_m"])[:, None], wet.shape).copy()
    else:
        depth = np.full(wet.shape, extension["constant_m"], dtype=float)
    # Check physical WW3 depths, then align excluded cells with the wet map.
    np.testing.assert_allclose(depth[wet], observed_depth[wet], rtol=0, atol=1e-5)
    depth[~wet] = 0
    # Restore positive depth for prescribed zero-spectrum outgoing ghosts.
    # Physical wet-cell depths and the scoring mask retain their input values.
    edge_slices = {"x_high": ((-1, slice(None)), (-2, slice(None))),
                   "x_low": ((0, slice(None)), (1, slice(None))),
                   "y_high": ((slice(None), -1), (slice(None), -2)),
                   "y_low": ((slice(None), 0), (slice(None), 1))}
    for edge in config.get("outflow_ghost_edges", []):
        ghost, interior = edge_slices[edge]
        if np.any(wet[ghost]):
            raise ValueError("An outflow ghost must not replace a reference ocean cell")
        depth[ghost] = depth[interior]
        boundary_mask[ghost] = True

    paths = []
    for pattern in config["spectrum_files"]:
        matches = sorted(folder.glob(pattern))
        if not matches:
            raise FileNotFoundError(folder / pattern)
        paths.extend(matches)
    station_spectra, station_x, station_y, station_names = [], [], [], []
    for index, path in enumerate(paths):
        with nc.Dataset(path) as ds:
            current_times = times_from_variable(ds["time"])
            current_freq = np.asarray(ds["frequency"][:])
            current_theta = np.asarray(ds["direction"][:])
            if index == 0:
                spectrum_times, freq, theta = current_times, current_freq, current_theta
            else:
                np.testing.assert_array_equal(spectrum_times, current_times)
                np.testing.assert_array_equal(freq, current_freq)
                np.testing.assert_array_equal(theta, current_theta)
            if "to_direction" not in ds["direction"].standard_name:
                raise ValueError("These cases require WW3 going-to spectral directions")
            station_spectra.append(np.ma.filled(ds["efth"][:], np.nan).transpose(0, 1, 3, 2))
            scale = config["station_coordinate_scale"]
            station_x.extend(np.asarray(ds[xname][0]) * scale)
            station_y.extend(np.asarray(ds[yname][0]) * scale)
            station_names.extend(nc.chartostring(ds["station_name"][:]).tolist())
    station_spectra = np.concatenate(station_spectra, axis=1)
    if spectrum_times[0] != times[0]:
        raise ValueError("The first spectrum must describe the grid's initial instant")
    initial_station = np.argmax(np.nansum(station_spectra[0], axis=(-2, -1)))
    template = station_spectra[0, initial_station]
    template_m0 = moments(template, theta, freq)["m0"]
    if not np.isfinite(template_m0) or template_m0 <= 0:
        raise ValueError("Missing positive initial station spectrum")
    hs0 = np.asarray(np.where(wet, fields["swh"][0], 0), dtype=np.float64)
    initial = hs0[..., None, None]**2 / (16 * template_m0) * template
    # Every observed nonzero initial station must have this separable shape.
    for spectrum in station_spectra[0]:
        integral = moments(spectrum, theta, freq)["m0"]
        if integral > 0:
            np.testing.assert_allclose(spectrum / integral, template / template_m0, rtol=2e-5, atol=1e-7)

    station_x, station_y = np.asarray(station_x), np.asarray(station_y)
    ix = np.argmin(abs(station_x[:, None] - x), axis=1)
    iy = np.argmin(abs(station_y[:, None] - y), axis=1)
    np.testing.assert_allclose(x[ix], station_x, atol=float(np.min(abs(np.diff(x)))) * 1e-4, rtol=0)
    np.testing.assert_allclose(y[iy], station_y, atol=float(np.min(abs(np.diff(y)))) * 1e-4, rtol=0)
    if not np.all(wet[ix, iy]):
        raise ValueError("A reference station is outside the active grid")
    provenance.update(spectrum_files=[str(p.resolve()) for p in paths],
                      archived_revision=config["reference_revision"],
                      initial_condition="t=0 hs envelope times common t=0 directional spectrum",
                      boundary_condition="constant t=0 spectrum at MAPSTA=2",
                      outflow_ghost_edges=config.get("outflow_ghost_edges", []),
                      station_coordinate_scale=config["station_coordinate_scale"])
    return dict(x=x, y=y, times=times, wet=wet, boundary_mask=boundary_mask,
                depth=depth, initial=initial, theta=theta, freq=freq,
                fields=fields, station_spectra=station_spectra,
                spectrum_times=spectrum_times, station_ix=ix, station_iy=iy,
                station_names=np.asarray(station_names), provenance=provenance)
