"""Build per-stream products and the unified (time, height) dataset."""
import json
import re

import numpy as np
import pandas as pd
import xarray as xr

from . import netcdf_reader, quality_control, time_alignment

HEIGHTS = (2, 4, 25, 60)
ATTRS = {
    "T": ("air_temperature", "degC"), "RH": ("relative_humidity", "percent"), "q": ("specific_humidity", "kg kg-1"),
    "p": ("air_pressure", "Pa"), "u": ("eastward_wind", "m s-1"), "v": ("northward_wind", "m s-1"),
    "H": ("upward_sensible_heat_flux_in_air", "W m-2"), "LE": ("upward_latent_heat_flux_in_air", "W m-2"),
    "ustar_level": (None, "m s-1"), "L_level": (None, "m"),
    "H_surface": ("surface_upward_sensible_heat_flux", "W m-2"), "LE_surface": ("surface_upward_latent_heat_flux", "W m-2"),
    "H_ebbr": ("surface_upward_sensible_heat_flux", "W m-2"), "LE_ebbr": ("surface_upward_latent_heat_flux", "W m-2"),
    "ustar": (None, "m s-1"), "CO2_flux": (None, "umol m-2 s-1"), "Rn": ("surface_net_downward_radiative_flux", "W m-2"),
    "G": ("downward_heat_flux_at_ground_level_in_soil", "W m-2"), "SW_down": ("surface_downwelling_shortwave_flux_in_air", "W m-2"),
    "SW_up": ("surface_upwelling_shortwave_flux_in_air", "W m-2"), "LW_down": ("surface_downwelling_longwave_flux_in_air", "W m-2"),
    "LW_up": ("surface_upwelling_longwave_flux_in_air", "W m-2"),
    "precipitation": (None, "mm per 30 min"), "surface_pressure": ("surface_air_pressure", "Pa"),
    "soil_temperature": ("soil_temperature", "degC"), "soil_moisture": (None, "m3 m-3"),
}


def specific_humidity(mixing_ratio):
    return mixing_ratio / (1.0 + mixing_ratio)


SOIL_STREAMS = ("sgpstampE13.b1", "sgpswatsE13.b1")


def wind_components(speed, direction):
    radians = np.deg2rad(direction)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def soil(tables, prefix):
    """Mean over pits at each depth within a stream, first stream wins at a shared depth: SWATS and STAMP differ
    by 0.09 m3 m-3 at 5 cm. Depths come from the files, so unmeasured depths stay absent."""
    out = {}
    for table in tables.values():
        columns = {}
        for name in table.columns:
            match = re.fullmatch(prefix + r"_[A-Z]_([0-9.]+)", name)
            if match:
                columns.setdefault(float(match.group(1)), []).append(table[name])
        for depth, series in columns.items():
            mean = pd.concat(series, axis=1).mean(axis=1)
            out[depth] = out[depth].combine_first(mean) if depth in out else mean
    return dict(sorted(out.items()))


def write_stream(root, stream, table):
    out = root / "processed" / f"{stream}.nc"
    out.parent.mkdir(exist_ok=True)
    data = table.tz_localize(None).rename_axis("time").to_xarray()
    data.attrs.update(source_datastream=stream, time_label="end of averaging interval, UTC",
                      time_rule=netcdf_reader.STREAMS[stream]["time"])
    data.to_netcdf(out)


def merged(tables, names):
    """First stream wins where it has data, so no value is ever invented."""
    series = None
    for table, name in names:
        if name in tables.get(table, {}):
            current = tables[table][name]
            series = current if series is None else series.combine_first(current)
    return series


def merged_records(tables, streams, names):
    """Like merged, but every name in a record comes from the same stream."""
    frame = None
    for code, stream in enumerate(streams):
        table = tables.get(stream, {})
        if all(name in table for name in names):
            rows = table[list(names)].dropna().assign(source=code)
            frame = rows if frame is None else frame.combine_first(rows)
    return frame


def source_codes(tables, names, time):
    """Index in names of the stream that supplied each record; -1 where none did."""
    codes = np.full(len(time), -1, "i1")
    for code, (table, name) in reversed(list(enumerate(names))):
        if name in tables.get(table, {}):
            codes[tables[table][name].reindex(time).notna().to_numpy()] = code
    return codes


def source_flags(tables, names, time, codes):
    """Raw flag of the stream that supplied each record; its meaning depends on that stream."""
    flags = np.full(len(time), np.nan)
    for code, (table, name) in enumerate(names):
        if "qc_" + name in tables.get(table, {}):
            chosen = codes == code
            flags[chosen] = tables[table]["qc_" + name].reindex(time).to_numpy(float)[chosen]
    return flags


def build(root):
    tables = {}
    for stream in netcdf_reader.STREAMS:
        raw = netcdf_reader.read_stream(root, stream)
        if len(raw):
            step = time_alignment.native_seconds(raw)
            write_stream(root, stream, time_alignment.to_grid(raw, step))
            tables[stream] = time_alignment.to_grid(quality_control.apply_flags(raw), step)
    if not tables:
        raise ValueError(f"no readable ARM files under {root}/raw")
    time = time_alignment.axis(list(tables.values()))
    soil_tables = {name: tables[name] for name in SOIL_STREAMS if name in tables}
    temperature, moisture = soil(soil_tables, "Tsoil"), soil(soil_tables, "SM")
    depths = sorted(set(temperature) | set(moisture))
    ds = xr.Dataset(coords={"time": time.tz_localize(None), "height": ("height", np.array(HEIGHTS, float)),
                            "soil_depth": ("soil_depth", np.array(depths, float))})

    def grid(series):
        return np.full(len(time), np.nan) if series is None else series.reindex(time).to_numpy(float)

    flag_note = {"long_name": "raw ARM flag of the supplying stream; meaning differs by stream; "
                              "missing for streams averaged to 30 min, whose flag fractions are in processed/<stream>.nc"}

    def source_attrs(streams_by_height):
        return {"missing": -1, **{f"sources_{key}": " ".join(streams) for key, streams in streams_by_height.items()}}

    def column(out, mapper):
        codes = np.stack([source_codes(tables, mapper(h), time) for h in HEIGHTS], axis=1)
        ds[out] = (("time", "height"), np.stack([grid(merged(tables, mapper(h))) for h in HEIGHTS], axis=1))
        ds[out + "_source"] = (("time", "height"), codes, source_attrs({f"{h}m": [s for s, _ in mapper(h)] for h in HEIGHTS}))
        ds[out + "_qc"] = (("time", "height"), np.stack([source_flags(tables, mapper(h), time, codes[:, i])
                                                         for i, h in enumerate(HEIGHTS)], axis=1), flag_note)

    def single(out, names):
        codes = source_codes(tables, names, time)
        ds[out] = ("time", grid(merged(tables, names)))
        ds[out + "_source"] = ("time", codes, source_attrs({"all": [s for s, _ in names]}))
        ds[out + "_qc"] = ("time", source_flags(tables, names, time, codes), flag_note)

    def by_depth(values):
        return np.stack([grid(values.get(d)) for d in depths], axis=1) if depths else np.empty((len(time), 0))

    twr = ("sgp30twrmrC1.c1", "sgp1twrmrC1.c1")
    flux = {4: ("sgp30co2flx4mC1.b1", "sgpco2flx4mC1.b1"), 25: ("sgp30co2flx25mC1.b1", "sgpco2flx25mC1.b1"),
            60: ("sgp30co2flx60mC1.b1", "sgpco2flx60mC1.b1")}

    def from_twr(name):
        return lambda h: [(s, f"{name}_{h}") for s in twr]

    def from_flux(name):
        return lambda h: [(s, name) for s in flux.get(h, ())]

    for out in ("T", "RH", "p"):
        column(out, from_twr(out))
    column("q", from_twr("r"))
    ds["q"] = (("time", "height"), specific_humidity(ds.q.values))
    wind = [merged_records(tables, flux.get(h, ()), ("wspd", "wdir")) for h in HEIGHTS]
    speed = np.stack([grid(None if w is None else w.wspd) for w in wind], axis=1)
    direction = np.stack([grid(None if w is None else w.wdir) for w in wind], axis=1)
    ds["u"], ds["v"] = [(("time", "height"), component) for component in wind_components(speed, direction)]
    codes = np.stack([grid(None if w is None else w.source) for w in wind], axis=1)
    ds["wind_source"] = (("time", "height"), np.nan_to_num(codes, nan=-1).astype("i1"),
                         source_attrs({f"{h}m": flux.get(h, ()) for h in HEIGHTS}))
    for out, name in (("wind_speed_qc", "wspd"), ("wind_direction_qc", "wdir")):
        ds[out] = (("time", "height"), np.stack([source_flags(tables, [(s, name) for s in flux.get(h, ())], time, ds.wind_source.values[:, i])
                                                 for i, h in enumerate(HEIGHTS)], axis=1), flag_note)
    column("H", from_flux("H"))
    column("LE", from_flux("LE"))
    column("ustar_level", from_flux("ustar"))
    column("L_level", from_flux("L"))
    surface = ("sgp30qcecorE14.c1", "sgp30ecorE14.b1", "sgpecorsfE14.b1")
    single("H_surface", [(s, "H") for s in surface])
    single("LE_surface", [(s, "LE") for s in surface])
    single("ustar", [("sgp30ecorE14.b1", "ustar"), ("sgpecorsfE14.b1", "ustar")])
    single("CO2_flux", [(s, "co2_flux") for s in surface])
    ebbr = "sgp30baebbrE13.c1"
    for out, name in (("H_ebbr", "H"), ("LE_ebbr", "LE"), ("Rn", "Rn"), ("G", "G")):
        ds[out] = ("time", grid(merged(tables, [(ebbr, name)])))
    rad = ("sgpqcrad1longE13.c2", "sgpqcrad1longE13.c1")
    for name in ("SW_down", "SW_up", "LW_down", "LW_up"):
        single(name, [(s, name) for s in rad])
    ds["precipitation"] = ("time", grid(merged(tables, [("sgpmetE13.b1", "precip")])))
    ds["surface_pressure"] = ("time", grid(merged(tables, [("sgpmetE13.b1", "p_sfc")])))
    ds["soil_temperature"] = (("time", "soil_depth"), by_depth(temperature))
    ds["soil_moisture"] = (("time", "soil_depth"), by_depth(moisture))
    for name, (standard, units) in ATTRS.items():
        ds[name].attrs["units"] = units
        if standard:
            ds[name].attrs["standard_name"] = standard
    ds.time.attrs.update(long_name="end of averaging interval", standard_name="time")
    ds.height.attrs.update(units="m", positive="up", long_name="nominal height above ground",
                           comment="T, RH, q, p at 2, 25, 60 m (tower); wind and fluxes at 4, 25, 60 m (flux systems). "
                                   "Tower-level fluxes are not surface fluxes.")
    ds.H_surface.attrs["comment"] = "ECOR at E14, quality-controlled where available; about 3 m"
    ds.attrs.update(Conventions="CF-1.8", title="ARM SGP Central Facility 30-minute tower and surface fluxes",
                    institution="DOE ARM user facility", source=json.dumps(sorted(tables)))
    out = root / "unified" / "sgp_cf_30min.nc"
    out.parent.mkdir(exist_ok=True)
    ds.to_netcdf(out)
    return ds, tables
