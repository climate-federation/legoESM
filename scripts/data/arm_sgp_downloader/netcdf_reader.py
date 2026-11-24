"""Read ARM files into tables indexed by the END of each averaging interval, in UTC."""
import re
import warnings
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

ARM_MISSING = -9000.0  # ARM writes -9999-style sentinels, sometimes without a fill attribute


# EBBR fluxes are positive toward the surface, measured: midday H and LE are negative while Rn is positive.
TOWARD_SURFACE = -1.0


def profile(heights, names, suffix="{h:02d}m"):
    return {f"{out}_{h}": (src + "_" + suffix.format(h=h), scale) for h in heights for out, src, scale in names}


TWRMR = [("T", "temp", 1.0), ("RH", "rh", 1.0), ("e", "vap_pres", 100.0), ("r", "mixing_ratio", 1e-3), ("p", "pres", 100.0)]
FLUX_OLD = [("H", "h"), ("LE", "le"), ("ustar", "ustar"), ("wspd", "mean_rot_u"), ("wdir", "wdir"), ("L", "Lmoni"),
            ("Tsonic", "mean_t"), ("var_u", "var_rot_u"), ("var_v", "var_rot_v"), ("var_w", "var_rot_w"),
            ("var_T", "var_t"), ("zm", "zm"), ("co2_flux", "fc_wpl_le")]
FLUX_NEW = [("H", "sensible_heat_flux"), ("LE", "latent_heat_flux"), ("ustar", "friction_velocity"),
            ("wspd", "wind_speed"), ("wdir", "wind_direction"), ("L", "monin_obukov_length"),
            ("Tsonic", "sonic_temperature"), ("tau", "momentum_flux"), ("co2_flux", "co2_flux"), ("h2o_flux", "h2o_flux")]


def plain(pairs):
    return {out: (src, 1.0) for out, src in pairs}


# time rule: "bounds" reads time_bounds; "start_stamp" follows the variable's stated start of interval;
# "end" follows the stated end-of-interval comment; "assumed_end" has no statement in the file.
# "time_is_start": QC ECOR repeats ECOR's time values but its bounds sit 30 min earlier; the two flux
# series match exactly (r=1) only with the ECOR labelling, which tower flux and radiation support.
STREAMS = {
    "sgp30twrmrC1.c1": {"time": "end", "vars": profile((2, 25, 60), TWRMR)},
    "sgp1twrmrC1.c1": {"time": "bounds", "vars": profile((2, 25, 60), TWRMR)},
    "sgp30co2flx4mC1.b1": {"time": "start_stamp", "vars": plain(FLUX_OLD), "height": 4},
    "sgp30co2flx25mC1.b1": {"time": "start_stamp", "vars": plain(FLUX_OLD), "height": 25},
    "sgp30co2flx60mC1.b1": {"time": "start_stamp", "vars": plain(FLUX_OLD), "height": 60},
    "sgpco2flx4mC1.b1": {"time": "bounds", "vars": plain(FLUX_NEW), "height": 4},
    "sgpco2flx25mC1.b1": {"time": "bounds", "vars": plain(FLUX_NEW), "height": 25},
    "sgpco2flx60mC1.b1": {"time": "bounds", "vars": plain(FLUX_NEW), "height": 60},
    "sgp30ecorE14.b1": {"time": "bounds", "vars": plain([
        ("wspd", "wind_spd"), ("wdir", "wind_dir"), ("ustar", "ustar"), ("H", "h"), ("LE", "lv_e"), ("co2_flux", "fc"),
        ("uw", "cvar_rot_uw"), ("vw", "cvar_rot_vw"), ("wT", "cvar_rot_wt"), ("wq", "cvar_rot_wq"),
        ("var_T", "var_t"), ("var_w", "var_rot_w")])},
    "sgp30qcecorE14.c1": {"time": "time_is_start", "vars": plain([
        ("H", "corrected_sensible_heat_flux"), ("LE", "corrected_latent_heat_flux"), ("co2_flux", "corrected_co2_flux")])},
    "sgpecorsfE14.b1": {"time": "bounds", "vars": plain([
        ("H", "sensible_heat_flux"), ("LE", "latent_flux"), ("ustar", "friction_velocity"), ("tau", "momentum_flux"),
        ("co2_flux", "co2_flux")])},
    "sgp30baebbrE13.c1": {"time": "end", "vars": {
        "H": ("sensible_heat_flux", TOWARD_SURFACE), "LE": ("latent_heat_flux", TOWARD_SURFACE),
        "G": ("surface_soil_heat_flux_avg", TOWARD_SURFACE), "Rn": ("net_radiation", 1.0)}},
    "sgpqcrad1longE13.c2": {"time": "assumed_end", "vars": plain([
        ("SW_down", "BestEstimate_down_short_hemisp"), ("SW_up", "up_short_hemisp"),
        ("LW_down", "down_long_hemisp"), ("LW_up", "up_long_hemisp")])},
    "sgpqcrad1longE13.c1": {"time": "assumed_end", "vars": plain([
        ("SW_down", "BestEstimate_down_short_hemisp"), ("SW_up", "up_short_hemisp"),
        ("LW_down", "down_long_hemisp"), ("LW_up", "up_long_hemisp")])},
    "sgpmetE13.b1": {"time": "end", "vars": plain([("precip", "tbrg_precip_total"), ("p_sfc", "atmos_pressure")])},
    "sgpswatsE13.b1": {"time": "assumed_end", "depth_vars": {
        "Tsoil_W": ("tsoil_W", 1.0), "Tsoil_E": ("tsoil_E", 1.0), "SM_W": ("watcont_W", 1.0), "SM_E": ("watcont_E", 1.0)}},
    "sgpstampE13.b1": {"time": "assumed_end", "depth_vars": {
        "Tsoil_W": ("soil_temperature_west", 1.0), "Tsoil_S": ("soil_temperature_south", 1.0),
        "Tsoil_E": ("soil_temperature_east", 1.0), "SM_W": ("soil_specific_water_content_west", 0.01),
        "SM_S": ("soil_specific_water_content_south", 0.01), "SM_E": ("soil_specific_water_content_east", 0.01)}},
}
PRESSURE_KPA = {"p_sfc"}
# Pre-2015 tower flux files explain their bits only in global text: 1 missing, 2 out of range, 4 dependency failed.
# Bits 8, 16 and 32 mark suspect values, which stay in the filtered product with their flag.
MENTOR_FAIL_BITS = 1 | 2 | 4
ORIENTATION_ONLY = "use for orientation only"
# Later files of the same streams grade this one test Indeterminate and every other stated bit Bad.
SUSPECT_BIT_TEXT = "valid_delta"


def _numbered(holder, pattern):
    matches = ((re.fullmatch(pattern, name), name) for name in holder.ncattrs())
    return {int(match[1]): str(getattr(holder, name)) for match, name in matches if match}


def _unexplained(variable, mask):
    """A flag or bit the file does not explain is rejected, never silently."""
    if mask.any():
        warnings.warn(f"{variable.name}: {int(mask.sum())} flags with no stated meaning rejected")
    return mask


def _text_bits(data):
    """Bits stated as lines of "0x4 = meaning" in the global text of older tower and SWATS files."""
    text = getattr(data, "Mentor_QC_Field_Information", "")
    return {int(bit, 16): meaning for bit, meaning in re.findall(r"0x([0-9a-fA-F]+) = ([^\n]*)", text) if int(bit, 16)}


def rejected(data, variable, flags):
    """Where the file's own flag meanings reject a value. ARM has four dialects, and non-zero alone means nothing:
    post-2015 tower fluxes put a quality grade on every record."""
    flags = np.where(flags > 0, flags, 0).astype(np.int64)
    if "MENTOR_QC_FIELD_INFORMATION" in data.ncattrs():
        return flags & MENTOR_FAIL_BITS != 0
    grades = _numbered(variable, r"flag_(\d+)_assessment")
    if grades:
        notes = _numbered(variable, r"flag_(\d+)_description")
        out = [n for n, grade in grades.items() if grade == "Bad" or ORIENTATION_ONLY in notes.get(n, "").lower()]
        return np.isin(flags, out) | _unexplained(variable, ~np.isin(flags, [0, *grades]))
    bits = _numbered(variable, r"bit_(\d+)_assessment") or _numbered(data, r"qc_bit_(\d+)_assessment")
    if bits:
        unknown = flags & ~sum(1 << (n - 1) for n in bits) != 0
        return (flags & sum(1 << (n - 1) for n, grade in bits.items() if grade == "Bad") != 0) | _unexplained(variable, unknown)
    stated = _text_bits(data)
    if stated:
        bad = sum(bit for bit, meaning in stated.items() if SUSPECT_BIT_TEXT not in meaning)
        return (flags & bad != 0) | _unexplained(variable, flags & ~sum(stated) != 0)
    return _unexplained(variable, flags != 0)


def _flags(data, source):
    """Raw flag and rejection mask, or None for a variable without a flag."""
    if "qc_" + source not in data.variables:
        return None
    variable = data["qc_" + source]
    raw = np.ma.filled(np.ma.asarray(variable[:], dtype=float), np.nan)
    return raw, rejected(data, variable, np.nan_to_num(raw)).astype(float)


def _epoch_seconds(data):
    if "base_time" in data.variables and "time_offset" in data.variables:
        return np.asarray(data["base_time"][...], float) + np.asarray(data["time_offset"][:], float), 0.0
    variable = data["time"]
    origin = re.search(r"since (\d{4}-\d{1,2}-\d{1,2})[ T](\d{1,2}:\d{2}:\d{2})", variable.units)
    base = pd.Timestamp(f"{origin.group(1)} {origin.group(2)}", tz="UTC").timestamp()
    factor = 60.0 if variable.units.startswith("minutes") else 1.0
    return base + np.asarray(variable[:], float) * factor, base


def _ddd_times(data):
    variable = data["yyyydddhhmmss"]
    if "start" not in variable.long_name:
        raise ValueError(f"unknown stamp meaning: {variable.long_name}")
    text = ["%013d" % int(value) for value in np.asarray(variable[:], float)]
    return pd.to_datetime(text, format="%Y%j%H%M%S", utc=True) + pd.Timedelta(minutes=30)


def interval_end(data, rule):
    if "yyyydddhhmmss" in data.variables:
        return _ddd_times(data)
    seconds, base = _epoch_seconds(data)
    if rule == "time_is_start":
        return pd.to_datetime(seconds + 1800.0, unit="s", utc=True)
    if rule == "bounds":
        if "time_bounds" not in data.variables:
            raise ValueError("stream declared to carry time_bounds has none")
        bounds = np.asarray(data["time_bounds"][:], float)
        units = data["time"].units if "time" in data.variables else ""
        factor = 60.0 if units.startswith("minutes") else 1.0
        origin = base if base else float(np.ravel(data["base_time"][...])[0])
        seconds = origin + bounds[:, 1] * factor
    return pd.to_datetime(seconds, unit="s", utc=True)


def _values(variable, scale):
    values = np.ma.filled(np.ma.asarray(variable[:], dtype=float), np.nan)
    for name in ("missing_value", "missing_data"):
        if name in variable.ncattrs():
            values[values == float(getattr(variable, name))] = np.nan
    values[values <= ARM_MISSING] = np.nan
    return values * scale


def read_file(path, stream):
    spec = STREAMS[stream]
    with netCDF4.Dataset(path) as data:
        index = interval_end(data, spec["time"])
        table = {}
        for out, (source, scale) in spec.get("depth_vars", {}).items():
            source = source if source in data.variables else source.lower()  # 2001-2005 SWATS files use lower case
            if source in data.variables:
                depths = np.asarray(data["depth"][:], float)
                for column, depth in zip(_values(data[source], scale).T, depths):
                    table[f"{out}_{depth:g}"] = column
                flags = _flags(data, source)
                if flags:
                    for raw, fail, depth in zip(flags[0].T, flags[1].T, depths):
                        table[f"qc_{out}_{depth:g}"], table[f"qc_fail_{out}_{depth:g}"] = raw, fail
        for out, (source, scale) in spec.get("vars", {}).items():
            if source not in data.variables:
                continue
            table[out] = _values(data[source], scale * (1000.0 if out in PRESSURE_KPA else 1.0))
            flags = _flags(data, source)
            if flags:
                table["qc_" + out], table["qc_fail_" + out] = flags
    return pd.DataFrame(table, index=index.rename("time_end"))


def read_stream(root, stream):
    # Test samples are trimmed copies, used only when the archive itself was not reachable.
    paths = sorted(Path(root).glob(f"raw/armlive/{stream}/*")) or sorted(Path(root).glob(f"raw/arm-test-data/{stream}.*"))
    frames = [read_file(path, stream) for path in paths if not path.name.endswith(".part")]
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames).sort_index()
    return table[~table.index.duplicated(keep="first")]
