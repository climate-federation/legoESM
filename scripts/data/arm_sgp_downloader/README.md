# arm_sgp_downloader

Retrieves and harmonises DOE ARM Southern Great Plains Central Facility tower, flux and surface data.
Python 3.11 or newer. `pip install -r requirements.txt`.

    python -m arm_sgp_downloader.cli discover
    python -m arm_sgp_downloader.cli download --period all
    python -m arm_sgp_downloader.cli process
    python -m arm_sgp_downloader.cli report

The root directory is `--root`, or the `ARM_SGP_ROOT` environment variable.

## Access

Anonymous access does not reach the ARM archive. ARM Live needs a token, the EOL GAPP ECOR dataset needs a site key,
and AmeriFlux US-ARM needs an identity. With no credential the package fetches what is public: ARM-DOE/arm-test-data
samples, LCRC ARM diagnostics, PLUMBER2 US-ARM (2003-2012) and Zenodo record 14261417 (raw E14 reference data).

With `~/.arm_live` (mode 600, `username:token`) it also fetches the 32 streams in `config.ARMLIVE_STREAMS`. The token
is added only at request time and is never written to the catalogue or to error messages. Raw 10 Hz wind and gas
streams, about 470 GB, are left out.

## Layout

    raw/<source>/          originals, read-only, never replaced
    catalogue.jsonl        URL, time, size, checksum and status of each file
    discovery.json         inventory and access probe results
    processed/<stream>.nc  each stream on the 30-minute grid, unfiltered, flags kept
    unified/sgp_cf_30min.nc
    report/                figures, validation.json, summary.csv, manifest.json

Downloads resume from `.part` files, check `Content-Range`, and refuse a second concurrent run through
`.download.lock`. ARM Live gives no checksum, so its files are checked by length and NetCDF signature only.

## Time

Every time is UTC and labels the end of its interval. Rules are per stream in `netcdf_reader.STREAMS`, read from file
attributes. Two come from measurement, not attributes:

- `sgp30qcecorE14.c1` repeats ECOR's times but its bounds sit 30 minutes earlier. ECOR labelling is used: the two
  series correlate at r = 1 with it.
- `sgpqcrad1longE13`, `sgpswatsE13` and `sgpstampE13` state nothing and are assumed to label interval end.

Records finer than 30 minutes are averaged within the interval, and an interval with under 80% of its records stays
missing. Precipitation is summed. Nothing is interpolated.

## Quality control

`processed/` is unfiltered. The unified dataset blanks only what its file rejects, then applies the loose limits in
`quality_control.LIMITS`. ARM uses four flag conventions, so non-zero alone means nothing:

| Files | Blanked | Kept, flag preserved |
|---|---|---|
| Tower flux after 2015, integer grades | Bad (9, 10), "use for orientation only" (7, 8) | 1-6 |
| Tower flux before 2015, bits in global text | 1 missing, 2 out of range, 4 dependency failed | 8, 16, 32 |
| Older tower and SWATS, `0x` bits in global text | every stated bit except the jump test | `valid_delta` (later files of these streams grade it Indeterminate) |
| Others, bit or integer assessments | Bad | Indeterminate |

A flag value or bit with no stated meaning is blanked with a warning. Soil values are mean over pits at each depth within one instrument. Where SWATS and STAMP share a depth (5 cm, 2016-2017) STAMP wins, because they differ by 0.09 m3 m-3 in water content.

## Unified dataset

Dimensions `time`, `height` (2, 4, 25, 60 m nominal), `soil_depth`. Levels that were not measured stay empty:
T, RH, q and p exist at 2, 25, 60 m; wind and fluxes at 4, 25, 60 m. LE after 2015 exists at 4 m only.
ECOR is at E14, about 3 m. Tower-level fluxes are not surface fluxes.
`tau_x` and `tau_y` are absent: the files give stress as a magnitude or in the sonic or rotated frame, not east and north components. Use `ustar` and `ustar_level` for its size.

Variables merged from several streams carry `<name>_source` (index into the `sources_*` attribute, -1 for none) and
`<name>_qc`, the raw flag of the stream that supplied the record. Wind speed and direction always come from one stream.
All fluxes are positive upward; EBBR's sign is reversed on reading, since its files are positive toward the surface.

## Tests

    cd scripts/data && python -m pytest arm_sgp_downloader/tests -q
