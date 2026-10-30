# SWG double-gyre round 4 climate-comparison preregistration

Date: 2026-10-12. Incoming tip: `402ba0c4b9bcf2bde0c384994bd4c45ae1f834de`.
The NEMO long-run daily variables have not been read: its 32 per-rank files
currently have an empty time dimension. The legoESM run has only 40 daily
files. Therefore no year-1 comparison value is known when this protocol is
frozen.

## Fixed comparison window and grid

The first report is labelled **year 1, spin-up** and uses exactly model days
1--365 from both models at 6.25 km on the 640 by 640 wet basin. A missing day,
duplicate day, non-finite wet value, non-fp32 NEMO field, mismatched clock, or
different grid refuses the comparison. Years 2--3 remain the registered
climate window and are OPEN until both three-year runs exist.

The NEMO reader assembles all per-rank IOIPSL files by their `nav_lon` and
`nav_lat` coordinates and requires each global point exactly once after halo
deduplication. It reads `sossheig`, `vozocrtx`, and `vomecrty`. The legoESM
reader consumes the driver's daily `ssh`, `u`, and `v` arrays plus its manifest.
Both native U/V fields are converted to tracer-cell velocity by the repository
shared `velocity_to_cell_centre` diagnostic before any comparison.

## Frozen diagnostics

For each model, on the identical wet mask and window, the instrument produces:

1. time-mean SSH, zonal velocity, and meridional velocity maps;
2. the time mean of instantaneous speed;
3. an EKE map, `0.5 mean((u-mean(u))^2 + (v-mean(v))^2)`;
4. area-weighted basin-mean KE and EKE at every daily sample;
5. at every wet longitude, the latitude and value of the maximum time-mean
   eastward zonal velocity; this complete speed-versus-longitude profile is the
   threshold-free eastward-extension diagnostic. No unasked cutoff is used to
   turn it into a scalar jet length;
6. the time mean of the repository shared isotropic kinetic-energy spectrum
   evaluated on each daily cell-centred velocity field.

The JSON report records the window, provenance, scalar comparisons, array
shapes and SHA-256 digests. A compressed NPZ carries every map, time series,
jet profile, and spectrum. One side-by-side PNG uses common limits for each
paired map and overlays both models' KE/EKE time series, jet profiles, and
spectra.

No pass/fail climate tolerance is introduced in Round 4. This is a descriptive
spin-up comparison, not an honesty-bar replacement. The preregistered honesty
bar remains bit identity at every stage/month row.

## Controls

A direct synthetic test uses constant plus separable time-varying velocities,
with analytic mean fields, speed, EKE, KE/EKE series, jet latitude/profile, and
spectral equality between two identical inputs. Coordinate assembly is tested
with overlapping halo tiles and refuses a missing interior point, a conflicting
duplicate, and a shifted clock.

CONFIRM the instrument only if all analytic values are reproduced and each
planted violation refuses or changes the report. REFUTE it on any silent
coordinate overwrite, unequal-window acceptance, mask asymmetry, or metric
computed from different staggering on the two sides.

## Sources

Maddison's open-access study defines mean kinetic energy from the mean velocity,
EKE from velocity departures, and diagnoses time-averaged kinetic-energy
spectra; its jet-extension comparison is visual rather than a published scalar
cutoff: <https://doi.org/10.1029/2024MS004883>.

The NEMO output variables and daily IOIPSL association are the executing SWG_DG
definitions in `tests/SWG_DG/EXPREF/file_def_nemo-oce.xml` and the compiled
`iom_put` calls in `tests/SWG_DG/BLD/ppsrc/nemo/diawri.f90`. The model-side
conversion and spectrum are the existing functions in
`packages/ocean/legoesm/ocean/diagnostics.py`.
