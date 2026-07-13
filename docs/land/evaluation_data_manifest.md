# Land-evaluation data manifest

Authoritative list of the datasets the land evaluation pipeline
(`docs/land/evaluation_pipeline.md`) uses, how each is obtained, and where it
lands. **Data directories are gitignored** — a fresh clone has none of the
downloaded/staged data. Reconstruct it per the per-dataset policy below;
`scripts/data/fetch_land_eval_data.py --check` reports what is present.

Access classes:

- **in-tree** — ships in the repo, nothing to fetch.
- **fetch** — reproducible download via `fetch_land_eval_data.py` or a
  documented account-gated portal.
- **bundle** — staged/large; place under `$LEGO_LAND_EVAL_DATA` (a shared
  filesystem/cloud root) at the listed relative path.

| # | Dataset | Class | Location (repo-relative, or `$LEGO_LAND_EVAL_DATA/…`) | Size | Source / DOI |
| --- | --- | --- | --- | --- | --- |
| in-tree | CHATS7 forcing NetCDFs (2007-04, 2007-05) | in-tree | `clm-ml-jax/src/input_files/tower-forcing/CHATS7/` | ~9 MB | gbonan/CLM-ml_v2.CHATS |
| in-tree | CHATS7 CLM initial state + soil-moisture correction | in-tree | `clm-ml-jax/src/input_files/clm5_0/CHATS7/` | — | ditto |
| `fortran_v2` | Fortran CLM-ML v2 reference `.out` (flux/fsun/aux/profile) | in-tree | `docs/output_files_clm_ml-v2/` | — | reference build |
| `jax_standalone` | JAX-standalone CLM-ML `.out` | in-tree | `clm-ml-jax/src/output_files/JAX_outputs_05_2007_31days/` | — | this repo |
| `zenodo_obs` (D1) | CHATS 30-min multi-level tower obs | fetch (public) | `docs/MLC_experiment_plan/Data/Zenodo_17426258_CHATS_30-min/` **or** `$LEGO_LAND_EVAL_DATA/Zenodo_17426258_CHATS_30-min/` | 4.0 MB | [doi:10.5281/zenodo.17426258](https://doi.org/10.5281/zenodo.17426258) (Bonan/Burns/Patton 2026, AFM 378:110960) |
| D2 | US-MMS FLUXNET FLUXMET HR 1999–2023 v1.3 | fetch (account) | `docs/MLC_experiment_plan/Data/AMF_US-MMS_FLUXNET_1999-2023_v1.3_r1/` | 361 MB | [AmeriFlux](https://ameriflux.lbl.gov/) BASE (US-MMS) |
| D4 | FI-Hyy FLUXNET FLUXMET HH 1997–2024 v1.3 | fetch (account) | `docs/MLC_experiment_plan/Data/ICOS_FI-Hyy_FLUXNET_1997-2024_v1.3_r1/` | 828 MB | [ICOS](https://www.icos-cp.eu/) / FLUXNET (FI-Hyy) |
| D4 | US-Ton FLUXNET FLUXMET HH 2001–2025 v1.3 | fetch (account) | `docs/MLC_experiment_plan/Data/AMF_US-Ton_FLUXNET_2001-2025_v1.3_r1/` | 725 MB | AmeriFlux BASE (US-Ton) |
| D3/D5 | MODIS MCD15A3H LAI/FPAR (+ MCD43A3 albedo) | fetch (account) | `docs/MLC_experiment_plan/Data/MODIS/<site>/` | ~0.3 MB | [AppEEARS](https://appeears.earthdatacloud.nasa.gov/) |
| processed | FLUXNET slim subsets (target windows) | derived | `data/fluxnet/<site>/*_slim.csv` | 105 MB | `scripts/data/filter_fluxnet_to_target_years.py` |
| `fluxnet_obs` | Per-timestep obs targets | derived | `results/<site>/obs_targets.csv` | small | written by `run_fluxnet_offline.py` |

## Obtaining each class

**in-tree** — already present after clone. `fetch_land_eval_data.py --check`
confirms.

**Zenodo (public)** — `fetch_land_eval_data.py --fetch zenodo` downloads
record 17426258 to the location above. No account needed.

**FLUXNET (AmeriFlux / ICOS)** — account-gated; download the FLUXNET/FLUXMET
product for each site code and unzip to the path in the table, then run
`scripts/data/filter_fluxnet_to_target_years.py` to produce the slim CSVs the
loaders read. The pilot windows are US-MMS 2011-10→2013-12 (HR), FI-Hyy
2012-07→2013-12 (HH), US-Ton 2011-07→2012-12 (HH).

**MODIS (AppEEARS)** — request MCD15A3H LAI/FPAR (and MCD43A3 albedo with the
`Albedo_BSA_vis/nir`, `Albedo_WSA_vis/nir` bands selected) point samples for
each site, QC-masked, 2011–2013.

**Shared bundle** — if a collaborator has the staged data on shared storage,
`export LEGO_LAND_EVAL_DATA=/path/to/bundle` and `references.py` finds each
reference under its `bundle_relpath` before failing.

## Provenance notes

- CHATS obs use `1e36` for missing; FLUXNET FULLSET uses `-9999`; loaders in
  `boundary_data/` convert both to NaN before any math.
- US-MMS FLUXMET is **hourly** (HR) — the CLM-ML sub-cycle budget adjusts
  (`num_ml_steps` at `dt=3600 s`), see `run_fluxnet_offline.py --num-ml-steps`.
- US-Ton mixes C3 oak + C4 grass; runs are C3-only (labelled), per the
  experiment plan scope-out.
- `docs/references/` (research PDFs) is never committed — cite by DOI here.
