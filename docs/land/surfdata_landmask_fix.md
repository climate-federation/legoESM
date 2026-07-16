# legoesm surfdata land-mask fix (c250617 → c260716)

## The bug

`legoesm_surfdata_c250617.nc` over-counted land area by **~27%** (187.6 vs 147.5 Mkm²),
inflating every LMIP global total (GPP PgC, ET km³) by ~1.27×. Root cause in
`land/surface_data/sources/clm5_surfdata.py::read_clm5_cover_veg`:

```python
f_land = natveg + crop           # PCT_NATVEG + PCT_CROP
```

CLM landunit percentages (`PCT_NATVEG/CROP/LAKE/GLACIER`) are **percent-of-LAND** — they
sum to 100 of each cell's land part and are populated *even over ocean* (mksurfdata fills
a land template everywhere; `LANDFRAC_PFT` gates which cells are active). Used un-gated as
percent-of-gridcell, land smeared into coastal/ocean cells (1862 spurious cells,
20.6 Mkm²). Verified against the CLM file: landunit sum ≡ 100 everywhere; coastal cells
average `LANDFRAC 0.19` but `NATVEG+CROP 82.6`.

## The fix

Gate the landunit percentages by `LANDFRAC_PFT` [0,1] → percent-of-gridcell:
`f_land = (natveg+crop)·landfrac`, likewise `f_lake`/`f_glacier`/`pft_frac` (preserving
`pft_frac.sum(0) == f_land`). Monthly LAI/SAI/height are physical per-PFT values → **not**
gated. `reconstruct_clm5_pft_frac` is unchanged (the AMIP LAI loader
`clm_surface_map.py` wants land-mean LAI and normalizes PFT weights, so its un-gated use
stays correct).

**Mechanical guard:** `assert_cover_within_landfrac` (gated cover ≤ `100·landfrac`) runs on
every `read_clm5_cover_veg` call — the only stage where `LANDFRAC_PFT` is available.
Non-vacuous self-test + ocean/coastal gating tests in
`tests/land/surface_data/test_clm5_surfdata.py`.

**Robustness across paths:** the LUH2 / anthropogenic *transient* builders build on top of
the `build_v1_surfdata` base file (`_load_base_static`), inheriting the gated
`pft_frac`/`f_land`. So fixing the CLM read + regenerating the base file corrects all
downstream surfdata. Every built file now stamps a self-documenting `source` attr
containing `"LANDFRAC_PFT-gated"` and the CLM source filename — a file lacking that string
predates the fix.

## Regenerating the file

`c260716` differs from `c250617` **only** in the land-mask gating (verified bit-identical
PFT composition / soil / LAI on 10,069 solid-land cells → same CLM source, clean diff):

```bash
python scripts/data/build_legoesm_surfdata.py \
    --skip-hwsd \
    --intermediate data/legoesm_surfdata_soil_0p25.nc \
    --clm-surfdata <CLM surfdata_0.9x1.25_hist_2000_16pfts_c240908.nc> \
    --out data/legoesm_surfdata_c260716.nc
```

Validation: land+lake+glacier area **187.6 → 146.9 Mkm²** (CLM `LANDFRAC_PFT` truth 147.5;
0.6 residual = dropped wetland/urban), **spurious ocean-land cells 1862 → 0**.

## Hosting

The harmonized surfdata is distributed via Zenodo and fetched by
`scripts/data/download_lmip_data.sh`. To publish the fix:

1. Upload `legoesm_surfdata_c260716.nc` as a **new version** of the existing Zenodo
   concept-record (21087964) — the concept DOI keeps resolving to the latest.
2. Paste the versioned file URL into `download_lmip_data.sh` (`SURFDATA_URL` default) or
   distribute it via `$LEGOESM_SURFDATA_URL` / `--surfdata-url`.
3. Bump `SURFDATA_NAME` (already `c260716`) so runs stage the corrected file.

Runs must point `--surfdata` at the c260716 file for corrected totals. The remaining
GPP-vs-obs gap (~182 vs FLUXCOM ~120 PgC) is the separate #730 uncalibrated
over-productivity, not a mask artifact.
