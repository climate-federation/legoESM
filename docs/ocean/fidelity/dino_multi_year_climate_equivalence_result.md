# Result: 20-year DINO climate equivalence

Date: 2026-08-31. Session:
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Producer:
`ddd3a8476afd87da4afa5747eb7e5057490b893c`. This result applies the
statistics and bands frozen in `PREREG_multi_year_climate_equivalence.md` to
the `nemo_dino_kamm_mlf` twin bridge. It is a climate-distribution verdict,
not state tracking or an equilibrium claim. Standalone and cross-recipe
transfer remain out of scope.

## Rule-1e schema reconciliation

Block 6 originally asked each legoESM arm for a nonexistent aggregate
`snap_days` array. The producer actually writes the five per-date keys
`T3d_dayN`, `S3d_dayN`, `eta3d_dayN`, `u3d_dayN`, and `v3d_dayN` from
`kamm_twin_90d.py:2104,2212-2217`'s `_stored_days` loop. The repaired admission derives the
dates only by matching those suffixes. It requires identical date sets for
all five fields within an arm, identical sets across all six arms, and exact
equality to both `reduced_days` and the preregistered 75-date cadence. A
missing field or date is fatal.

All six existing arms passed: each has 419 keys, 75 occurrences of every
snapshot field, dates 360--7200, and identical field/key schemas. No model
member was rerun. The unchanged NEMO inputs and all twelve existing members
were then reduced offline.

## Frozen-band verdict

The disposition is **`DISTINGUISHABLE_AT_20Y`**. The table reports the
decisive resolved scalar for each family (or the largest unresolved scalar),
where `floor=sqrt(L_std^2+N_std^2)` and the frozen scalar bands are CONFIRM
when `R_hi<=2`, REFUTE when `R_lo>2`, and UNRESOLVED otherwise.

| family | verdict | decisive scalar | gap | floor | R | bootstrap R 95% CI | simultaneous R_hi |
|---|---|---|---:|---:|---:|---:|---:|
| ACC series | UNRESOLVED | `acc.full.month09` | -4.511147e-1 Sv | 4.997803e-1 Sv | 0.9026 | [0.1643, 3.0998] | 4.1250 |
| basin/row transports | REFUTE | `row.190.mean` | -4.711734e-3 Sv | 5.478857e-5 Sv | 85.9985 | [71.8750, 162.2120] | inf |
| MLD seasonal cycle | REFUTE | `mld.north of band.month05` | -3.164195e-2 m | 1.057164e-2 m | 2.9931 | [2.4400, 5.1834] | 6.1342 |
| T/S water-mass census | REFUTE | `north of band.abyss_ge1400m.S_mean.mean_y16_y20` | -1.092044e-5 | 1.046256e-7 | 104.3764 | [90.4394, 206.7472] | inf |
| upper/deep density contrasts | UNRESOLVED | `density.deep.month04` | 7.112532e-4 kg m-3 | 5.305986e-4 kg m-3 | 1.3405 | [0.5521, 5.3660] | 6.1655 |
| variability | REFUTE | `census.north of band.abyss_ge1400m.S_mean.deseasonalized_std` | -7.718259e-7 | 8.001027e-8 | 9.6466 | [8.0836, 17.7710] | 28.6396 |

The infinite simultaneous bounds in two already-refuted families come from
separately labeled zero-floor/quantized registered scalars. Those rows remain
`UNRESOLVED_QUANTIZED`; they neither create nor strengthen a REFUTE. The
decisive rows above have finite nonzero 20-year ensemble floors and their
lower bootstrap bounds exceed 2.

Endpoint diagnostics are not verdict inputs. They demonstrate the registered
floor-growth lesson: the y20/y1 floor ratios are 31.964 for full ACC, 36.599
for channel ACC, 49.966/48.310/13.656 for the south/channel/north basin
transports, and 2378.48/590.00 for the upper/deep density reductions. This
verdict is therefore horizon-specific.

All fail-closed plants fired: CONFIRM, REFUTE, UNRESOLVED, quantized collapse,
family maximum, wet U-row transport, wet T/S census cell, wet MLD column, and
monthly-cycle rotation. The from-rest y20 and y40 archive controls reproduced
142.80981676941545 Sv and 176.12412451228047 Sv exactly; they were not used in
any floor or verdict.

## Bound artifacts

- Raw admission manifest:
  `/data/abyssal/dbalwada/dino-climate-equivalence-20y-01a04e34/manifests/raw_capstone_manifest.json`,
  SHA-256 `e32a8eb1b5f7f250fa53dc6fded7d6154b6e7532ca4d4df0b179e2c9984b3809`.
- Frozen score JSON:
  `/data/abyssal/dbalwada/dino-climate-equivalence-20y-01a04e34/dino_multi_year_climate_equivalence_score.json`,
  SHA-256 `7ad72da798f754ff011c08af6f16cf44fdf4aa5aa7a682d458c9335c93c64bc0`.
- Compact member-statistic artifact:
  `/data/abyssal/dbalwada/dino-climate-equivalence-20y-01a04e34/dino_multi_year_climate_equivalence_reduced.npz`,
  SHA-256 `1a4bb42da37ab5f5dc12ab2f89e4b73deef01d6b80569195bbdef680516a8ca8`.
- Scorer log:
  `/data/abyssal/dbalwada/dino-climate-equivalence-20y-01a04e34/logs/dino_multi_year_climate_equivalence_score.log`,
  SHA-256 `93bd0de3b2957151d1e7289b727c6a96dfb55b1392fd1cbe782d7ff31e934a80`.
