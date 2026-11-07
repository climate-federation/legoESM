# NEMO testcase Lane 4 — ORCA2 Phase-2g O1/V2 handoff

Date: 2026-09-06

Parent: `d44ca7abaaf4461ffb57014bbff31d89216c9c26`

Preregistration: `f93e43a013bb6c7a586215b3075c4a9f1308f6b6`

Result: **STOP for two user-shell NEMO runs and hand O1-B to
`LANE3B_OWNER`.**  The 91 reproducible records are pinned as the icebergs-off
VARIANT oracle V2.  The ORCA2-owned O1 `fld_read` boundary is exact for all
nine CORE fields (`0 / 13,320` each).  Operand substitution puts the first
over-bar boundary at the shared NCAR bulk leaf, so the ownership rule forbids a
Lane-4 arithmetic change.  The remaining two source-unowned O1 fields have
been canonicalized WRITE-only; two replacement run directories are ready but
were not executed by the agent.

## 1. VARIANT oracle V2

The completed user-shell twins are:

- `variant_icebergs_off_phase2f_canonical_a_10step_np2`;
- `variant_icebergs_off_phase2f_canonical_b_10step_np2`.

Both independently pass the Phase-2b VARIANT gate and ordinary-output identity
against the uninstrumented icebergs-off control.  The gate derives four exact
restart shards and eight exact history payloads.  Each run passes the ten
legacy record plants, six post-`sbc` surface-input plants, and the variant
plant; every planted invocation exits nonzero.  The ocean restart hashes are
`28fbf31286d90f31e6f72083b942a322a4d2cbfb18db6bf4983d76bae82c7280`
and `4f7873be26a96f9694470e5e7b67b93e7956180045effe948e2d3080eca4a326`;
the ice restart hashes are
`4015292b4663e27a8e44109c1c001603d1d82519339d8cc1285c6d4f268d8be2`
and `f6bea2738a8e10286653d09ee8bf1bdd18a91111c9b22c3260200975f847a229`.

The twins have 91 / 91 inherited records raw-byte exact: the frozen Phase-1
90-stream inventory plus `oracle_ocean_surface_input_kt00000001.bin`.  Run A
supplies the V2 bytes.  Its 91-entry manifest is
`phase2g/variant_v2_all_91_records.sha256`, SHA-256
`9455cccb19ccbecd06f457fb6db71f8fdebc8bfebd2dc4d4909e5cb25de46668`;
the nested 90-entry manifest has SHA-256
`13b34464996a1967ce8f5c93cf59ff62ea77cbcea4de0880c8b19f8164b1d26c`.
The earlier accepted variant records are preserved and superseded by V2.

The provisional O1 files differ in raw bytes only in source-unowned storage.
Their digests are
`58394a0eb1852d0ba078250e19f1d7a7d14d1d99a267b99f16f077044850892e`
(A) and
`fcabd197fb40a5ad7b7a9c3cadfcfba1a4e8b1f2f8844a12ff6757ab0b2a7a9d`
(B).  The defined-field gate compares 359,640 f64 values exactly and excludes
exactly 26,640 f64 values.  It passes 27 fieldwise one-ULP O1 controls, two O1
schema controls, and seven inherited defined-cell controls, all through their
production validators.  This O1 is valid as a defined-field operand but is
labelled `VARIANT_V2_PROVISIONAL_O1_DEFINED_FIELDS`; the replacement twins
below must supersede it before O1 is promoted to the raw-byte oracle.

## 2. Complete O1 source-defined inventory

The record contains only rank-0 `A2D(0)`, 90 by 148, and no stored halo band.
Its headers are `(1,1,0,90,148,9,0,64)` and
`(1,1,1,90,148,20,0,64)`, with no trailing bytes.  Source inspection and the
twin diff prove the following inventory:

| frame | fields | stored status |
|---|---|---|
| 0 | all nine `sf(...)%fnow(:,:,1)` inputs | source-defined in all 13,320 cells |
| 1 | `pcd_du` / schema name `cd_du` | wholly unowned: assigned only under inactive `ln_abl` |
| 1 | `qlwn` | wholly unowned: optional MFS result with resolved `ln_MFS=.false.` |
| 1 | other eighteen outputs | source-defined in all 13,320 cells |

The only A/B differences are four values at `(i=0..3,j=0)` in each of
`cd_du` and `qlwn`.  Those locations do not define a narrower exclusion: both
complete fields are semantically unowned.  The selected NCAR arm is at
preprocessed `sbcblk.f90:908-936`; the sole `pcd_du` assignment is behind
inactive `ln_abl` at `:896-906`.  NEMO `fld_read` establishes spatial mapping
and boundary linking at `fldread.F90:370-399`, wind rotation at `:705-759`,
and temporal selection/interpolation at `:181-227`.

## 3. O1-M — shared NEMO `fld_read` map

The new shared implementation is
`packages/ocean/legoesm/ocean/forcing/nemo_fld_read.py`.  It is selected
explicitly by the ORCA2 card as `surface_input_operator="nemo_fld_read"` and
is intended for the future ORCA1 NEMO-identity card as well; there is no
ORCA2 arithmetic fork.  It mirrors NEMO's complete-pass bicubic derivative
ordering, deck weights, record clocks, unit factors, and paired wind rotation.
Static grid trigonometry follows the campaign scalar-libm policy; mapped
arrays run under production JIT, CPU, and fp64.

Against O1 frame 0, every source-defined cell is exact:

| mapped field | result |
|---|---:|
| U wind `wndi` | 0 / 13,320 |
| V wind `wndj` | 0 / 13,320 |
| air temperature `tair` | 0 / 13,320 |
| specific humidity `humi` | 0 / 13,320 |
| shortwave `qsr_down` | 0 / 13,320 |
| longwave `qlw_down` | 0 / 13,320 |
| precipitation `precip_raw` | 0 / 13,320 |
| snow `snow_raw` | 0 / 13,320 |
| sea-level pressure `slp` | 0 / 13,320 |

All nine cell-score plants exit nonzero.  This is **CONFIRMED AT_BAR** and
removes only `nemo_fld_read_forcing` from the card's unmeasured registry.  The
entry gate remains exact: T, S, u, v are each 0 / 399,600 and ssh is
0 / 13,320.  Its eleven structural/entry plants all exit 1.

## 4. O1-B — first over-bar boundary

The gate substitutes the oracle O1-M fields into the one shared
`bulk_flux_omip.air_sea_fluxes(algo="ncar")` implementation.  Thus mapping
error is excluded and SI3 exchange remains `ORACLE_SUPPLIED`.  The rank-0 wet
T-cell domain has `n=8,794`.

| bulk output | unequal / n | maximum absolute difference |
|---|---:|---:|
| `theta_air` | 117 / 8,794 | 6.641000762849103e-2 |
| `ssq` | 3,964 / 8,794 | 4.5102810375396984e-17 |
| `utau` | 8,794 / 8,794 | 1.1616490101035923e-3 |
| `vtau` | 8,794 / 8,794 | 1.2697428825805923e-3 |
| sensible heat | 8,794 / 8,794 | 1.8866460687673907 |
| latent heat | 8,794 / 8,794 | 11.99689540648491 |
| evaporation | 8,794 / 8,794 | 4.785976045696383e-6 |

All seven cell-score plants exit nonzero.  This is **CONFIRMED DEBT** at
`O1-B/NCAR`, owner `LANE3B_OWNER`, and is the first over-bar boundary in NEMO
execution order.  It is handed to Lane 3b/GYRE unchanged.  Lane 4 made no
shared-bulk repair and did not enter RGB QSR, EOS/HPG, external mode,
transports, FCT, BBL, or TKE/EVD/IWM.  SI3 remains
`UNMEASURED_PENDING_ICE_MERGE_ORACLE_SUPPLIED_EXCHANGE`.

## 5. O1 canonical writer and scalar-math rebuild

The config-copy `MY_SRC/sbcblk.F90` now creates a local `A2D(0)` array, fills
it with `0._wp`, and writes it in the `pcd_du` and `qlwn` schema positions.
It does not assign or consume a model field, and does not change headers,
counts, execution options, or arithmetic.  Its SHA-256 is
`83c2bd19292c39205c310974931638b1dd9cd23e623cc983364c0cbf3b3cf69b`.
The canonical gate requires both whole fields to be zero and contains a
binding nonzero plant.

The copied configuration was rebuilt with `makenemo -n ORCA2_OMIP_L4 -m
conda-scalarmath`.  The successful log retains `-fno-tree-vectorize`, and
`nm -D` reports zero `_ZGV*` symbols.

| artifact | bytes | SHA-256 |
|---|---:|---|
| `build/nemo_ORCA2_OMIP_L4_phase2g_o1canon.exe` | 55,129,776 | `a34c795bb273b338f5b9dd271843aa46f123d4af3dea7f8de8c028b515ddf31a` |
| `build/build_ORCA2_OMIP_L4_phase2g_o1canon.log` | 10,866 | `c0d73f008e64653e49d3c26536bab5af880f6fa07845dd853857349aeccf5a74` |
| `build/phase2g_o1canon_MY_SRC.sha256` | 2,270 | `ac8b80f6cdb2eda8c9cf5a5c2e102ce1d9785dc2f3ac1ffdc8dedb2f1dad04b1` |
| `build/phase2g_o1canon_ZGV.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| retained failed build log | 3,391 | `a07079631b4808d02c036b9c4a519c7228870e6b594e537f923ce875da3a9d10` |

The first build attempt failed before producing a binary because the new local
declaration was initially placed in the adjacent input routine.  The placement
was corrected; the failed log is preserved and explicitly retracted.

## 6. Replacement twin handoff

Both directories use the one binary above, the unchanged icebergs-off VARIANT
deck, two MPI ranks with resolved `jpni=2`, `jpnj=1`, and Bash timing.  Each
contains 19 copied deck files, 40 absolute immutable input symlinks, one
absolute binary symlink, two manifests, and its self-contained launcher.  The
launcher checks the executable and every deck/input digest before
`mpirun -np 2 --oversubscribe`; OMP, OpenBLAS, and MKL thread counts are one.

| arm | launcher SHA-256 | 63-entry prepared-manifest SHA-256 |
|---|---|---|
| A | `29a22e637398a1a4bab5d6927eb3f318cfa41da240c63edc1297360c57d15a95` | `828b6ad4802197576068e0b01332aca543ffc2984d468b16b97778e3c6404c81` |
| B | `7d900db5693b030837953a681125cce114d1448b47f0e017dacd8ef69af89ea5` | `9deb253ddf3a3a2de9d257105c543146c81e6ebfceb29d70ede39985ec63de75` |

Run these one at a time, unchanged:

```text
/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2g_o1canon_a_10step_np2/run.sh
/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2g_o1canon_b_10step_np2/run.sh
```

On resume, schema-walk both 92-record inventories, require 92 / 92 raw-byte
identity, independently run the ordinary-output and planted controls for each,
and require the O1 whole-field zero contract.  Only then promote the
replacement O1 and retire its provisional defined-field label.  No additional
NEMO record is currently requested: the numerical ladder is already stopped
at the shared O1-B debt.

## 7. Evidence digests

Root: `/data/abyssal/dbalwada/nemo-testcases-l4/phase2g`

| artifact | bytes | SHA-256 |
|---|---:|---|
| `canonical_a_phase2b_gate.json` | 116,574 | `fe07a0dc8e6d3382c282fc647cecaa9cac46448f68603e98f7fe592b90d39d62` |
| `canonical_b_phase2b_gate.json` | 116,574 | `c4420e784c780b7924ed79a375bcd56dd0e5d7d1ecd436022ec8fd23896c0d6c` |
| `o1_defined_twin_gate.json` | 197,768 | `a2e841d22ced403946826e4a8b0b1bcbb8beb32d441943567148a521cc640718` |
| `o1_numerical_gate.json` | 4,244 | `1055cccaebc49b8da1893ba74fa8981c33a56f8affac7c3e26a5af87dfff736d` |
| `card_entry_gate.json` | 1,503 | `893f38dfa12c0ee7054e4ebfb2a23f23b16fbfe1f96c42f17137112210228d12` |

Each corresponding retained stdout has the same digest as its JSON.

## 8. ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| pin 91 reproducible records as VARIANT oracle V2 | ASKED | 91 / 91 raw exact; run A hash manifest is canonical |
| canonicalize the remaining O1 unowned slots | ASKED | complete `pcd_du` and `qlwn` fields zeroed in writer-only storage |
| prepare replacement twin runs | ASKED | two hash-guarded directories prepared; agent did not execute MPI |
| start O1 mapping and NCAR boundaries | ASKED | O1-M exact; O1-B first debt, with operand substitution and owner label |
| extend the ORCA2-specific input path | ASKED | one shared NEMO `fld_read` implementation, selected explicitly on the card |
| fix shared NCAR bulk debt | UNASKED and forbidden | registered to `LANE3B_OWNER`; no arithmetic changed here |
| execute MPI | UNASKED and prohibited | user shell must run both launchers |
| enter SI3 | UNASKED pending merge | oracle-supplied exchange remains the registered input |
| delete or overwrite an earlier artifact | UNASKED and forbidden | none deleted; prior arms and failed build log are retained and labelled |
| initial declaration in adjacent writer routine | UNASKED implementation error | build failed before a binary; corrected, with failed log retained |

All numerical claims above are **CONFIRMED** by the committed gates.  No
unmeasured downstream boundary is described as plausible fidelity.
