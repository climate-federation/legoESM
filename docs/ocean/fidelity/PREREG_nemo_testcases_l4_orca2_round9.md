# NEMO testcase Lane 4 — ORCA2 card round 9 preregistration

Date: 2026-09-23

Parent: `3511d6c8ae4350f91a8a145cdcfbce5427b8e374`

Status: **PREREGISTERED BEFORE ANY ROUND-9 MEASUREMENT.**

Scope is round 8's OPEN item 1 on the ocean-only `orca2_vector_een_c2` card:
the lateral momentum viscosity COEFFICIENT, where the production arm now
refuses.  The six-entry sea-ice registry is frozen and stays out of scope.
Decision 52's labels are binding: every number is either **given NEMO's
entry** or **independent**, never mixed in one table without its label.

## What the record fixes, and what nothing in this round may choose

Read from the record's own resolved configuration, not from a deck comment.
The record is the pinned ORCA1-ice reference run
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`.

| resolved setting | value | where it is printed |
|---|---|---|
| lateral viscosity on | `ln_dynldf_OFF = F` | run `ocean.output:1175` |
| operator family | div-rot, `nn_dynldf_typ = 0` | run `ocean.output:1176,1193` |
| order | laplacian, `ln_dynldf_lap = T`, `ln_dynldf_blp = F` | run `ocean.output:1177-1178` |
| direction | iso-level, `ln_dynldf_lev = T` | run `ocean.output:1180,1195` |
| coefficient variation | `nn_ahm_ijk_t = -30` | run `ocean.output:1184` |
| coefficient source | `ahmt_3d` and `ahmf_3d` read from `eddy_viscosity_3D.nc` | run `ocean.output:1197,1200-1201` |
| the file itself | `eddy_viscosity_3D.nc` -> `/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/eddy_viscosity_3D.nc`, sha256 `fc142ce094a0f68d255ff79b04f8fe5d6634b33b0bc1c07b61870cede39499cb` | the run directory's symlink, and the run's own `input_files.sha256` line 12 |
| north-fold type | present, **T pivot** | run `ocean.output:207-208` |
| inner global domain | 180 by 148, halo width two | run `ocean.output:38,44,68-69` |
| rank layout | two ranks split in longitude only | run `ocean.output:196-202` |

No selector default, tunable, threshold, resolution, timestep, carried state
or data source moves in this round.  The GYRE card keeps the formula it has
today; the new source is selected only by the ORCA2 card, mirroring the
namelist value the record resolves.

## The statement this round transcribes

With `nn_ahm_ijk_t = -30` NEMO does not compute a coefficient at all.  It
opens `eddy_viscosity_3D.nc` and reads the whole three-dimensional field at T
points and at F points
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/ldfdyn.f90:348-353`), each
read carrying its own grid-point nature and north-fold sign (`'T'` and `'F'`,
both `+1`).  The read itself completes the field with the ordinary lateral
boundary exchange for that nature
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iom.f90:958-975`), whose
T-pivot rules are
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/lbcnfd.f90:583-639` for a
T-point field and `:722-746` for an F-point field.  Because the resolved
operator is the laplacian, the field is then multiplied by the corresponding
mask over levels one to `jpkm1` and left alone at the last level — no square
root, which is the bilaplacian arm
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/ldfdyn.f90:388-396`).

`rn_Uv` and `rn_Lv` are read and printed, and the `-30` arm never consults
them: `zah0` is computed at `ldfdyn.f90:313` and is not referenced inside the
`CASE( -30 )` block.  legoESM's `A_h` therefore acts only as the on/off switch
under this source, exactly as `rn_Uv` does in NEMO.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R9-P1 | Under `nn_ahm_ijk_t = -30` the compiled routine reads `ahmt_3d`/`ahmf_3d` from the file, exchanges each with its own grid-point nature at sign +1, then multiplies levels 1..`jpkm1` by `tmask`/`fmask` with NO square root. | The compiled `CASE( -30 )` block, the `iom_get` call arguments, the `lbc_lnk` call in the read path, and the laplacian arm of the post-select masking all read as stated. | Any of those four statements reads otherwise. |
| R9-P2 | The recorded `ahmt`/`ahmf` streams in `output.init` equal the file's values cast to double and multiplied by the card's own `tmask`/`fmask`, on every owned cell of BOTH ranks over the card's thirty levels: 0 unequal doubles. | Zero unequal over 148x180x30 for each of the two fields. | Any unequal bit. |
| R9-P3 | The north-fold exchange changes nothing on the owned domain for either field, because the shipped input file is already fold-consistent. | Applying the compiled T-point and F-point T-pivot rules to the file's owned rows leaves every owned cell unchanged. | Any owned cell moves, in which case the exchange is part of the transcription and is transcribed, cited, and gated. |
| R9-P4 | With the file source selected, the card supplies coefficients bit-identical to the record on all cells of both ranks, the F-point field carried on legoESM's vertex layout by `vertex[j,i] = F[j-1,(i-1) mod n_lon]`. | Zero unequal doubles for both fields against the record, at every level, on both ranks. | Any unequal bit, or an index map that needs a different shift. |
| R9-P5 | The change is inert for GYRE: GYRE keeps the formula source, so nothing it executes moves. | Base and tip GYRE ten-step ladders give zero differing rows and array-equal residual arrays, and the thirty-day member snapshots are byte-identical. | Any differing row, unequal residual array, or differing snapshot digest. |
| R9-P6 | With the coefficient supplied, the ORCA2 ladder advances past this stop and reaches a first non-bit ARITHMETIC statement at kt=1, named with a cited compiled owner and a registered kt=10 magnitude. | The ladder returns a first non-bit row inside a Runge-Kutta stage, with a kt=10 magnitude for the same field. | The ladder stops on another unbuilt statement, in which case that statement is named and cited and the kt=10 magnitude stays UNMEASURED. |

Failed predictions stay in the receipt as **REFUTED** and are never quietly
dropped.

## Controls and stop rules

- Perturbing one representable value of the loaded coefficient must make the
  gate refuse and name the offending column, row and level.
- Dropping the mask must make the gate refuse, so the masking is not vacuous.
- Dropping the F-point index shift must make the gate refuse, so a
  self-consistent but wrongly staggered map cannot pass.
- Reading the file at its stored single precision instead of casting to double
  must make the gate refuse, or the comparison is not measuring precision.
- A rigid two-line shift of a round-9 compiled citation must fail the citation
  gate.
- GYRE's trajectory is proven unchanged before anything lands, at the round's
  base tip and at its final tip, with the evaluation protocol byte-identical.
- No stabiliser, clip, damp or limiter NEMO lacks may be added.
- The card's own level 31 does not exist; NEMO's level 31 is reported as
  coverage, not asserted as a legoESM cell.

## Labels

The coefficient is read from an INPUT file the card already owns, not from a
recorded run state, so every coefficient number in this round is
**independent**.  Decision 52's sea-surface-height bridge is unchanged and
remains the only explicit entry replacement; the ladder's trajectory rows keep
their existing `INDEPENDENT_WITH_DECISION52_SSH` label.
