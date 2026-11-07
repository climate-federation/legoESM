# NEMO testcase Lane 4 — ORCA2 card round 9 transcription receipt

Date: 2026-09-23

Starting tip: `3511d6c8ae4350f91a8a145cdcfbce5427b8e374`

Preregistration: `0a98fc754`

Status: **HELD.**  Round 8's OPEN item 1 is discharged: the lateral momentum
viscosity coefficient the ORCA2 run uses is no longer computed from a formula
the run never selects — it is READ, exactly where and exactly as NEMO reads it,
and gated bit-exact against NEMO's own recorded coefficient on every owned cell
of both ranks.  Round 8's OPEN item 3 is also discharged: the operator ran the
per-rank acquisition, and the tripolar fold is now certified in BOTH directions.
The ladder advances past this statement and stops on the NEXT unbuilt one, so
the kt=10 magnitude stays **UNMEASURED** — not zero, not extrapolated.

No configuration, selector default, tunable, threshold, cadence, resolution,
timestep, carried state, data source, NEMO source or sea-ice registry entry
changed.  Sea ice remains out of scope and the six-entry `unmeasured_features`
tuple is unchanged.

Independent review **not run (codex paused)**; a separate fresh adversarial
reviewer was run in its place (section 8a).

## 1. What NEMO actually does, and what nothing here may choose

The resolved run fixes every relevant setting and each was read from the
record's own output, not from a deck comment.  The record is the pinned
ORCA1-ice reference run
`orca1ice_surface_entry_every_step_a_np2`.

| resolved setting | value | where it is printed |
|---|---|---|
| lateral viscosity on | `ln_dynldf_OFF = F` | `ocean.output:1175` |
| operator family | div-rot, `nn_dynldf_typ = 0` | `ocean.output:1176,1193` |
| order | laplacian, `ln_dynldf_lap = T`, `ln_dynldf_blp = F` | `ocean.output:1177-1178` |
| direction | iso-level, `ln_dynldf_lev = T` | `ocean.output:1180,1195` |
| coefficient variation | `nn_ahm_ijk_t = -30` | `ocean.output:1184` |
| coefficient source | `ahmt_3d` and `ahmf_3d` read from a file | `ocean.output:1197,1200-1201` |
| north-fold type | present, **T pivot** | `ocean.output:207-208` |
| rank layout | two ranks split in longitude only | `ocean.output:196-202` |

With `nn_ahm_ijk_t = -30` NEMO computes no coefficient at all.  Four statements,
in this order:

| statement | compiled owner |
|---|---|
| open the file and read the whole three-dimensional field at T points and at F points, each with its own grid-point nature and north-fold sign (`'T'` and `'F'`, both `+1`) | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/ldfdyn.f90:348-353` |
| the read path completes each field with the ordinary lateral boundary exchange for that nature | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iom.f90:958-975` |
| the laplacian arm then multiplies levels one to `jpkm1` by `tmask`/`fmask` — **no square root**, which is the bilaplacian arm | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/ldfdyn.f90:388-393` |
| the T-pivot exchange rules the read applies, per grid-point nature | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/lbcnfd.f90:584-638` (T) and `:722-746` (F) |

`rn_Uv` and `rn_Lv` are read and printed and this arm never consults them: the
mixing coefficient they build (`ldfdyn.f90:314`) is not referenced inside the
`CASE( -30 )` block.  legoESM's `A_h` therefore keeps only its on/off role under
this source, exactly as `rn_Uv` does in NEMO.

The file is `eddy_viscosity_3D.nc`, which the run directory symlinks to
`/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/eddy_viscosity_3D.nc`,
sha256 `fc142ce094a0f68d255ff79b04f8fe5d6634b33b0bc1c07b61870cede39499cb` —
the same digest the run's own `input_files.sha256` records.  It stores both
fields at single precision; every stored value is exact in double, so the cast
the read performs loses nothing and a precision control cannot fire here.

## 2. What landed, and what it deliberately does not add

One selector, named after the namelist field it mirrors, whose default is the
formula the shared code already ran.  The GYRE card does not mention it and is
bit-identical; only the ORCA2 card selects the file source, and only the ORCA2
card carries the read field — on the vertical coordinate, beside the other NEMO
literal operands that card already reads from its deck.  The two div-curl
operators now accept a full three-dimensional coefficient as well as the
latitude vector; with a latitude vector they are unchanged.

**The north-fold exchange is NOT re-implemented.**  On the shipped input file it
is the identity over the owned domain for both natures, and that is a MEASURED,
GATED statement rather than an assumption: the gate refuses unless the file's
last owned T row is still its own mirrored left half and its last owned F row is
still the row below at the reversed longitude.  Writing an exchange that is
provably a no-op would have been code with no consumer; asserting the condition
instead fails closed if a future input file breaks it.

## 3. The measurement, and its label

Every number in this section is **independent**: the coefficient is read from an
INPUT file the card already owns and masked with the card's own masks, with no
operand loaded from a recorded run state.  Decision 52's sea-surface-height
bridge is unchanged and remains the only explicit entry replacement.

| quantity | scope | result |
|---|---|---|
| T-point coefficient, all cells, both ranks | 148 x 180 x 30 | **0 / 799,200 unequal** |
| F-point coefficient, all cells, both ranks | 148 x 180 x 30 | **0 / 799,200 unequal** |
| T-point coefficient, the tripolar fold row alone | 180 x 30 | **0 / 5,400 unequal** |
| F-point coefficient, the tripolar fold row alone | 180 x 30 | **0 / 5,400 unequal** |

NEMO's last level is never masked and legoESM has no such level; as coverage,
the record's level 31 is the raw file value for both fields (**0 / 26,640
unequal** each).

**The controls, each of which must leave the comparison unequal.**

| control | unequal | largest difference |
|---|---|---|
| the coefficient without its mask | 368,648 / 799,200 | 40000.0 |
| the F field read without its index shift | 84,325 / 799,200 | 80000.0 |
| the T field read as the F field | 133,601 / 799,200 | 80000.0 |

**The fold-consistency condition that licenses omitting the exchange**, measured
on the input file: the T rule's destination half already equals its mirrored
source half (**0 / 2,759**), and the F rule's last owned row already equals the
row below at the reversed longitude (**0 / 5,580**).

**The F index map, proven against code that predates this round.**  The claim is
`vertex[j, i] = NEMO_F[j-1, (i-1) mod n_lon]`.  Under that map legoESM's own
vertex mask equals NEMO's four-T-cell F-point product cell for cell away from
the fold row (**0 / 26,460 unequal**), while shifting the map by one column
either way leaves 1,180 cells unequal and shifting it by one row leaves 1,052.
The vertex row that has no NEMO source at all is the south wall; the card leaves
it zero and legoESM's vertex mask is zero there, so it is inert.

**Both gate plants fire**: perturbing one representable value of either
coefficient makes the gate refuse and name the cell (exit 2).

## 4. Round 8's other half of the fold row, now recorded

The operator ran round 8's per-rank acquisition
(`ORCA2_ROUND8_EEN_RANK_ACQUISITION_PASS`,
`phase3/orca2_rounds/round8/acquisition/een_per_rank_a_np2`).  The EEN operand
gate now joins the two ranked dumps into all 180 fold-row longitudes.  The new
record is a different build, so its rank-zero half was required to reproduce the
admitted record first: **0 unequal** for both operands.  Then, over the WHOLE
fold row:

| quantity | scope | result |
|---|---|---|
| frozen vertex thickness, whole owned block | 148 x 180 x 30 | **0 / 799,200 unequal** |
| frozen vertex thickness, the fold row | 180 x 30 | **0 / 5,400 unequal** |
| live vertex thickness, whole owned block | 148 x 180 x 30 | **0 / 799,200 unequal** |
| live vertex thickness, the fold row | 180 x 30 | **0 / 5,400 unequal** |

with all three of round 8's controls still firing on the full row: no fold
exchange 1,481 / 5,400 (largest 254.03568364297735 m), the T-origin mirror
487 / 5,400 (479.1374821040031 m), the fold row as its own source 1,555 / 5,400
(479.1374821040031 m).  **The tripolar fold is now certified in both
directions**, and round 8's one-direction caveat is retired.

## 5. How far the ladder now reaches, and what stops it

Past the viscosity coefficient, into the shortwave penetration, where it stops:

> `shortwave_penetration_tendency is the two-band Jerlov kernel but got
> scheme='nemo_qsr_rgb'`

The resolved run selects the three-band chlorophyll attenuation with the
Morel-Berthon vertical profile: `ln_qsr_rgb = .true.` with `nn_chldta = 1`
(`ocean.output:1207,1211`) makes `tra_qsr` dispatch `qsr_RGBc`
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traqsr.f90:213,258-468`).
The shared ocean physics pipeline has only the two-band kernel and carries
neither the chlorophyll field nor the live thickness the three-band kernel
reads.  The RGB kernel itself EXISTS in legoESM and is wired into the external
surface-forcing path; it is that pipeline, not the kernel, that has no channel
for it.

The ladder gate records this the way the previous stop was recorded — named,
cited, with no magnitude registered — and exits non-zero.  Its stop registry
shrank with the transcription: the refusal round 9 removed can no longer be
labelled, and a test asserts that.

That the coefficient branch now EXECUTES is not assumed.  At the round's base
the same step raised the viscosity refusal, which proves the viscosity is
evaluated BEFORE the shortwave pipeline; it no longer raises, and the step now
reaches the later routine.  Independently, perturbing one cell of the carried
coefficient moves the tendency (section 8).

## 6. Prediction ledger

| ID | verdict | evidence |
|---|---|---|
| R9-P1 | **CONFIRMED** | All four compiled statements read as predicted: the `CASE( -30 )` block reads both fields with natures `'T'`/`'F'` at sign +1, the read path calls the lateral exchange, and the laplacian arm masks levels 1..`jpkm1` with no square root. |
| R9-P2 | **CONFIRMED** | The recorded streams equal the file's values cast to double and masked with the card's own masks: 0 / 799,200 unequal for each field, both ranks (section 3). |
| R9-P3 | **CONFIRMED** | The exchange is the identity on the owned domain for both natures: 0 / 2,759 and 0 / 5,580 (section 3). |
| R9-P4 | **CONFIRMED** | The card's carried coefficient is bit-identical to the record everywhere, under the stated index map, which is itself proven against older code with three firing controls (section 3). |
| R9-P5 | **CONFIRMED** | GYRE base and tip agree: see section 7. |
| R9-P6 | **REFUTED / kt=10 UNMEASURED** | The first non-bit statement moved later again but is still not an arithmetic row: the ladder now stops at the shortwave penetration (section 5). |

R9-P6 is the honest outcome of a ladder that gained a statement and found the
next one missing.  No magnitude is registered for a row that was never computed.

One prereg detail was **corrected**: the mixing coefficient this arm never uses
is assigned at `ldfdyn.f90:314`, not `:313`, and the laplacian masking arm ends
at `:393`, not `:396`.  Both were read off the compiled file and fixed in the
code and here; no claim depended on either string.

## 7. GYRE, which shares this implementation

The changed files serve GYRE too, so GYRE was re-measured at the round's base
tip and at its tip with the eval protocol held byte-identical.

| check | result |
|---|---|
| ten-step trajectory, certified rows compared | 70 |
| rows whose status changed | 0 |
| violations | 0 |
| largest oracle-residual worsening | 0.0 ULP |
| first row over the bar | unchanged (u and v at kt=2) |
| per-cell residual arrays (210 of them) | every one array-equal |
| 30-day member, daily state snapshots | 30 files, 0 differing bytes |
| day-30 snapshot digest | `14a7e64b4512860eded79bb12a2120885b97400ecb4acb7e6a2bd3d84a245469`, the same as rounds 7 and 8 |

GYRE is inert here **by construction**, not merely by measurement: its card
never sets the new selector, so its coefficient still comes from the formula and
the new branch is unreachable.  A unit test pins that default.

## 8. Implementation scope, tests and non-vacuity

222 inserted lines across six model files, most of them comment and docstring:
one config field, one vertical-coordinate pair of fields, a card-level builder,
four dispatch guards, and a broadcast relaxation in each of the two operators.
No parallel ladder, parser, launcher or second gate was created; the per-rank
fold certification EXTENDED round 8's gate rather than adding one.

New direct tests: seven on the selector (the file arm consumes the carried
field and moves when one cell of it is perturbed, refuses without it, refuses an
unknown value, refuses any other operator, refuses a no-slip side drag, refuses
the flux-form dissipation diagnostic, and the default stays on the formula), two
on the e3-weighted operator ORCA2 actually runs (a three-dimensional coefficient
equal to the latitude profile is bit-identical to it and a perturbed cell moves
the tendency; two-dimensional velocities with a three-dimensional coefficient
raise), eight on the builder and the coefficient gate's helpers (masking, the
level slice, the index map, its wrap column and its sourceless south row, the
fold-consistency refusal), and two on the per-rank record (the two halves join,
and a halo slot that is not empty is refused).

**Non-vacuity, measured.**  The new tests were run unchanged at the round's base
tip: the selector file reports **5 failed, 12 passed** — the five new ones fail,
the twelve pre-existing ones pass — and the round-9 gate's test module cannot
even be collected there, because the builder it imports does not exist.  The
base clone was left clean.

## 8a. Independent review

The mandated read-only independent review **was not run with codex, which the
user paused**.  A separate fresh adversarial reviewer with no part in the work
was run in its place, on the committed diff and the compiled NEMO source.

It **confirmed the index map independently of the builder**, by lighting single
u-faces, v-faces and dry cells and reading which vertices moved, and it restated
both compiled T-pivot fold rules in owned zero-based indices from the run's own
`ipi`, `ipj`, `ihls` and `Ni0glo` and matched them to the gate's two checks.  It
also re-ran the coefficient gate itself and reproduced its numbers.

It found six defects, ALL of them this round's own, and all are fixed:

| finding | what was wrong | fix |
|---|---|---|
| the vertex mask's comment claimed to be NEMO's fmask analogue at free slip | ORCA2 resolves `rn_shlat = 2` plus a per-strait override (run `ocean.output:339,342,345-346`), so NEMO's fmask carries the lateral boundary condition in its VALUE | comment corrected; the second masking is now named as a measured deviation and carried in OPEN |
| "A_h is only the on/off switch" | false: the no-slip side drag reads the scalar `A_h` outside this operator | that combination now raises, as does the flux-form dissipation diagnostic; neither is selected by any card |
| both operator docstrings declared a latitude-vector coefficient | stale after this round | both accepted shapes declared |
| the e3-weighted operator — the one ORCA2 runs — had no three-dimensional-coefficient test | the new selector tests exercised the non-e3 operator | two tests added, with a perturbation control |
| a dead duplicate control assignment in the gate | leftover | removed |
| the builder docstring cited `zah0` one line high | | corrected, together with two other line numbers (section 6) |

It raised four risks.  Two are now MEASURED and closed: the card's masks are
NEMO's over every cell, not only where the coefficient is non-zero (0 / 799,200
unequal for each), and the vertex row with no NEMO source is inert because
`compute_vertex_mask` pole-pads that row with zeros unconditionally, which the
gate asserts.  Two stay open: both fold rules are written for this run's halo
width rather than parameterised by it, and the downstream masking of OPEN item 3
is unmeasured against NEMO's own tendency.

It also noted that the builder's unit test restates the builder's own formula.
That is true, and it is a REGRESSION PIN, not the proof: the claim is carried by
the gate's index-map row, which the reviewer independently confirmed.

One prereg control is declared **NOT APPLICABLE** rather than quietly dropped:
"reading the file at its stored single precision must make the gate refuse"
cannot fire, because every value the file stores is exactly representable in
double, so the cast the read performs is lossless.  That is measured, not
assumed.

## 9. OPEN

1. **The next statement in execution order is the shortwave penetration.**  The
   resolved run dispatches the three-band chlorophyll attenuation
   (`traqsr.f90:213,258-468`); the shared physics pipeline has only the two-band
   kernel.  That is a shared-model change and will need the full GYRE identity
   proof again.
2. Only after that can the ladder produce a first non-bit ARITHMETIC statement
   and a kt=10 magnitude.  Neither is registered here.
3. **What the OPERATOR then does with this coefficient is a separate,
   UNMEASURED statement, and it is large.**  NEMO stores `ahmf` with `fmask`
   already folded in — values 0, 0.5, 1 and 2, the free-slip / strait /
   no-slip lateral boundary condition — and `dyn_ldf_lev` reads it exactly as
   stored.  The legoESM div-curl multiplies it AGAIN by its own per-level vertex
   mask, which is 0 or 1.  Measured: that second masking zeroes NEMO's non-zero
   coefficient in **48,287 of 799,200** F-point cells.  This is a statement
   about the MASK, downstream of the coefficient certified here; nothing in this
   round claims it is right.
4. **legoESM's vertex mask and NEMO's F-point product disagree on 4 of the 180
   fold-row longitudes** (measured, reported by the round-9 gate).  Same owner
   as item 3.
5. The independent sea-surface height still differs by up to 1.55 cm on 16,433
   of 26,640 surface cells, owned by the initial sea-ice category configuration,
   which is out of scope on this lane.
6. The recorded runoff tracer-source operands remain an explicit later boundary;
   this round did not reach them.
7. **The barotropic vertex thickness is still unmeasured** (round 8's OPEN item
   6, untouched here).
8. GitHub issue 1455 remains an operator-post action because no GitHub connector
   is installed in this environment.

## Choices

ASKED: none were needed.  Every setting this round touched is fixed by the
record's resolved configuration and was read from it (section 1).
UNASKED: none.  The new selector's default is the behaviour the shared code
already had, so no card except ORCA2 changes and no default moves; the ORCA2
card's value is the namelist value the record resolves.  No scheme selection,
tunable, threshold, cadence, resolution, timestep, carried state, data source or
previously-tolerated condition moved, and no stabiliser NEMO lacks was added.

## 10. Gate and test results at the round's final tip

| check | result |
|---|---|
| round-9 coefficient gate | **AT_BAR**; every row 0 unequal, all three controls firing, the card's masks equal to NEMO's mesh mask (0 / 799,200 each) |
| the same gate with either coefficient perturbed by one representable value | refuses, exit 2, naming the cell |
| EEN operand gate with the per-rank record | fold row **AT_BAR over all 180 longitudes** for both operands, all three controls firing |
| ORCA2 ladder gate | exit 4, `STOP_PRODUCTION_QSR_RGB_PIPELINE_GAP` at kt=1, no magnitude registered |
| receipt citation gate | PASS, 274 citations mapped, 0 failures, 0 map-audit failures, 0 unmapped |
| citation gate with a rigid two-line plant | refuses, exit 2 |
| the seven named push gates plus this round's new tests, at the final tip | **193 passed in 1,161.83 s** |
| the ocean-fidelity battery, once, with twelve workers | **1 failed, 1,528 passed, 7 skipped in 2,440.19 s** -- the one failure is the listed pre-existing sea-ice scalar-math provenance red (`A MY_SRC is not verbatim`); no other failing identifier appeared, and round 8's two stale stop assertions are green now that they follow the registry |

Forty-six mapped citations were rigidly re-anchored across two commits: the
transcription and then the review fixes inserted lines above them in six files,
the gate resolved every anchor to its new line, and no cited text changed.
