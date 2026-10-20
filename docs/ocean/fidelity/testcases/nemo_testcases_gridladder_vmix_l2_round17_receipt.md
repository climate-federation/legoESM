# GRIDLADDER-VMIX lane — round 17 (vmix_r17): L2 corrected and scored

Status: **L2 SCORED; DEBT; NOT CERTIFIED.** This round implements
Decision 138's second L rung: certified G5 plus ORCA2's standard isoneutral
tracer diffusion. Eddy-induced velocity remains off. The preregistration was
committed as `59e9622500` before implementation or any L2 measurement. No L2
NEMO record had been admitted at the first writing of this receipt. The
operator has now admitted the corrected acquisition as `oracle_l2_r17`; the
superseding review repair and D127 score are in section 6. The first smoke
record remains retained as `oracle_l2_r17_refused1` and is not scored.

Read first: the lane standing note, campaign synthesis and debt register,
round-16 survey, L1 receipt, and the ORCA2 round-313/314 receipts. GitHub issue
#1455 was not reachable from this sandbox; no claim from it was invented.

## 1. Survey: the exact ORCA2 selection and statements

ORCA2's six configuration overrides are individually pinned here:

| resolved key | value | source |
|---|---:|---|
| `ln_traldf_lap` | T | `ORCA2_OMIP_L4/EXP00/namelist_cfg:300` |
| `ln_traldf_iso` | T | `ORCA2_OMIP_L4/EXP00/namelist_cfg:301` |
| `ln_traldf_msc` | T | `ORCA2_OMIP_L4/EXP00/namelist_cfg:302` |
| `nn_aht_ijk_t` | 20 | `ORCA2_OMIP_L4/EXP00/namelist_cfg:304` |
| `rn_Ud` | 0.018 m/s | `ORCA2_OMIP_L4/EXP00/namelist_cfg:306` |
| `rn_Ld` | 200000 m | `ORCA2_OMIP_L4/EXP00/namelist_cfg:307` |

Its `EXP00/namelist_ref` resolves to `cfgs/SHARED/namelist_ref`; every other
resolved row is pinned rather than silently inherited:

| resolved key | value | source |
|---|---:|---|
| `ln_traldf_OFF` | F | `cfgs/SHARED/namelist_ref:918` |
| `ln_traldf_blp` | F | `cfgs/SHARED/namelist_ref:920` |
| `ln_traldf_lev` | F | `cfgs/SHARED/namelist_ref:923` |
| `ln_traldf_hor` | F | `cfgs/SHARED/namelist_ref:924` |
| `ln_traldf_triad` | F | `cfgs/SHARED/namelist_ref:926` |
| `rn_slpmax` | 0.01 | `cfgs/SHARED/namelist_ref:930` |
| `ln_triad_iso` | F | `cfgs/SHARED/namelist_ref:931` |
| `rn_sw_triad` | 1 | `cfgs/SHARED/namelist_ref:932` |
| `ln_botmix_triad` | F | `cfgs/SHARED/namelist_ref:933` |

The cfg rows above override the reference's `ln_traldf_lap`,
`ln_traldf_iso`, `ln_traldf_msc`, `nn_aht_ijk_t`, `rn_Ud`, and `rn_Ld`.
L2 deliberately retains G5's `ln_ldfeiv = F`: full ORCA2's EIV selection at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:317-323` belongs to a later rung.

All executed-routine citations below resolve to the preprocessed source under
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/`.

- **Slopes — `SUBROUTINE ldf_slp`.** The entry is
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfslp.f90:99`. It forms U/V density
  gradients at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfslp.f90:185-213`, applies
  the slope/stability caps and mixed-layer ramp at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfslp.f90:222-256`, Shapiro-filters U/V at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfslp.f90:260-275`, forms the W slopes at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfslp.f90:284-317`, filters them at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfslp.f90:319-334`, and exchanges them at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfslp.f90:338`. `stp_RK3` calls it once
  before the RK stages at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:189`;
  mode 20 is static, so the time-dependent coefficient call at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:193` is not taken.
- **Coefficient — `SUBROUTINE ldf_tra_init` and `SUBROUTINE ldf_c2d`.** The
  initializer enters at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:181`, reads
  reference then cfg at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:231-232`,
  resolves the standard isoneutral Laplacian at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:278-289`, enables slopes at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:312-313`, forms
  `zUfac = 0.5*rn_Ud` at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:353-358`, dispatches mode 20 at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:386-390`, and masks the result at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:427-434`. `ldf_c2d` enters at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfc1d_c2d.f90:125`, evaluates U and V
  independently as `pUfac*MAX(e1,e2)**knn` at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfc1d_c2d.f90:160-164`, then copies each
  coefficient down the column at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfc1d_c2d.f90:168-172`.
- **Explicit tensor and stabilising correction — `SUBROUTINE tra_ldf`,
  `SUBROUTINE traldf_iso_lap`, and `SUBROUTINE traldf_iso_a33`.** `tra_ldf`
  enters at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf.f90:69` and dispatches the
  standard operator at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf.f90:105-110`.
  `traldf_iso_lap` enters at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:73`, obtains A33/`akz` at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:154`, and keeps
  `ah_wslp2-akz` explicit at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:257-291`.
  `traldf_iso_a33` enters at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:746`, forms full A33 at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:780-795`, and forms the
  nonnegative MSC stabiliser at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:811-830`. The stage-3 caller
  places lateral diffusion before the vertical solve at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-749`.
- **Implicit K33 — `SUBROUTINE tra_zdf_imp`.** It enters at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:134`. With DDM off it builds one
  active-tracer matrix at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:171-176`,
  adds MSC `akz` to heat's `avt` at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:180-191`, and adds the same K33
  term to salt at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:198-208`. The
  last rule is the separate salt-matrix path already landed by ORCA2 round
  313; L2 extends it rather than duplicating it.

The G5 writer-free binary (`e8ebf4ae617f7aa61475776fcbe96b09e04bf09e7b846af39166bee1155808c7`)
and record binary (`41cd9a63844218e7c37e75708813649603b80df0366102b8ce3e340f89a60275`)
already compile these statements. Preflight found their `ldfslp.o`,
`ldftra.o`, `ldfc1d_c2d.o`, `traldf.o`, `traldf_iso.o`, and `trazdf.o`
`.text` identical to ORCA2's objects; the record build's `stprk3_stg.o` is
also identical. **Nothing is built.** The writer-free stage object differs
because of its instrumentation history, not these six operator objects.

Inputs are only the live grid metrics, T/S, masks, thickness, EOS/N2, and
namelist scalars already carried by G5. No new model state and no new data
file are needed.

## 2. L2 implementation and retained differences

`ORCA2L2-iso` is a G5-derived SI3 card. It extends the existing `gm_redi`
path shared with VORTEX SMT-3: native neutral slopes, NEMO caps and MLD ramp,
carried step-entry N2, reciprocal/live-QCO geometry, Shapiro filtering,
mode-20 metric coefficients from `rn_Ud`, standard rotated Laplacian, MSC,
and implicit K33. Its `kappa_GM` is 0, so EIV is off. L1's BBL, geothermal
heating, MLE, damping, tracer advection, momentum diffusion, TKE, EVD, DDM,
and IWM remain off. The initial-state comparison found 0 changed fields
between G5 and L2.

The first L2 trace exposed a selector-spelling defect: this carried-N2 path
accepted `nemo_seos` but passed `nemo_eos80` through to an internal API that
accepts `eos80`. The fix removes the `nemo_` prefix only in the active
carried-N2 branch. G5, V1-V5, L1, full ORCA2, and DINO controls demonstrate
that this branch-local correction is flag-off elsewhere.

Every known difference from NEMO's statements remains explicit:

1. The shared tensor uses JAX/vectorised association for horizontal flux,
   area reciprocal, divergence, and returned tendency; NEMO fixes pairwise
   parentheses and mutates the existing `Krhs`. VORTEX SMT rounds 14-24
   measured and held source-order candidates that missed their ratchets.
2. legoESM returns the lateral tendency to WS-RK3 composition instead of
   mutating NEMO's tracer array in place. Placement matches; bit identity is
   unclaimed until the L2 record exists.
3. legoESM hands computed MSC `akz` to its implicit solver; NEMO adds it while
   constructing `tra_zdf_imp`. Heat/salt K33 ownership matches round 313, but
   complete-stage arithmetic association is unmeasured.
4. `rn_Ld = 200000`, `ln_triad_iso = F`, `rn_sw_triad = 1`, and
   `ln_botmix_triad = F` are inert on the selected mode-20 standard branch.
   The deck reader nevertheless pins and validates them; no fake state was
   introduced merely to store inactive namelist values.

## 3. Purity, controls, and tests

The committed one-step probe normalises only Python address text. Results:

| card | lines | pure callbacks | SHA-256 | vs base |
|---|---:|---:|---|---|
| G5 | 168423 | 126 | `6b6df2ce2612e1422ed25ccd7557cab49fd5113bc05f9c9aab98ff3b68303887` | identical |
| V1 | 174013 | 135 | `b19946243d43141632c1c99061570cc4bc430e6960d05b5de2d772daaebb3dec` | identical |
| V2 | 174046 | 135 | `d927e9b0c8cbf224534a879aeed4742f9af851a7175631810a0ad26164ff8bbc` | identical |
| V3 | 174349 | 137 | `a89254938730cc12cfd20d9b09b79cc8c4379d367dfe38ca277f4fee6c659728` | identical |
| V4 | 176180 | 142 | `56d8aa4535b198d0e8f327813b2360836f5d4d41241b33d0266a2e33c572160d` | identical |
| V5 | 176180 | 142 | `56d8aa4535b198d0e8f327813b2360836f5d4d41241b33d0266a2e33c572160d` | identical |
| L1 | 169959 | 126 | `687eadaa7fecd04133c6ea2fa87e509b06acf4ba9b110fe06fdb3f2723564869` | identical |
| full ORCA2 | 134776 | 6 | `fcebf726db2f0258a50f67a46e19e0729ff0e22c26acab0de120528882df9c3c` | identical |
| DINO | 123235 | 6 | `89418f6a0b2eabae92618513aecda1470beeb80ea71dd8e2ab0c472efc83c18b` | identical |
| L2 | 188303 | 132 | `23e4b2c36229ca7d6ac16f840f036d936aae45c09437adc86b8c4f9d64187c3d` | differs from G5 |

G5, V1-V5, and L1 compare to their frozen round-16 traces. The archived
full-ORCA2/DINO traces are explicitly stamped `dirty=True`; the probe refused
them (`expected_identical=false`). Those two controls were therefore measured
directly in both the requested read-only base `fab19df07cd3` and this tree;
line counts, callback counts, and hashes above match exactly. VORTEX SMT-3's
shared constructor is structurally equal to its pre-extraction configuration;
the focused test checks all fields.

Tests and active controls:

- focused L2 file before the resumed predicate correction: **10 passed** in
  14.04 s;
- L1, VORTEX SMT-3 record/card, and V1-V5 regression files: **64 passed** in
  1447.86 s;
- post-EOS focused step-entry selection: **13 passed, 52 deselected** in
  17.54 s;
- L2 checker against G5 itself: 0 wet moves, 0 dry moves, 0 completed-tracer
  differences, 0 differences in T/S/u/v/SSH, `pass=False`, exit 1;
- planted dry-cell tracer movement is refused; a planted kt-1 primary
  barotropic velocity mismatch is refused; the required barotropic smoke
  comparison covers kt-1 `uu_b`, `vv_b`, `un_adv`, and `vn_adv`;
- final read-only preflight at `0435ddf85a4c`: `L2_PREFLIGHT_OK`; nothing
  built and the intended evidence root is
  `gridladder_vmix_rounds/round17/oracle_l2_r17`.

Final gates (CPU, x64, landing-script `PYTHONPATH`):

- exact default ORCA2 landing list, including the citation tests,
  private-import ratchet, and parallel-receipt guard: **145 passed in
  1160.51 s (19:20)**;
- this receipt's citation gate: **PASS**, 48 citations, 0 unmapped, 0
  failures, and 0 map entries failing audit;
- planted shift of the `ldf_slp` entry citation: gate **FAIL**, 1 failure,
  exit 1, as required;
- all new Python/shell files pass Ruff or `bash -n`; `git diff --check` is
  clean.

## 4. ACQUISITION_NEEDED (exact; operator executes)

From a clean checkout, run:

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_orca2iso/run.sh --run`

The default invocation is read-only preflight. The run mode is fail-closed:

- **Binaries:** reuse the two SHA-pinned G5 binaries above. Build nothing.
  Preflight verifies the six operator `.text` sections, record-stage object,
  committed producer scripts, and the admitted G5 stage stream
  (`7ac36ec2fe76bf9b07e270b48dcfa05ee41334792b8472914b1aa053b73a4f2c`).
- **Namelist diff against G5:** replace only G5's six-line OFF
  `&namtra_ldf` group with all fifteen resolved rows in section 1. G5 deck
  SHA is `c57eeb83d8f59c5b1d44df1295a10f5f30d28537358c8d327fcecc3d538f569e`;
  rendered L2 deck SHA is
  `0f683bf8838a6b979c1df7bff247cb3867b86061b0e9966df019233d8e62c14b`.
  `ln_ldfeiv = F`; the ice deck is unchanged at
  `9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8`.
- **Inputs:** exactly G5's sixteen SHA-pinned files: domain, initial T/S,
  eight atmospheric/precipitation/pressure inputs, two interpolation weights,
  chlorophyll, SSS, runoff, and `namelist_ref`. No new input.
- **Records:** the ordinary restarts at writable steps 2..N distributed over
  twins A/B/C; surface-input and step-entry records every kt; and the existing
  kt-1 stage-3 tracer stream containing before-`tra_ldf`, after-`tra_ldf`, and
  completed post-`tra_zdf` T/S. Twins must agree on every record they share.
- **Acting proof:** stage-3 after-minus-before lateral diffusion is nonzero in
  at least 1 wet T/S cell and exactly 0 in dry cells; completed T/S differs
  from G5; kt-2 entry T or S differs while u, v, and SSH are exact; the smoke
  kt-1 barotropic frame's primary velocities and advecting transports are exact
  against G5. The self-file inert control above proves the tracer test can fail,
  and a planted kt-1 frame mismatch proves the barotropic test can fail.
- **Logs:** require every resolved `&namtra_ldf` row, the standard-isoneutral
  operator banner, MSC T, mode 20, `rn_Ud = 0.018`, `rn_Ld = 200000`, and
  `ln_ldfeiv = F`; refuse BBL and geothermal heating.
- **Steps:** preflight; 2-step record-binary smoke plus acting/barotropic
  check; writer-free run toward 100 steps to locate the stability boundary;
  twins A/B/C through `min(boundary, 10)` rounded down to the 2-step surface
  cadence; admission. Only after that record exists may a later round score
  the preregistered labels.

Predictions P1/P2/P4 remain live and unscored P3 remains explicitly
withheld. There is no decision request. Independent review has not been run
by the author: **codex review: operator**.

## 5. Resume: refused smoke diagnosed; admission predicate corrected

The operator's reading is **confirmed in the compiled ORCA2 order**. In
`SUBROUTINE stp_RK3` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:62`), kt 1
calls `stp_2D` and writes `oracle_bt_frames_kt00000001.bin` immediately on
return (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:201-202`), before the three RK
stages. Stage 3 subsequently applies `tra_ldf` and completes the tracer update
through `tra_zdf` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-749`). The
end-of-step swap then makes that stage-3 `Naa` state the next step's `Nbb`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:233-237`).

At kt 2, `SUBROUTINE stp_2D`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:63`) recomputes density from
`ts(:,:,:,:,Kbb)` and immediately applies the hydrostatic pressure gradient
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:140-141`). That changed three-
dimensional momentum RHS is vertically averaged into the external-mode forcing
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:194-206`) and passed to the
split-explicit solve (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:302-303`). Thus
the kt-2 end-state `ub_e`, `vb_e`, `ubb_e`, and `vbb_e` are not required to
equal G5 after kt 1's T/S changed.

The corrected predicate keeps every other condition and replaces that invalid
restart comparison with exact comparison of the kt-1 barotropic frame against
G5. On `oracle_l2_r17_refused1/smoke_2step`, the kt-1 frame is byte-identical
(shared SHA-256 `a76f98edc18eb18ba417c245c197fccf57f8205aa3c9f7fefa60ca22f744bb51`),
whereas the kt-2 frames differ. The corrected checker reports the same wet/dry,
completed-tracer, and kt-2 entry counts as the refused run, with
`barotropic_unequal=0 pass=True`. Its unit check runs this exact refused smoke
record and then plants one kt-1 `uu_b` mismatch; the plant reports one unequal
value and fails. No score was run.

Focused L2 file after the resumed correction: **11 passed** in 12.83 s (CPU,
x64, landing-script `PYTHONPATH`). The exact default ORCA2 landing list,
including the citation tests, private-import ratchet, and parallel-receipt
guard: **145 passed in 1206.76 s**. This receipt's citation gate: **PASS**, 55
citations, 0 unmapped, 0 failures, and 0 map entries failing audit; a planted
shift of the new kt-2 density/HPG citation exits 1. Clean-tree acquisition
preflight: `L2_PREFLIGHT_OK`, default evidence root
`gridladder_vmix_rounds/round17/oracle_l2_r17`; nothing built.

## 6. Superseding review repair and D127 score

The corrected acquisition completed and was admitted:
`L2_ACQUISITION_PASS steps=1..10 restarts=2..10`. Its evidence root is
`gridladder_vmix_rounds/round17/oracle_l2_r17`, with identical admitted twins
A/B/C. The independent review then held four omissions in the first build.
All four are repaired before the score:

1. Mode 20 now always selects the shared metric coefficient path. The card's
   U/V fields are bit-identical to the independent NEMO mesh calculation
   `0.009*MAX(e1,e2)` at
   `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfc1d_c2d.f90:160-164`, before the mask at
   `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:427-434`. Their unmasked extrema
   are 134.99730994125315 and 2001.5806220738168 m2/s. The test reads the
   admitted `mesh_mask.nc`; it no longer derives both sides from the card.
2. The once-per-step slope density is EOS-80, matching the executed
   `CALL eos(ts,Nbb,rhd)` at
   `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:188`. Three fixed kt-1 NEMO PRD
   values and the full independently evaluated field are checked; replacing
   it with the simplified EOS fails both the selector and record controls.
3. The northern pivot is open on 1,319 wet V levels. Density/tracer gradients,
   live QCO V geometry, Shapiro halos, and the returned U/V/W/W slopes all use
   the shared pivot maps; the four slopes use NEMO's associated sign-minus
   exchange at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfslp.f90:338`. No second fold
   helper was introduced. Fixed NEMO pivot values supplement the structural
   fold checks.
4. The focused suite has independent value anchors for all three corrections.
   Its slow kt-1 record gate runs the complete selection and then individually
   removes metric scaling, EOS-80, and the fold. Every omission produces a
   distinct DEBT tracer signature. That gate passed in 430.51 s; the ordinary
   focused tests passed 14/14 before the record arm, and the three independent
   metric/EOS/fold value tests passed together after the final anchors.

Because item 1 changes the shared SMT constructor, both downstream registries
were re-run on their admitted NEMO records. SMT-3 retained all 50 row statuses
and the same first-over-bar boundary (kt=2: T/u/v/SSH); 34 normalized residuals
moved, 30 toward its oracle and four away. SMT-4 reproduced all 50 normalized
residuals and statuses exactly. Thus the shared-constructor change does not
invalidate either registered verdict.

### D127 result

The exact preregistered scorer ran the independent chain for kt=1..10 and the
given-NEMO-entry starts at kt=1,3,5,7,9, each with its second-step continuation;
together the latter covers kt=1..10. It scores SSH, barotropic U/V, 3-D U/V,
T and S, and every full/partial/dry cell in each available restart. D127 is
exact/AT-BAR at normalized error <=1e-15 and DEBT otherwise.

- Independent: 68 surface rows = 5 exact and 63 DEBT; 36 full-depth class
  summaries = 36 DEBT. Every dry-cell count is zero. At kt=1 SSH, barotropic
  U/V and 3-D U/V are exact; T/S first become DEBT at the L2 stage. Thereafter
  the tracer difference propagates through the independent dynamics.
- Given entry: 68 surface rows = 25 exact and 43 DEBT; 20 full-depth class
  summaries = 20 DEBT. Every odd entry resets SSH and all four velocity rows
  to exact NEMO values, while T/S are DEBT in that same first L2 step; each
  even continuation then carries propagated dynamic debt. Every dry cell is
  exact.

The stage walk established only that stages 1 and 2 are exact and stage-3 T/S
is the first differing state. Its attribution of the first differing statement
to `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:287-291` is **retracted** by
the admitted operand walk in section 7: the first mismatch was already in the
face thickness consumed at the top of the horizontal-flux loop. The score in
this section is superseded below.

## 7. HOLD disposition: GYRE isolation, provenance, operator walk, and re-score

### GYRE is unchanged

The carried-slope BN2 rounding selection is now explicitly restricted to the
non-TKE L2 branch. GYRE's TKE path continues to obey its own false
`nemo_bn2_source_rounded` selector. The production spy observes two GYRE BN2
calls and `source_rounded=False` on both; removing the non-TKE guard makes that
value check fail.

The ten-step GYRE record check was run both at parent
`56aeb343b5f7beff61891b1ce3cd1316c87ce3bb` and on this correction. All 50
trajectory field rows, all 20 barotropic rows, both first-over-bar summaries,
and the status are identical. The two residual archives are byte-identical at
SHA-256
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
Both JSON records, both residual archives, and their producer/comparison
manifest are durable under
`gridladder_vmix_rounds/round17/gyre_bn2_regression/`; no evidence remains
dependent on `/tmp`.

### The slope observer is reproducible and admitted

The exact write-only instrumentation is committed as
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_orca2iso/ldfslp_slope_dump.F90.patch`
(SHA-256
`a8d8223cc3be43c58e440f30c8d4924afe4c8a362e64c93e66eeb88aaf324f8f`).
The dump, instrumented source, executable, build configuration, namelists,
log, marker, checksums, and provenance now live under
`gridladder_vmix_rounds/round17/oracle_l2_r17/observer_ldf_slopes/`.
The unmodified producer source is pinned at
`8d0e6d42e7c19dca848734179584198d2735b154ab3fbea02a7f693129dc53de`,
the instrumented source at
`2cd8e353c2b85c5eb6d5fc040043325c722e427508c86e4f1e5d118ca96d987b`,
the executable at
`91064e46d04610c47e2786affea83ead177e24cc07cb88e7a3fb18cd651745c5`,
and the dump at
`e8b07d7f8dfffdbbe69020f125eb628af9529ee8079595a5a5e65572023d5724`.
The observer reproduces 42 of 45 same-named admitted streams byte for byte,
including both step-entry streams and every stage stream at kt 1 and 2; the
three moving streams are unrelated SI3 observers and do not feed the dump.

The admitted production value gate runs the actual jitted L2 step. It checks
the four final slope arrays exactly against that dump. Before the source-order
repair, U and V differed in 1,610 and 1,465 cells by at most
`4.336808689942018e-19`, while both W slopes were exact. A statement walk found
PRD, PN2, gradients, metrics, bounds, and masks exact; the first U/V mismatch
was the face-depth assignment. NEMO forms `zdepu` and `zdepv` with explicit
scalar statement boundaries at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfslp.f90:245-250`. L2 alone now preserves
those boundaries through the U/V slope and Shapiro statements. All four
production slope arrays are exact; disabling that selector fails the dump
comparison. L1's BBL and L2 now use the same shared north-neighbour helper.

### Compiled-order `traldf_iso_lap` walk and correction

At kt 1, SSH, barotropic U/V, and 3-D U/V were exact, but the old stage-3
lateral trend differed in all 430,168 wet tracer cells: 32,227 exceeded
`1e-6` of scale, 1,431 exceeded `1e-3`, and the maximum normalised difference
was 0.16. Tracer, coefficient, masks, and all slopes matched the admitted
observer before entry to the operator. The old attribution to the final
increment was therefore false.

Walking `traldf_iso_lap` in compiled order showed exact tracer-gradient
statements at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:183-204`. The first differing
statements were `zA11` and `zA22`: NEMO multiplies native `e3u_3d`/`e3v_3d`
by the live `r3u`/`r3v` at Kmm at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:230-231`; the candidate had
aliased both reference ladders to `e3t_0` and used step-entry sea level.
Stage 3 defines Kmm as the stage-2, N+1/2 time level at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:229`.

The measured lead separated the two inputs. Half-step sea level with the old
T-point alias left 24,929 cells above `1e-6`, 1,433 above `1e-3`, and the same
0.16 maximum. Native face reference ladders plus half-step Kmm reduced the
direct-operator residual to 7,878 cells above `1e-6`, zero above `1e-3`, and
maximum `3.55187264925e-4`. The fix is L2-only: resolve the already-carried
native face ladders and use the materialised stage-2 sea level only for L2's
stage-3 Redi flux geometry. The jitted value gate exactly reconstructs both
face arrays independently, pins their canonical hashes, and plants two
failures: the T-point alias changes 18,803 U-face and 18,300 V-face values,
while step-entry sea level changes 413,030 U-face and 415,175 V-face values.

### Superseding kt-1 score

With all HOLD repairs enabled, kt-1 SSH, barotropic U/V, and 3-D U/V remain
exact/AT-BAR. T and S remain DEBT, but the score is improved and the owner is
no longer misattributed:

| field | unequal | max abs | RMS |
|---|---:|---:|---:|
| T | 16,428 | `5.092867354017372e-05` | `4.3848188806266483e-07` |
| S | 16,428 | `2.572830925373637e-05` | `2.6974345661453047e-07` |

The remaining residual has not been localised past the corrected face
geometry and is not claimed to begin at the final divergence increment.
Metric, EOS-80, north-fold, and Kmm-geometry omissions each have fixed exact
T/S value tuples in the record gate; the former `!= signature` assertions are
gone. The native-face and U/V source-order plants are checked directly at the
operator operands, where their value effects are observable rather than
rounded away in the final state.

The review-confirmed purity result is retained: one-step jaxprs are identical
for G5, V1-V5, OMT-0-4, rung 0, VORTEX, and SMT-1/2. L1, full ORCA2, DINO,
and SMT-3/4 have changed jaxprs but bit-identical values. The final test and
gate results supersede the earlier counts in this receipt:

- GYRE production BN2 selector: **1 passed**; ten-step record payloads and
  residual archive are bit-identical as recorded above;
- focused L2/L1/private-import set: **32 passed, 2 deselected in 248.30 s**;
- jitted production face/slope value gate: **1 passed in 158.57 s**;
- kt-1 selected score plus four exact omission plants: **1 passed in
  540.25 s**;
- exact clean ORCA2 landing list from `claude_rounds/land.sh`, including the
  citation tests, private-import ratchet, and parallel-receipt gate:
  **145 passed in 1294.27 s**;
- this receipt's explicit citation gate: **PASS**, 61 citations, 0 unmapped,
  0 failures, and 0 map entries failing audit. `git diff --check` is clean.

codex review: operator

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r17/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: L2 SCORED; DEBT; NOT CERTIFIED
