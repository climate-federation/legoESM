# Round 180 receipt — developed `ldf_slp` causal walk

**Status: HELD.**  No physics, configuration, carried state, restart schema,
card default, or trajectory baseline changed.  The first non-bit causal row
before the developed `uslp` boundary is the inherited mixed-layer index
`nmln`: 14 of 600 wet columns differ by exactly one vertical level under both
the complete production JIT step and complete eager execution.  The first
local write, `zhmlpt`, differs only after that input boundary.  No downstream
`ldf_slp` statement is eligible to land while `nmln` remains open.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round180.md`, commit
`d6165c5f7`.  Final science producer: `adbe3e1bc`.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round180/`.

## Admission and compiled order

The existing Round-179 record passed the committed admission unchanged:

* step-1080 and step-1440 restarts are byte-identical to the uninstrumented
  Round-132 daily restarts;
* the fixed 7,324,076-byte layout, header, current producer stamp, 18,000-cell
  wet census, finite-value checks, and integer selectors pass;
* the record's `uslp` and `vslp` rebuild bit-for-bit from its own raw slopes,
  masks, and compiled Shapiro association; and
* the magic-byte corruption prints `STATUS PLANT-FIRED` and exits 69.

The active compiled step calls `ldf_slp(kstp,rhd,rn2b,Nbb,Nbb)` at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/stprk3.f90:173-180`.
The routine reads `nmln`, live depth, and mixed-layer inverse depths at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/ldfslp.f90:187-229`,
then forms density gradients at `:231-261` and the face gradients,
limiter, selectors, recurrence, and Shapiro outputs at
`:270-361`.  The same build produces `nmln` earlier by calling
`zdf_mxl` at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/zdfphy.f90:326-330`;
that producer initializes the index, accumulates positive `rn2b*e3w`, updates
the last below-threshold level, and writes the live mixed-layer depth at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/zdfmxl.f90:109-124`.

## Production-step result

Both modes start from NEMO's admitted day-180 restart.  In each mode the
diagnostic step and a separately compiled ordinary step differ in zero
returned-state bytes.  JIT is authoritative; eager names the same first row.
Rows after `nmln` are directional propagation evidence, not independent
attributions, because their registered inputs are already non-bit.

| row | owner | JIT unequal | JIT max abs | eager unequal | eager max abs |
|---|---|---:|---:|---:|---:|
| `nmln` | inherited | 14 | 1.00000000000000000e+00 | 14 | 1.00000000000000000e+00 |
| `gdept_1d` | inherited | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `r3t_Kmm` | inherited | 600 | 1.10710594337926072e-16 | 600 | 1.10710594337926072e-16 |
| `ssmask` | inherited | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `zhmlpt` | owned | 14 | 1.19126800665216024e+02 | 14 | 1.19126800665216024e+02 |
| `r1_hmlu` | owned | 17 | 2.70802751099338615e-02 | 17 | 2.70802751099338615e-02 |
| `r1_hmlv` | owned | 19 | 2.70801937640774773e-02 | 19 | 2.70801937640774773e-02 |
| `r3u_Kmm` | inherited | 600 | 2.52902015193505469e-16 | 600 | 1.66140430406247486e-16 |
| `r3v_Kmm` | inherited | 596 | 2.47862169157342382e-16 | 596 | 1.65710137669042301e-16 |
| `miku` | inherited | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `mikv` | inherited | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `hmlp` | inherited | 14 | 1.33101408745854371e+02 | 14 | 1.33101408745854371e+02 |
| `gdepw_1d` | inherited | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `mikt` | inherited | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `r1_hmlw` | owned | 14 | 1.69981978641380108e-02 | 14 | 1.69981978641380108e-02 |
| `prd` | inherited | 17,999 | 1.11022302462515654e-16 | 5,238 | 2.22044604925031308e-16 |
| `tmask` | inherited | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `umask` | inherited | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `vmask` | inherited | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `zgru` | owned | 17,368 | 2.11834043287278864e-04 | 7,285 | 2.11834043287284501e-04 |
| `zgrv` | owned | 17,063 | 2.26489838731620905e-04 | 7,062 | 2.26489838731636084e-04 |
| `pn2` | inherited | 17,400 | 1.89067425612906258e-06 | 17,400 | 1.89067425612906258e-06 |
| `zdzr` | owned | 18,000 | 3.89435353492625770e-05 | 18,000 | 3.89435353492625770e-05 |
| `r1_e1u` | inherited | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `r1_e2v` | inherited | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `zau` | owned | 17,368 | 1.99843437063470636e-09 | 7,285 | 1.99843437063475971e-09 |
| `zav` | owned | 17,062 | 2.13669659180774469e-09 | 7,062 | 2.13669659180788779e-09 |
| `zbu_raw` | owned | 17,400 | 2.00010847461573780e-05 | 17,400 | 2.00010847461573780e-05 |
| `zbv_raw` | owned | 17,100 | 3.46357293075403401e-05 | 17,100 | 3.46357293075403401e-05 |
| `e3u_live` | inherited | 580 | 1.00039977407712986e+01 | 580 | 1.00039977407712986e+01 |
| `e3v_live` | inherited | 570 | 1.00040051288949350e+01 | 570 | 1.00040051288949350e+01 |
| `e3w_1d` | inherited | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `zbu_limited` | owned | 17,400 | 2.00010847461573780e-05 | 16,828 | 2.00010847461573780e-05 |
| `zbv_limited` | owned | 17,096 | 3.46357293075403401e-05 | 16,498 | 3.46357293075403401e-05 |
| `iku` | owned | 31 | 2.60000000000000000e+01 | 31 | 2.60000000000000000e+01 |
| `ikv` | owned | 41 | 2.70000000000000000e+01 | 41 | 2.70000000000000000e+01 |
| `zfi` | owned | 11 | 1.00000000000000000e+00 | 11 | 1.00000000000000000e+00 |
| `zfj` | owned | 11 | 1.00000000000000000e+00 | 11 | 1.00000000000000000e+00 |
| `zmli` | owned | 22 | 1.00000000000000000e+00 | 22 | 1.00000000000000000e+00 |
| `zmlj` | owned | 22 | 1.00000000000000000e+00 | 22 | 1.00000000000000000e+00 |
| `zdepu` | owned | 4,626 | 2.64933889576282056e-02 | 580 | 2.64933889576282056e-02 |
| `zdepv` | owned | 4,575 | 2.64934085234500571e-02 | 570 | 2.64934085234500571e-02 |
| `zuslp_pre` | owned | 1,898 | 2.32638117883178999e-05 | 1,898 | 2.32638117883382456e-05 |
| `zvslp_pre` | owned | 1,887 | 3.82344075268032513e-05 | 1,887 | 3.82344075267675133e-05 |
| `zwz` | owned | 17,400 | 4.42004605680172431e-04 | 17,400 | 4.42004605676869517e-04 |
| `zww` | owned | 17,092 | 1.03001762598001981e-03 | 17,091 | 1.03001762597274699e-03 |
| `zuslp_post` | owned | 2,478 | 2.32638117883178999e-05 | 2,478 | 2.32638117883382456e-05 |
| `zvslp_post` | owned | 2,457 | 3.82344075268032513e-05 | 2,457 | 3.82344075267675133e-05 |
| `uslp` | owned | 16,820 | 1.84356055604884082e-04 | 16,820 | 1.84356055603913938e-04 |
| `vslp` | owned | 16,530 | 2.73965483207163664e-04 | 16,530 | 2.73965483211071996e-04 |

The JIT/eager cell counts differ for floating-point `prd` and its consumers
because full-step XLA fusion changes last-bit association.  They do not differ
on the first boundary: both modes identify the same 14 integer columns.

## Prediction verdicts and control

* **CONFIRMED:** the first non-bit causal row is `nmln`.
* **CONFIRMED:** `prd` is non-bit (17,999 JIT; 5,238 eager; at most two ULP).
* **CONFIRMED:** `pn2` is non-bit (17,400 cells, maximum
  `1.89067425612906258e-06 s-2`).
* **CONFIRMED:** JIT and eager name the same first boundary.
* **CONFIRMED:** the first non-bit local row follows the inherited boundary.
* **REFUTED AS STATED:** the preregistered one-ULP mutation cannot be applied
  to `nmln`, because it is an integer branch index.  The smallest representable
  causal mutation is one vertical level.  That production-step plant fires:
  one `nmln`, one `zhmlpt`, two `r1_hmlu`, and four `uslp` cells move; `prd`,
  `pn2`, and `gdept_1d` remain unchanged; the returned state moves 305 bytes.

## Landing and trajectory disposition

There is no candidate patch.  `nmln` is an input to `ldf_slp`, produced by the
earlier vertical-physics mixed-layer recurrence.  Changing a slope statement
would fit a downstream symptom.  Therefore the Decision 43/45/55/59 ladder,
month, year, DINO, tanks, generic recipe, and ORCA2 gates do not run, and no
before arm moves.

The unchanged certified context is: first over bar kt=3; kt2 U/V
`8.330180021824198e-17 / 9.714451465470120e-17`; kt3 T/S
`4.9400710722e-7 / 4.0086298725e-8`; day-30 T RMS
`6.57257437477e-5 K`; day-240 `1.64483607012e-2 K`; and day-360
`1.12256600186e-2 K`.  DINO shares the native mixed-layer/slope path and is
explicitly at risk for any future producer candidate; ORCA2 remains
UNMEASURED-with-spec for such a candidate.

## Independent review and tests

The required read-only `codex exec` pass was attempted against the committed
diff.  It produced no scientific verdict because the in-process app-server
could not initialize in the read-only sandbox.  Its terminal line is quoted
verbatim:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore: **independent review unavailable in-sandbox**.  This is not a SHIP,
HOLD, or DO NOT SHIP verdict.

The focused battery covers the year-owner harness, native GM/Redi operator,
and NEMO testcase recipe: **175 passed in 110.77s**.  The receipt citation gate
and shifted-citation plant are recorded after this receipt is committed.

| artifact | SHA-256 |
|---|---|
| `admission.log` | `df4026802ac86240064074d3e39007b2333428d00288a32ba8e2f92ff7b16572` |
| `admission_plant.log` | `5d0b7fd4dcb83a8f141d423e6f6f751a24f79d68a6730a2b6d1f24b7c51b580b` |
| `daily_record_audit_final.json` | `b2bdc59ae036fd488cf03bd1bc05a898392d062aa3fc3b1a3bc420e51753655a` |
| `developed_slope_walk_final.json` | `3179f470bef1335c2c2e2dcf2aa9bd90358c85bfb27feaead69b741fc1971604` |
| `developed_slope_plant.log` | `5fb991804e6ce52905870f51eb85f0733e6387b8ef8b14c05a5fc166d0552fe9` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |
| `focused_tests.log` | `58ec60d90f8a73339c3b010b508be1654fd022bf9063e9ce5cc927c05ac01910` |

Choices made: none.  The new hook is default-off and diagnostic-only; no card,
scheme, threshold, cadence, state field, or production default changed.

## OPEN — Round 181

Stay upstream in execution order and walk `zdf_mxl` at the same developed
day-180 entry.  Extend this harness; do not create another bridge.  Score the
compiled recurrence at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/zdfmxl.f90:109-124` from
initialization through each
`MAX(rn2b,0)*e3w` accumulation and the strict below-threshold index update.
Use the admitted Round-179 `pn2`, `e3w_1d`, `r3t_Kmm`, `nmln`, and `hmlp`
operands first; acquire only a per-level cumulative stream if those operands
cannot discriminate the first statement.  Test the one-variable hypothesis
that legoESM recomputes the mixed-layer criterion instead of consuming the
same carried step-entry `rn2b` that NEMO uses.  A candidate must make `nmln`
bit-exact given NEMO inputs under production JIT and eager, then pass the full
month/year/card/tank gate; DINO shares the statement and must be measured.
