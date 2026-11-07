# RECEIPT — TSUNAMI lane, round 2 (the key_RK3 build)

Date 2026-10-08. Lane tip at start `5b9bde8a196a`. Preregistered in
`PREREG_nemo_testcases_l1_tsunami_round2.md` (commit `182f3604c`), before
any record and before the seam probe.

Status: **STOPPED_FOR_RECORD.** The RK3 acquisition is written and
preflighted; NEMO has not been run. The card is on the RK3 program and still
refuses execution (blockers B4, B4j, B6, B7).

Take-aways:

1. **The deviation is recorded** (DECISION 100): the card carries
   `TSUNAMI_DEVIATIONS = (("key_RK3", False, True, "DECISION 100"),)` and a
   test proves key_RK3 is the ONLY key that differs from `cpp_TSUNAMI.fcm`.
2. **Tracers are not absent** in the RK3 program (correcting the brief): T
   and S are stepped every stage by the thickness ratio alone, with no
   advection, so they stop being uniform where ssh moves and feed the
   pressure gradient. legoESM has no such arm (new blocker B7).
3. **The card's step walls the j-seam.** Measured in legoESM alone: one step
   is bitwise translation-equivariant across the i-seam, not across the
   j-seam (402 ssh cells unequal), and the existing y-wrap does not change
   that (new blocker B4j).

## 1. The RK3 step program (survey)

Full table with every statement cited: preregistration section 1. In one
line: the case's own stprk3 runs zero forcing, the external mode once, three
NEMO stages with the hybrid barotropic update, the swap, the ssh
extrapolation and output; it cuts zdf_phy, eos_rab/bn2, slopes, coefficient
updates, restart and control. Every stage still calls eos + hpg + EEN
vorticity (stages 2-3), the flux-form (1 + r3) momentum step (stages 1-2),
dyn_zdf (stage 3), the barotropic correction, the tracer step, and the
halo exchange. Momentum and tracer advection and lateral diffusion are OFF
operators that are still called.

The case's diawri (diff against NEMO's) writes only ssh, uu_b, vv_b, and at
32 bits.

## 2. What was built

| artefact | what it is |
|---|---|
| `run_rk3.sh` | builds TSUNAMI_OMIP_L1_RK3 (+ _P3) from the shipped case with key_RK3 added and key_xios dropped; refuses unless the compiled keys are exactly key_RK3 key_qco key_vco_1d, the step calls stp_RK3 and not stp_MLF, the stprk3 is the case's, and both writers are present in P3 and absent in the reference (D-TSU-1 closed) |
| `stprk3_rk3_step_record.patch` | additive patch on the case's stprk3, reusing the round-1 writer module: kt 1..10 step entry, after stp_2D, each stage's Naa before the swap; every kt the after-step state and the extrapolated ssh |
| substep record | round 1's gated copy of VORTEX's dynspg_ts writer, unchanged (shared RK3/MLF code, VORTEX ran it under RK3) |
| record checker | required `--program mlf|rk3`; RK3 records admitted by EXACT group set; output field corrected to `sovvbaro` |
| card | RK3 program, deviation record, RK3 stage selectors stated, `rn_atfp` removed (read only in the leapfrog branch), j_periodic card data, eight further stage selectors stated after review |
| y-wrap scope | `halo_latlon.meridional_periodicity(enabled)`: sets the existing global for a block and restores it, also on error |

Preflight: patches apply unfuzzed, the step patch removes no line, the
patched stprk3 and the case's own stpmlf and diawri pass `gfortran
-fsyntax-only` under key_qco key_vco_1d key_RK3 against the LOCK_EXCHANGE
RK3 build's modules (`RK3_PATCH_SYNTAX_OK`, a misspelt-symbol plant refused).

## 3. Blocker dispositions

| id | disposition |
|---|---|
| B1, B2 | void (DECISION 100); removed |
| B3 | CLOSED as card data: `NEMOTestcaseCard.j_periodic`, default False, TSUNAMI sets the deck's ln_Jperio; validator refuses a mismatch. No shared card executor exists yet, so the field is consumed only through the scope helper (see B4j) |
| B4 | OPEN. NEMO statements across the periodic seams named below; none compared to NEMO yet |
| B4j | NEW. The card's step is not j-periodic (measured) |
| B5 | CLOSED for legoESM: the step already accepts one wet level; the only door is the coordinate builder's opt-in flag, set only by this card. Rest state, zero forcing, one step: every field bitwise unchanged; the same step from the card's bump moves ssh (non-vacuity) |
| B6 | NEW. No "no momentum advection" arm; the carrier runs UP3 |
| B7 | NEW. No "tracer stepped without advection" arm; the carrier runs FCT2 |

B3 caller grep (`set_meridionally_periodic|get_meridionally_periodic|meridional_periodicity(`,
saved at `round2/b3_caller_grep.txt`): readers in the atmosphere sharded
step, `operators_latlon_cgrid.py`, and two ocean operator sites; the only
setter callers are the inertial-oscillation test (unchanged) and this
round's tests. Library default unchanged (False); every other card's
j_periodic is the default False.

B4 statements NEMO executes across the seams, and the card's halo path:

| NEMO statement | i-seam | j-seam |
|---|---|---|
| sub-step halo of ua_e, va_e, hu_e, hv_e, hur_e, hvr_e, ssha_e (dynspg_ts) | equivariant | walled |
| halo of the time-mean transports un_adv, vn_adv (dynspg_ts) | equivariant | walled |
| halo of uu_b, vv_b after the transport-to-velocity division (dynspg_ts) | equivariant | walled |
| halo of the N+1 thickness ratios r3u, r3v, r3f (stage 1) | equivariant | walled |
| halo of uu, vv, T, S at each stage's after level | equivariant | walled |

"Equivariant" = the card's step is bitwise translation-equivariant across
that seam (test pins it); it is not a NEMO comparison. "Walled" = the step
is not equivariant across it: 402 ssh, 416 uu_b, 441 vv_b cells unequal, the
same count with the y-wrap scope off. Nothing transcribed.

## 4. Review

Single review (codex), verdict **DO NOT SHIP** on `5b9bde8a1..153a11c80`
(log `round2/codex_review.txt`). Disposition:

- CONFIRMED, fixed: the build guard passed vacuously when a preprocessed
  file was missing; it now refuses (six guard cases exercised).
- CONFIRMED, fixed: eight stage-path selectors were inherited, not stated;
  stated, checked against NEMO's resolved deck, refused if switched on.
- CONFIRMED, fixed: two comments said the RK3 program never evaluates
  density and makes no ldf calls; both false, corrected.
- CONFIRMED, fixed by weakening the claim: j_periodic is consumed by no
  executor; the comment now says so (B3 row above).
- PLAUSIBLE, accepted: R5 (bitwise output vs record) is ill-posed because
  the output is 32-bit. Amended below; the frozen text is kept.
- Codex found no defect in stage-before-swap placement, record names, the
  exact RK3 group sets, `sovvbaro`, face rolling or the rest-state test.

## 5. Retractions and amendments

- Round 1's checker would have refused every record: it looked for an
  output field the case never writes (`somebaro`). Fixed; a test reads the
  case's histwrite names.
- Round 1's receipt said the case's output routine "writes no state"; its
  non-XIOS branch writes ssh, uu_b, vv_b.
- R5 AMENDED (not scored as written): NEMO's output equals the record's
  f_ssh_bb rounded to float32. Debt D-TSU-2: the passivity admission compares
  float32 outputs, so it cannot see a last-bits perturbation; the writers
  are read-only by construction.

## 6. Gates

| gate | result |
|---|---|
| card + checker + inertial oscillation + private-import ratchet + carried-pair ratchet | `67 passed, 1 xfailed` (log sha256 `de33656e53d5a66e`; its 3 citation failures were the shifted recipe spans, fixed by re-anchor) |
| citation-gate tests after re-anchor | `17 passed` (sha256 `cf3445754f64a26b`) |
| recipe module + VORTEX card tests | `99 passed, 3 deselected` (sha256 `0e520a98074b1f75`) |
| card + checker + citation-gate tests, final tree | `53 passed, 1 xfailed` (the xfail is the strict B4j pin; log sha256 `0db67918961a3c3a`) |
| pre-existing red, not this round's | the VORTEX test's certified-card digests (GYRE, LOCK_EXCHANGE, OVERFLOW) fail identically at `5b9bde8a1` |
| plants (each reverted, tree clean) | output field `somebaro` RED; an RK3 stage group dropped RED; j_periodic default True RED; scope not restoring RED; deviation unrecorded RED; i-seam wrong shift RED |
| preflight on the committed tree | `PREFLIGHT_OK`, exit 0; shipped keys planted with key_RK3: `REFUSE`, exit 68 |
| citation gate, this receipt (from `## Citations`) | `PASS`, 30 citations, 0 failures, clean tree at `a501e96837e6`, json sha256 `7b4b122a1ea7da2f` |
| citation gate, planted shift of `stprk3_stg.F90:552-554` | `FAIL` (`SYMBOL-NOT-AT-LINE`), exit 1, json sha256 `e866adb2c088eea4` |

## 7. Choices made this round

| choice | ASKED or UNASKED |
|---|---|
| build with key_RK3 | ASKED (DECISION 100) |
| drop key_xios, ln_meshmask on | UNASKED, round-1 / VORTEX precedent |
| new run_rk3.sh beside the leapfrog run.sh (kept, now void) | UNASKED |
| entry frame records only the slots RK3 initialises | UNASKED |
| j_periodic default False on the shared card type | UNASKED, keeps every other card unchanged |
| B4j, B6, B7 declared rather than built | UNASKED, follows "no unmeasured transcriptions" |

Every UNASKED item is offered for revert.

## 8. ORCA2 pointer

ORCA2 rung 0 runs this RK3 program: the same stage routine and the same
dyn_spg_ts, with nn_bt_flt = 3. The seam statements of section 3 are the
ones ORCA2 runs across its own periodic seam and north fold. A bit-exact
TSUNAMI stage ladder is evidence for ORCA2's stage hybrid update, flux-form
(1 + r3) step and barotropic correction; the filter weights do not transfer.

## 9. How to acquire, and what is open

```
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_tsunami/run_rk3.sh
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_tsunami/run_rk3.sh --run
```

Evidence lands in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round2/oracle_tsunami_r2`.

OPEN for round 3: score G1-G6 and R1-R6 (R5 as amended) against the record;
build the kt = 1 ladder scorer; decide B6/B7's shared arms with the record
in hand; B4j (the card's barotropic path ignores the y-wrap); D-TSU-2.

## Citations

The case's step: `tests/TSUNAMI/MY_SRC/stprk3.F90:98`,
`tests/TSUNAMI/MY_SRC/stprk3.F90:107`, `tests/TSUNAMI/MY_SRC/stprk3.F90:113`,
`tests/TSUNAMI/MY_SRC/stprk3.F90:118`, `tests/TSUNAMI/MY_SRC/stprk3.F90:123`,
`tests/TSUNAMI/MY_SRC/stprk3.F90:125`, `tests/TSUNAMI/MY_SRC/stprk3.F90:129`,
`tests/TSUNAMI/MY_SRC/stprk3.F90:136`. Its leapfrog step is empty under the
key: `tests/TSUNAMI/MY_SRC/stpmlf.F90:37`. NEMO calls it:
`nemogcm.F90:166`. Output at 32 bits: `tests/TSUNAMI/MY_SRC/diawri.F90:626`,
and the meridional field name: `tests/TSUNAMI/MY_SRC/diawri.F90:655`.

The stage routine: the hybrid update `stprk3_stg.F90:44`; stage 1
`stprk3_stg.F90:143-145`; the ratio halo `stprk3_stg.F90:158`; stage 3
`stprk3_stg.F90:224-226`; density `stprk3_stg.F90:322`; the momentum step
`stprk3_stg.F90:373-378`; dyn_zdf `stprk3_stg.F90:430`; the tracer step
`stprk3_stg.F90:552-554`; the stage halo `stprk3_stg.F90:636`.

The external mode: density `stp2d.F90:127`; the after-ratio guess
`stp2d.F90:151`; no momentum advection `dynadv.F90:129`; no tracer
advection `traadv.F90:461`; the sub-step weights `dynspg_ts.F90:245`; the
sub-step halo `dynspg_ts.F90:787-789`; the transport halo
`dynspg_ts.F90:854`; the velocity halo `dynspg_ts.F90:895`; rn_atfp is read
only in the leapfrog branch, `dynspg_ts.F90:920`.
