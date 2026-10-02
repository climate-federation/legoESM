# GYRE round 202 — adopting the shared stage-one tracer ratio order

**Date:** 2026-10-02
**Lane:** `fidelity/nemo-testcases-l2-gyre-codex2`
**Base:** `52e9a3112ee5` (round 201)
**Adopted from:** ORCA2 lane commit `a0b2f7a5da06f416e530361355df8603b7826a3d`
(2026-09-27, "preserve shared stage-one ratio order"), admitted against the
GYRE gate by ORCA2 round 102
(`nemo_testcases_l4_orca2_round102_gyre_admission_receipt.md`).
**Disposition:** **HELD — the statement is transcribed and measured, but the
ORCA2-registered year numbers DO NOT reproduce on this lane past day 90, so
nothing was re-pinned and the model change does not land.**

## The statement, and its citation

NEMO's RK3 stage 1 needs the thickness ratio `1 + r3t` at the stage level
`Kaa = N+1/3`.  It does **not** interpolate the free surface and then form one
ratio from it.  It first builds the *after-level* ratio arrays from `ssha`
(`GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:167`, the call
`dom_qco_r3c_RK3`), and only then interpolates the two endpoint ratios
(`:177`):

```
r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)   ! at N+1/3 (Kaa)
```

The whole block is `GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:160-179`;
GYRE resolves the `np_LIN`/`np_HYB` branch, which is the one quoted.  The two
orders are real-equivalent and not bit-equivalent: SSH-first rounds one
product of an interpolated operand, ratio-first rounds two products and their
sum, exactly as the Fortran does.  This is the same compiled statement the
ORCA2 lane cited from its own build at the same line range.

legoESM now transcribes that association in
`nemo_r3t_rk3_stage1_stretch` (`packages/ocean/legoesm/ocean/eos.py`), and the
stage-one tracer weight `_qt_13` in
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` calls it.
The other three tracer weights (`_qt_b`, `_qt_12`, `_qt_aa`) are untouched.

## What was taken, and what was deliberately not

| file in `a0b2f7a5d` | taken | why |
|---|---|---|
| `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` (9 lines) | **yes** | the call site of the shared statement |
| `packages/ocean/legoesm/ocean/eos.py` (34 lines) | **yes** | the shared transcription itself |
| `tests/ocean/unit/test_nemo_ws_tracer_rk3.py` (40 lines) | **yes** | its unit control |
| `docs/.../nemo_testcases_l2_gyre_phase3_round8_receipt.md` | no | the ORCA2 lane's edit of its own receipt text; this lane's copy is current |
| `scripts/.../orca2_l4/nemo_testcase_l4_orca2_round45_qco_rk_gate.py` | no | an ORCA2-lane gate; the file does not exist on this lane |
| `scripts/.../nemo_testcase_l2_gyre_decision43_gate.py` | no | the ORCA2 lane's version of the Decision-43 gate, with ORCA2 census rows |
| `tests/ocean/fidelity/test_nemo_testcase_l2_gyre_decision43_gate.py` | no | its test, same reason |
| `scripts/.../nemo_testcase_receipt_citation_gate.py` | no | the ORCA2 lane's citation-map edit; this lane adds only the entry its own receipt needs |

Each refused file is an ORCA2-lane artefact and lives on that branch.  Nothing
else in the tree moved: no default, no card, no deck, no threshold, no carried
state, no scheme selector.

## The certified GYRE ladder

## The from-rest year — WHY THIS ROUND IS HELD

The order of operations this round was: measure the year, check it against the
numbers ORCA2 round 102 registered, and only then re-pin.  It failed at the
check, so no pin moved.

A fresh seed-0 member ran the full 2,160 steps and wrote all 360 daily
snapshots (`nemo_testcase_l2_gyre_year_fromrest.py --member 0 --days 360
--snap-steps 6 --tag r202`, 1,474.6 s, member
`phase3/year_fromrest/lego_seed0_r202`).  The day-by-day gap against NEMO was
then scored with the committed scorer (`nemo_testcase_l2_gyre_year_owners.py
--day-gap`), on BOTH arms, same scorer, same NEMO restarts, same eight days.

| day | this lane, BEFORE | ORCA2 certified | agree | this lane, AFTER | ORCA2 merged | agree |
|---:|---:|---:|:--:|---:|---:|:--:|
| 30 | `2.3432510206121264e-06` | `2.3432510206121264e-06` | yes | `2.3432465132112266e-06` | `2.3432465132112266e-06` | **yes** |
| 60 | `1.4793247420315405e-05` | `1.4793247420315405e-05` | yes | `1.4793247973304582e-05` | `1.4793247973304582e-05` | **yes** |
| 90 | `1.6332637650962138e-05` | `1.6332637650962138e-05` | yes | `1.633271203963844e-05` | `1.633271203963844e-05` | **yes** |
| 120 | `1.0965898339728307e-04` | `1.0965898339728307e-04` | yes | `1.0965906837581848e-04` | `1.0965907352116351e-04` | **NO** |
| 180 | `6.115333823687429e-05` | `6.115333823687429e-05` | yes | `6.115335288161464e-05` | `6.115335539300055e-05` | **NO** |
| 240 | `6.581707093530567e-05` | `6.581707093530567e-05` | yes | `6.58170624837412e-05` | `6.58170609494473e-05` | **NO** |
| 300 | `5.466049869672802e-05` | `5.466049869672802e-05` | yes | `5.466050113289237e-05` | `5.466049845187051e-05` | **NO** |
| 360 | `5.407735418221895e-05` | `5.407735418221895e-05` | yes | `5.407736527246344e-05` | `5.4077419367442036e-05` | **NO** |

Read it as a diff, which is the whole point of measuring both arms with one
scorer:

* **Before the change the two lanes agree on all eight days to every printed
  digit.**  The pre-adoption GYRE year is the same number on both trees.
* **After the change they agree on days 30, 60 and 90 and part company from
  day 120.**  Day 30 even agrees at the file level: this round's `day030.npz`
  has SHA-256 `4e36c106403b495e95327213292f0d1655d605fca6b0a75c67cb17833f067cba`,
  which is exactly the digest ORCA2 round 102 registered for its merged
  member.  Day 240 (`a63befc30bf03b443e54a51a0dc0541a186a0d20106e3ba94133b1489f488534`)
  and day 360 (`dcb7bc46c8bc75bd215b4752c8145b9025d00da38a6fdac8c6babd3cd6074899`)
  do not match ORCA2's registered digests
  (`8b9cd60475626373d9a2fec0baa508f06c1f91aed4c3d7a24fc5d8b3f5878c8a`,
  `e3e0a068346c7866f0a318bb32141f2fb326cc05c158a1c95bc39b091f585e25`).
  The digest check and the rms check fail together on the same days, so this
  is not a scoring artefact.

The round order said, in one line, that any differing digit stops the round.
It differs, so the round stops: **no certified GYRE number and no certified
digest was re-pinned, and the production code does not carry the statement.**
The three rows the order asked to register are therefore NOT registered.

**What is and is not established.**  CONFIRMED: the transcription itself
reproduces ORCA2's first three registered days exactly, day 30 bit for bit —
the statement is the same statement on both lanes.  NOT ESTABLISHED, and NOT
investigated this round because the order forbids it: why the two trees part
company from day 120 when their pre-adoption years are identical.  The
obvious candidate is PLAUSIBLE only — the ORCA2 number comes from a MERGED
tree (ORCA2 round 100's merge) that carries lane content this tip does not,
and a difference that is inert on the old trajectory need not be inert on a
new one.  Run-to-run nondeterminism is the competing explanation and is not
excluded here, though rounds 199 and 201 both reproduced 360 of 360 snapshots
byte-identically on this host, which argues against it.  The discriminating
measurement is a second independent 360-day member on this tip with the
statement applied: identical numbers refute nondeterminism and leave the tree
difference; different numbers settle it the other way.  It costs about
25 minutes and is the first item of the next round.

For the record, the old pin does fail against the new year, which is the
non-vacuity the re-pin step would have needed: 344 of 360 daily snapshot
files differ from the certified carried arm
(`phase3/round202/gyre_year_byte_identity.txt`).  That number is reported as
a measurement, not as grounds for a pin.

## The other cards

## Gates

## Non-vacuity

## Independent review

## Choices made this round

## OPEN
