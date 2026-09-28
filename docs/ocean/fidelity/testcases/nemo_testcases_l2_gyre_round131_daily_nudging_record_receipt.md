# NEMO testcase L2 GYRE phase 3 — round 131 daily-reset record receipt

Date: 2026-09-20

Incoming tip: `404e9fc626199f61e16e1150ccc08e9501240c1b`

Status: **STOPPED_FOR_RECORD — the operator-named daily NEMO root contains
only days 1–30, so 330 of the 360 required daily boundaries are absent and no
360-day reset arm can run without inventing oracle state.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round131/`

## Outcome first

The prerequisite for the requested four-family attribution is not present.
The committed admission gate ran from clean commit
`de9a2695de120890c5c4f2fcecce32cd3969b4d5`, inspected every available daily
restart, and stopped before one legoESM model step:

| record fact | required | observed | verdict |
|---|---:|---:|---|
| completed daily boundaries | 360 | 30 | **REFUTED** |
| step range | `6..2160` by 6 | `6..180` by 6 | days 31–360 absent |
| required family variables per existing file | 16 | 16 | exact schema present |
| files with schema checked | 360 | 30 | all available files clean |
| 30-day overlaps with the year control | 12 | 1 | only day 30 available |
| run end step | at least 2160 | `nn_itend=180` | source run ended day 30 |
| restart cadence | 6 steps | `nn_stock=6` | daily while the run existed |

The single overlap is stronger than a tolerance comparison: both day-30
restart files have SHA-256
`853b3d41b2aa512e934430cc1fcbf36ea574c2148419d6c4b98a1e16db94cfc6`.
Every required array digest is also identical. The daily record is therefore
the certified year trajectory through day 30; it simply does not continue.

The separate NEMO year root has restarts at days 30, 60, ..., 360. Those 12
monthly boundaries are valid scoring controls, but they cannot supply the 348
intervening six-step boundaries required by daily resets. Reusing or
interpolating them would change the experiment and is refused. The requested
three-step follow-up is also unsupported: even the daily root has no `kt=3`
boundary.

## Compiled-source record contract

The source card that produced the named daily record is the compiled
`GYRE_OMIP_L2_P3_SM_R41ADVSP` branch. At the end of an RK3 step NEMO swaps the
completed `Naa` state into `Nbb` and forms the next-step extrapolated SSH at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90:222-226`; after the
diagnostics it calls the restart writer with those resolved levels at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90:249-260`.

That compiled writer stores `sshn`, `un`, `vn`, `tn`, `sn`, the two depth-mean
vectors, and the RK3 `ssha` slot at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/restart.f90:176-184`. The active
time-split external-mode branch separately writes all six carried histories
at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynspg_ts.f90:974-980`, and the
active TKE branch writes `en`, `avt_k`, `avm_k`, and `dissl` at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdftke.f90:895-901`.

Those compiled statements justify the 16-field family schema; they do not
justify synthesizing a boundary NEMO did not write. The audit's file inventory
is therefore the deciding evidence for this round.

## Frozen prediction ledger

* **P0 CONFIRMED.** The preregistered formal prediction was 30 present and 330
  missing boundaries, last step 180, `nn_itend=180`, and a named nonzero stop.
  The audit measured exactly those values and returned exit 4 with
  `STATUS STOPPED-FOR-RECORD`.
* **P1 NOT EXECUTED BY PREREGISTRATION.** Record admission failed, so none of
  the T/S, vector/history, SSH/history, or TKE reset arms started. There is no
  owner-family ranking and no western-upper-100-m birthplace claim.
* **P2 CONFIRMED.** Production, cards, carried state, NEMO source, and the
  certified year harness are unchanged. No physics or configuration decision
  is hidden in this stop.
* The conditional forecast that the vector family would remove the most
  day-240 growth is **UNMEASURED**, not confirmed or refuted.

## Requested attribution table

The table is deliberately a readiness table rather than invented score rows.
The immutable free arm remains
`phase3/round130/arms/r109_handoff/lego_seed0_year`, whose day-240 T3D RMS is
`1.64467402331753935e-2 K`.

| family | required daily fields | 360-day arm | day-240 removal | disposition |
|---|---|---|---:|---|
| T/S | `tn`, `sn` | not run | UNMEASURED | missing days 31–360 |
| vector + histories | `un`, `vn`, `ub_e`, `vb_e`, `ubb_e`, `vbb_e` | not run | UNMEASURED | missing days 31–360 |
| SSH + histories | `sshn`, `ssha`, `sshb_e`, `sshbb_e` | not run | UNMEASURED | missing days 31–360 |
| TKE | `en`, `avm_k`, `avt_k`, `dissl` | not run | UNMEASURED | missing days 31–360 |

No kt ladder, month, day-240, day-360, DINO, tank, or ORCA2 row can move
because no candidate implementation or trajectory was produced. ORCA2
remains UNMEASURED-WITH-SPEC from the preceding receipts.

## Mechanical controls and evidence

The audit JSON is
`phase3/round131/daily_record_audit.json` (SHA-256
`b51699af1b990acc9780cb9eb54399d4c96d420ef624b2546dadd4bdd944c669`).
It records the clean producer commit, all observed and missing steps, the
namelist values, per-file schema checks, and per-variable overlap hashes.

The gate first passed a complete virtual 360-boundary, 16-field control. Its
two plants then started from that complete control, so the already incomplete
real record cannot make them vacuously succeed. Both exited 1:

```text
STATUS PLANT-FIRED: missing-boundary
STATUS PLANT-FIRED: required-variable
```

The missing-boundary plant removes only `kt=2160`; the required-variable plant
removes only `dissl`. Their logs and the real exit-4 audit log are retained in
the evidence root.

## Independent adversarial review

The required read-only Codex pass was invoked against the two Round-131
commits, the audit JSON, both NEMO roots, the field registry, the controls, and
the stop verdict. It could not initialize in the sandbox and produced no
verdict. Its verbatim terminal output was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Thus, **independent review unavailable in-sandbox**. The complete invocation
is retained as `phase3/round131/codex_review.log`; there is no `DO NOT SHIP`
verdict to override.

## Verification

All commands used the required CPU/fp64 environment.

* The direct record-gate and receipt-citation suites reported exactly
  `23 passed in 4.00s`.
* The final unplanted citation run found five citations, zero unmapped
  citations, zero failures, an empty whole-map audit, all nine internal
  controls firing, and `status: PASS`.
* Shifting the registered completed-step citation by two lines returned exit
  1 with `status: FAIL` and `SYMBOL-NOT-AT-LINE`; the whole-map audit remained
  empty. The artifacts are `phase3/round131/citation_gate.json` and
  `phase3/round131/citation_gate_shifted_plant.json`.
* The first citation attempt correctly refused an ambiguous external-history
  guard because the read and write arms share its text. The map now pins the
  second occurrence; neither cited line nor extent was weakened.

No full model or all-tree pytest battery ran: record admission stopped before
a model step, and this round changes no model, card, harness, or NEMO source.
The focused gate, source-citation, and fail-closed controls cover the complete
committed change.

## OPEN — next round

1. Locate or provide a certified NEMO seed-0 restart series at every
   `kt=6,12,...,2160` on the same trajectory. It must include all 16 registered
   fields, retain the compiled-source/binary provenance, and reproduce all 12
   existing monthly checkpoints bit-for-bit.
2. Run the committed Round-131 admission gate first. Do not start a reset arm
   unless it reports `STATUS ADMITTED` from a clean stamped commit.
3. Once admitted, run all four independent 360-day daily-reset arms and score
   every field at days 30/60/90/120/180/240/300/360 against the immutable
   Round-130 free arm. Rank only measured day-240 T3D removal.
4. Run the three-step cadence only if matching NEMO boundaries exist at every
   `kt=3,6,...,2160`; daily data are not a substitute.
5. If time remains after all four arms, re-score `r99_wclock` alone as the
   deferred Round-130 OPEN item. It must not displace the owner-family
   experiment.

No NEMO acquisition script is requested in this round because the binding
Round-131 order explicitly forbids an acquisition. Production remains at the
incoming landed tip.
