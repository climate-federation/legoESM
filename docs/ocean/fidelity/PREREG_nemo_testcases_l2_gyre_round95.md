# Preregistration: NEMO-testcases L2 GYRE round 95 stage-twin completion

Date: 2026-09-14. Frozen at incoming tip `5972a8016597` before any Round-95
scientific comparison. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round95/`.

## Question and existing instrument

Decision 41 still requires completion of the one Round-46/51 stage-twin gate
before an operator owner may be named. The operator ran the Round-94
acquisition: NEMO compiled and ran successfully, and the requested kt=2
stage-3 transport record exists. The acquisition then failed closed during
admission because the new basename had not been registered in the central
time-level registry. Round 95 repairs that omitted registration, admits the
existing record without rerunning NEMO, and extends the same gate. It does not
create a second harness and it does not propose a production-physics landing.

The compiled acquired program calls `tra_adv_trp` at
`GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/stprk3_stg.f90:792-803` and
writes the kt=2 stage-3 `zFu/zFv/zFw` immediately afterward at
`GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/stprk3_stg.f90:809-834`.
The compiled driver computes vertical physics before the external solve and
all three RK stages at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:155-215`; the stage
record reads `en`, `avm_k`, `avt_k`, and `dissl` from the live stage boundary
at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:431-433` and
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/l2_r46_stage.f90:106-147`.

The pre-implementation search found the existing time-level registry, the
existing acquired-record reader/admission gate, and the existing production-
JIT live-stage trace. They are extended in place. No reader, stage harness, or
TKE numerical method is duplicated.

## Frozen measurement contract

1. Register only `oracle_tracer_transport_kt00000002_s3.bin` as NOW, citing
   the acquired compiled call and write sites. The registry test must prove the
   new name resolves and a shifted-citation plant must fail later in the
   receipt gate.
2. Run the existing admission gate against Round 75 and the acquired Round-94
   directory. The restart and mesh remain byte-identical; only the named
   transport record is new. Its consumed-field plant must exit nonzero.
3. Add the admitted record to the existing stage-twin transport map. All six
   kt=2 stage-3 transport rows across the given-entry and chained tables move
   from `UNMEASURED_WITH_SPEC` to a measured class.
4. Extend the existing WRITE-only production-JIT trace to expose the actual
   TKE energy operand selected inside the closure path. The trace must surface
   the value produced at the consuming site, including the cold-start seed;
   the gate may not reconstruct it and may not inject a model state merely to
   make the row measurable. The three chained kt=1 `tke_en` rows then score
   that traced value against NEMO.
5. Every previously measured row must reproduce Round 94 exactly. Any changed
   overlap is retained as a refutation and blocks an ownership verdict.

## Frozen predictions and falsifiers

1. Acquisition admission is predicted to pass after the cited registration,
   with restart and mesh identical and exactly one allowed new record. A bad
   header, wrong kt/stage, changed baseline file, extra new file, or green
   consumed-field plant refutes admission.
2. The kt=2 stage-3 record is predicted to contain finite binary64 `zFu/zFv/
   zFw` at the registered NOW boundary. All six paired-table rows become
   measured. Any missing or malformed payload keeps the gate UNMEASURED.
3. The actual traced kt=1 cold-start TKE operand is predicted bit-identical to
   NEMO `en` in all three unchanged stage-boundary rows. Any unequal cell is
   retained as DEBT, not hidden behind the former `None` sentinel.
4. After those nine rows close, `missing_required_rows` is predicted empty and
   the gate may name the first owned non-bit row. The expected first measured
   row remains kt=1 stage-1 U, AT-BAR but non-bit at
   `5.421010862427522e-20`; the larger same-stage W debt remains
   `3.5937485546815465e-8`. Any earlier external debt, missing row, or changed
   overlap refutes that ordering and ownership remains withheld.
5. Instrumentation is WRITE-only. The certified trajectory remains kt2
   T/S/U/V `1.4210854715202004e-14`, `2.1316282072803006e-14`,
   `2.7377110452773967e-12`, `3.284922138989399e-12`; kt3 T/S
   `1.627497246303733e-4`, `6.327735185607253e-6`; and day-30 T RMS
   `1.2397011295506804e-2 K`. Any changed value refutes non-interference.

## Rule-12 and testcase dispositions

| lane | frozen disposition |
|---|---|
| GYRE stage twin | Complete the one existing kt=1/2 paired gate; missing rows or a failed overlap check withhold ownership |
| GYRE kt=1..10 and days 1..30 | No production candidate; preserve the immutable Round-85 comparator and rerun after instrumentation to prove non-interference |
| LOCK_EXCHANGE-zco and OVERFLOW-zps | WRITE-only hooks remain private and off; run focused tank tests and make no tank-physics claim |
| DINO | Shared WS-RK3 trace-shape risk is explicit; no numerical-neutrality or regional-cancellation claim |
| ORCA2 | **UNMEASURED-WITH-SPEC:** record native stage entries, closure carries, transports, histories, and next-stage outputs; require the paired bitwise tables and red plants; reject missing rows, AT-BAR loss, or earlier first-over-bar |

No production configuration, coefficient, timestep, stabilizer, carried-state
policy, year harness, reconciliation gate, freshwater pair, #1484 guard, held
manifest, NEMO source, NEMO executable, or physical operator may change.
