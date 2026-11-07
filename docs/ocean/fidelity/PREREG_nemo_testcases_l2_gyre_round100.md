# Preregistration: NEMO-testcases L2 GYRE round 100 production-JIT stage-one discriminator

Date: 2026-09-16. Frozen at incoming production tip
`f3289962e6c43cb22e6cc90dfc3ef8e7fd1d1649` before parsing the Round-99
scientific payload, applying a held patch, or running a new numerical
comparison. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round100/`.

## Question and magnitude

The campaign remains at the first owned production boundary, kt=1 stage-1 W.
Round 99 made all 17 stage outputs bit-identical given NEMO's entry by pairing
the compiled full momentum source, independently interpolated HYB ratio,
stage-local clock, and W recurrence, but Rule 12 rejected that same-stage
candidate: 85/954 rows moved and 56 rows contained 131,713 cellwise
violations. The headline magnitude targets remain kt3 T
`1.627497246303733e-4 K` and day-30 T RMS
`1.2397011295506804e-2 K`; local W exactness is not assumed to own either.

Operator note L asks whether an eager-only local proof was promoted even
though production executes the statement inside `self._step_jitted`. The
registered discriminator is the held Round-89 vector RK3 assignment because
its isolated proof was zero-unequal while its held bundle was the known
large-damage member. This round labels three distinct executions and never
calls an isolated `jax.jit` result a production result.

## Compiled program and admitted record

The executing vector statement forms `Kaa = (Kbb + rDt*Krhs)*mask` at
`GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/stprk3_stg.f90:666-684`.
The stage program sets `rDt=rn_Dt/3`, saves the full-step SSH, independently
interpolates stage SSH, calls the QCO ratio builder on the full-step SSH, and
then independently interpolates `r3t(Kaa)` at
`GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/stprk3_stg.f90:140-193`.
The transport-form W call consumes those stage slots at
`GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/traadv.f90:266-280`; its recurrence
uses `r1_Dt*e3t_3d*(r3t(Kaa)-r3t(Kbb))` at
`GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/sshwzv.f90:298-311`.

The operator-produced Round-99 record is admitted without rebuilding or
rerunning NEMO. Its record stamp names producer `f3289962e6c4`, record SHA-256
`eb8264eb41d6b01c709d8dd4d62ccf14433480677f65b7240ceb1c9cd75a758c`,
and executable SHA-256
`5911e1386ee683d0627850da22496bbf88e8f509cc488b3819212d05f6438ce2`.
The admission artifact reports PASS, 197 admitted differences, and zero
violations. Scientific fields remain unparsed at this freeze point.

The pre-implementation search found the existing Decision-41 stage twin,
Round-89 assignment walk, live production-step stage operands/raw outputs,
Round-99 fail-closed reader, and held Round-99 candidate. This round extends
those instruments; it adds no second stage harness or numerical implementation.

## Frozen execution discriminator

For each U/V assignment row exercised on recorded NEMO inputs, report:

1. **isolated-closure eager** — the shared assignment helper called eagerly;
2. **isolated-closure JIT** — only that small helper closure under `jax.jit`;
3. **production step** — the raw stage assignment returned by the existing
   live-stage trace from the full public `model.step` / `self._step_jitted`,
   driven from the recorded stage entry. Score it both against NEMO's raw
   output and against a scalar replay from the operands captured inside that
   same production step, so upstream operand debt cannot be mistaken for a
   fusion error.

The held Round-89 source-materialized assignment is predicted bit-identical in
the isolated eager and isolated-JIT modes: zero unequal U and V cells. The
production-step transcription against its own captured Kbb/Krhs is also
predicted bit-identical. Against NEMO's raw stage-1 output, restored production
is predicted to retain the known upstream full-RHS residual (formerly 7,620 U
and 8,460 V cells at `5.421010862427522e-20`) unless the held Round-99
full-RHS member is present. This latter row is an operand result, not a fusion
result.

The production-fusion hypothesis is **CONFIRMED for this member** only if an
isolated row is BIT while the production-step transcription from the same
captured operands is non-BIT. It is **REFUTED for this member** if all three
transcription rows are BIT; an upstream production-vs-NEMO row does not count
as confirmation. A one-ULP change to the production-step assignment oracle
must make its named row non-BIT and the plant process exit nonzero. The
commit-stamp plant must fail before record consumption.

If the discriminator fires, production-step JIT becomes binding for every
subsequent local proof and stage row; eager and isolated-JIT rows remain
diagnostics only. If it does not fire, this round does not generalize that
negative result to the separately observed TKE K_H closure.

## Frozen same-call ratio and held-candidate checks

The admitted Round-99 record writes `ssh(Kaa)`, stored `r1_ht_0`, live
`r3t(Kaa)`, `ssh(Kbb)`, `r3t(Kbb)`, and `ht_0` from the actual W call at
`GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/sshwzv.f90:278-293`. The reader
must verify the record/producer stamp and physical EOF before exposing fields;
a shifted producer stamp must exit nonzero.

Frozen predictions:

1. `ssh(Kbb)*r1_ht_0` is BIT with recorded `r3t(Kbb)` on wet owned cells.
2. `ssh(Kaa)*r1_ht_0` is predicted non-BIT with recorded `r3t(Kaa)` in the
   previously identified 201/600 wet cells, maximum
   `2.6469779601696886e-23`; a BIT result refutes the Round-99 association
   diagnosis.
3. The source-ordered HYB interpolation of the separately formed full-step
   ratio is predicted BIT with recorded `r3t(Kaa)`. Any unequal wet cell moves
   ownership back to an earlier ratio operand/statement.
4. Reapplying the held Round-99 ratio/clock/full-RHS patch and driving the
   existing stage twin through the full compiled production step is predicted
   to reproduce all 17 kt1-stage1 output rows BIT. Its raw assignment and W
   rows must also be BIT in the new production-labelled discriminator. Any
   unequal required row refutes the former local proof and forbids a ladder.

The Round-99 candidate already has a canonical Rule-12 failure. This round
does not reinterpret that failure as a JIT failure unless the registered
production discriminator names one. If no narrower same-stage candidate
emerges, production is restored and the prior trajectory figures are retained:
kt2 T/S `1.4210854715202004e-14` / `2.1316282072803006e-14`, kt2 U/V
`2.7377110452773967e-12` / `3.284922138989399e-12`, kt3 T/S
`1.627497246303733e-4` / `6.327735185607253e-6`, and candidate day-30 T RMS
`1.2397011291846179e-2 K`. Any newly eligible numerical candidate must run the
full 954-row ladder and a fresh days 1--30 member; no recorded candidate is
silently promoted.

## Rule-12 and testcase dispositions

| lane | frozen disposition |
|---|---|
| GYRE stage twin | Three labelled assignment modes, same-call ratio rows, production-step plant, and all required kt1-stage1 outputs; missing rows fail closed |
| GYRE kt=1--10 | Run only for a newly eligible production-JIT-exact same-stage candidate; compare all 954 rows to the immutable Round-96/97 before arm |
| GYRE days 1--30 | Fresh member only for such a candidate; otherwise retain the measured Round-99 held result without a new claim |
| LOCK_EXCHANGE-zco | Run focused shared-path/tank checks only if shared production numerics change |
| OVERFLOW-zps | Same, retaining partial-cell construction |
| DINO | **SHARED-STATEMENT RISK:** the shared WS-RK3/W arithmetic executes; no neutrality claim, and 96--98% regional cancellation remains explicit |
| ORCA2 | **UNMEASURED-WITH-SPEC:** resolve its integrator, then record native production-JIT stage entries, raw assignments, W operands/carries, outputs, histories, and red plants |

No production configuration, carried-state policy, coefficient, timestep,
stabilizer, year harness, reconciliation gate, freshwater pair, #1484 guard,
NEMO source, or NEMO executable may change.
