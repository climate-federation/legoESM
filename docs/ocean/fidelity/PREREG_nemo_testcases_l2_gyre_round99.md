# Preregistration: NEMO-testcases L2 GYRE round 99 direct stage-one W operands

Date: 2026-09-16. Frozen at incoming tip
`3e7a15c1e64e036e2e066dcfceaa663f25406f24` before parsing the Round-98
scientific payload or changing the shared implementation. Evidence belongs
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round99/`.

## Question, magnitude, and acquisition repair

Decision 41 keeps the walk at kt=1 stage 1. Round 98 corrected the W reference
boundary and found 5,289 cells still unequal at
`2.6469779601696886e-23 m s-1` after its source-ordered scalar replay, but it
could not distinguish an inherited stretch operand from the W recurrence
because the direct-intermediate record was not admitted. The magnitude targets
remain kt3 T `1.627497246303733e-4 K` and day-30 T RMS
`1.2397011295506804e-2 K`.

The operator completed the requested NEMO run. Its wrapper stopped only at an
incorrect byte-count assertion: the record exists at 884,028 bytes and the
recorded executable matches the built executable. The compiled source declares
both `hdiv` and local `ze3div` on
`Nis0-1:Nie0+1,Njs0-1:Nje0+1,jpk`, not on `jpi,jpj,jpk`
(`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/oce.f90:98-104` and
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/sshwzv.f90:258-260`). With
`jpi,jpj,jpk = 36,26,31` and `nn_hls = 2`, the compiled loop bounds set
`Nis0,Njs0 = 3,3` and `Nie0,Nje0 = 34,24`
(`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/mppini.f90:1501-1516`). Thus each
local divergence field is `34*24*31`, while the two `r3t` slots, `e3t_3d`, and
`pww` retain their compiled full extents. The exact registered size is
`16 + 9*4 + (2*34*24*31 + 2*36*26 + 2*36*26*31 + 1)*8 = 884028`
bytes, matching the compiled writes at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/sshwzv.f90:283-291,314-317`.

**Pre-measurement control retraction (2026-09-16):** the first committed
version incorrectly included one additional `36*26*31` term while printing the
same 884,028-byte result. The synthetic exact-EOF reader test rejected that
formula before the oracle payload was parsed. The corrected expression above
counts the two local divergence fields, two full 2-D `r3t` fields, two full
3-D fields (`e3t_3d` and `pww`), and one scalar. The failed arithmetic claim
remains visible here rather than being silently replaced.

The existing Round-98 acquisition script and the existing Decision-41 stage
gate will be extended in place. Admission mode must verify the recorded
`binary.sha256`, record size, STOP 0, source/toolchain manifests, record digest,
and restart/mesh identity, then create the missing stamp and admission outputs
without invoking `makenemo` or `mpirun`. Every possible nonzero exit prints a
named `REFUSE:` line first. The wrong-size plant and consumed-field admission
plant must both exit nonzero.

## Frozen direct-walk predictions and falsifiers

1. The repaired reader must consume physical EOF exactly. It must expose local
   `hdiv/e3div` as `(24,34,31)`, full `r3_kaa/r3_kbb` as `(26,36)`, full
   `e3t_0/ww` as `(26,36,31)`, and the scalar `r1_dt`. A truncated or one-value
   extended record must exit nonzero.
2. On the owned `22*32*30` wet cells, the direct `hdiv` and `e3div` fields are
   predicted bit-identical to the source-rounded Round-98 trace. Direct
   `r3_kbb`, `e3t_0`, and `r1_dt` are predicted bit-identical to the already
   admitted stage operands. Any unequal cell moves the first statement to that
   earlier boundary and refutes this prediction.
3. The first predicted non-bit boundary is the Round-98 reconstructed
   `r3_kaa`: the direct compiled value is predicted to differ because the
   former replay re-reduced the static column before multiplying SSH, whereas
   NEMO consumes its already materialized `r1_ht_0`. Replacing only that replay
   operand with the direct `r3_kaa` is predicted to make the source-ordered
   bottom-up recurrence and final W bit-identical. Exact reconstructed
   `r3_kaa`, or any remaining final-W unequal cell with all direct operands,
   refutes this prediction.
4. A one-ULP change to direct `r3_kaa` and a one-ULP change to an exact incoming
   carry must each flip their named row and exit nonzero. The existing wrong
   commit-stamp plant must fail before record consumption.

If the direct record instead names an earlier non-bit input, ownership remains
there and no downstream W patch is eligible. If the prediction is confirmed,
the only eligible W candidate is the one shared NEMO-identity implementation
of the compiled `r3t = ssh*r1_ht_0`, `e3t*hdiv`, and left-associated recurrence
statements; it may be composed only with Round 97's held same-stage full-RHS
member. The given-NEMO-entry kt=1 stage-1 U, V, W, and every affected output
must then be BIT before trajectory measurement.

## Frozen trajectory predictions and Rule-12 falsifiers

For an eligible same-stage composition, the frozen prediction is no headline
movement from the immutable Round-85 / Round-96--98 before arm: kt2 T/S
`1.4210854715202004e-14` / `2.1316282072803006e-14`, kt2 U/V
`2.7377110452773967e-12` / `3.284922138989399e-12`, kt3 T/S
`1.627497246303733e-4` / `6.327735185607253e-6`, and day-30 T RMS
`1.2397011295506804e-2 K`. Any movement refutes bitwise invariance and is
retained in the receipt.

Landing is refused if any of the 954 registered rows that was AT-BAR leaves
the bar, if first-over-bar moves earlier than kt2 U/V, or if any moved row is
absent from the full Rule-12 table. The 30-day member is required for a locally
exact trajectory candidate even if Rule 12 rejects it. A failed candidate is
restored and retained only as a held manifest patch.

## Testcase dispositions

| lane | frozen disposition |
|---|---|
| GYRE stage twin | Admit direct fields; require exact EOF, binary identity, red layout/admission/ULP/stamp plants, and BIT U/V/W for any candidate |
| GYRE kt=1--10 | Run only for a locally exact same-stage composition; compare all 954 rows with the immutable before arm |
| GYRE days 1--30 | Run a fresh member for any locally exact trajectory candidate and score every day against `year_owners` |
| LOCK_EXCHANGE-zco | The shared WS-RK3/W statements execute; run focused shared-path and tank gates only for an eligible candidate |
| OVERFLOW-zps | Preserve partial-cell construction and run focused shared-path and tank gates only for an eligible candidate |
| DINO | **SHARED-STATEMENT RISK:** the shared QCO W helper executes; no neutrality claim, and the 96--98% regional-cancellation warning remains explicit |
| ORCA2 | **UNMEASURED-WITH-SPEC:** prove the resolved integrator, then record native stage-entry transports, direct W intermediates/carries, stage outputs, histories, and closure state with red plants |

No production configuration, carried-state policy, coefficient, timestep,
stabilizer, year harness, reconciliation gate, freshwater pair, #1484 guard,
NEMO source, or NEMO executable may change.

## Pre-measurement addendum: the compiled HYB ratio interpolation

Frozen after the direct record confirmed the first non-bit input at
`r3t(Kaa)`, but before replaying the following newly read statement or changing
production. The initial prediction that a re-reduced static column owned that
row is **REFUTED**: rebuilding NEMO's stored reciprocal expression leaves the
same 201 unequal cells. Reading forward in the compiled stage program exposes
an earlier association the initial walk omitted.

GYRE resolves the compiled `n_baro_upd` default to `np_HYB`. Stage 1 first
saves the full external-mode `ssha`, independently interpolates `ssh(Kaa)` as
`r2_3*ssh(Kbb) + r1_3*ssha`, forms `r3ta = ssha*r1_ht_0`, and then independently
interpolates `r3t(Kaa) = r2_3*r3t(Kbb) + r1_3*r3ta`
(`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:150-190` and
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/domqco.f90:256-258`). Multiplying
the already-interpolated `ssh(Kaa)` by a reciprocal is algebraically equal but
not the statement NEMO executes.

The existing admitted full external SSH, Kbb `r3t`, and static geometry are
sufficient for a registered replay. The frozen prediction is that this
source-ordered HYB ratio interpolation makes direct `r3t(Kaa)` BIT, after
which every direct W recurrence row and final W remain BIT. Any unequal HYB
ratio cell refutes that prediction and triggers the same-call operand
acquisition. If it is BIT, the only candidate remains the same-stage W
materialization plus Round 97's full-RHS member; it must make given-entry
stage-1 U, V, and W BIT before the already-frozen Rule-12 and 30-day tests.

## Pre-measurement addendum: the stage-local W clock

Frozen after candidate `cdddd17dce3a` was measured, and before changing or
measuring the stage clock. That first candidate is **REFUTED** and remains in
the receipt: it made given-entry kt=1 stage-1 U/V BIT, but W stayed unequal in
18,000 wet cells at `1.318522148478393e-7 m s-1`; zFw, T, and S consequently
remained DEBT. Its direct operand replay nevertheless made the compiled HYB
ratio, every W recurrence boundary, and final W BIT. The shared call therefore
still differed from the proven direct program.

The first differing shared-call statement is now the clock. The card's full
step is 14,400 s, while the admitted direct record carries `rDt = 4,800 s`.
The compiled stage program sets `rDt = r1_3*rn_Dt` and then
`r1_Dt = 1/rDt` before stage 1
(`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:140-150`). The
vector-invariant tracer path subsequently calls the same transport-form W
routine (`.../traadv.f90:266-280`), whose compiled recurrence consumes that
module `r1_Dt` (`.../sshwzv.f90:305-311`). legoESM's public path instead passes
the 14,400-s full-step clock unless a private diagnostic arm is selected.

The newly frozen candidate removes that model-only choice: the one shared
stage program always passes `(rn_Dt/3, rn_Dt/2, rn_Dt)` at stages 1--3, exactly
where the compiled program assigns those values. Prediction: given-NEMO-entry
kt=1 stage-1 W and zFw become BIT; the already-BIT U/V remain BIT; T/S become
BIT because they consume that exact transport. Any unequal required stage-1
output refutes the candidate. Only if the complete stage row closes may the
unchanged 954-row Rule-12 and days 1--30 falsifiers run.

This is explicitly a cancelling-pair candidate, not a revival of the
Round-21 single-clock arm. Round 21 measured stage-local clock with the old
full-step `r3_after-r3_before` and correctly found it DEBT; production's
full-step delta/full-step clock pair was AT-BAR. Candidate `cdddd17dce3a`
replaced the first half with NEMO's independently interpolated stage-local
`r3t(Kaa)-r3t(Kbb)` while retaining the full-step clock, and the stage table
proved that opposite half-state DEBT. NEMO executes neither half-state: it
pairs the stage-local ratio delta at `stprk3_stg.f90:175-193` with the
stage-local reciprocal clock assigned at `:147-148`. The frozen prediction
above therefore applies only to that two-statement pair; either half by itself
remains refuted evidence.
