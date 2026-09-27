# Round 187 receipt — developed shortwave record reconciliation

**Status: HELD.**  The Round-186 shortwave record is 1,174,060 bytes because
its compiled `qsr` field has the 32-by-22 no-halo extent, not the 36-by-26
full local extent assumed by the original reader.  That closes the operator's
1,856-byte refusal exactly.  The preregistered size-only diagnosis is
**REFUTED**, however: after the size repair the fail-closed reader found a
second defect, expecting the step-1080 stage-3 RHS slot to be 3 when the
record and compiled slot rotation say 1.  Per the frozen falsifier, the
scientific `qsr_2BD` walk stopped.  No physics, configuration, carried state,
default, stabilizer, or certified trajectory changed.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round187.md`, commit
`b15a75b0e`.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round187/`.

## Compiled layout reconciliation

The exact compiled write is a 16-byte magic, nine default integers, then
`gdepw_1d`, `qsr`, `r3t(:,:,Kmm)`, `e3t_3d`, `tmask`, `wmask`, the actual
increment, and its post-call replay at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:925-931`.
The compiled surface-boundary module allocates `qsr` over
`Nis0:Nie0,Njs0:Nje0`, not `jpi,jpj`, at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/sbc_oce.f90:210-212`.
Those bounds remove `nn_hls=2` on each edge and therefore produce 32 by 22
values on the 36 by 26 grid, as established by
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/mppini.f90:1501-1507`.

The corrected byte equation is

```text
16 + 9*4 + 8*(31 + 32*22 + 36*26 + 5*36*26*31)
= 1,174,060 bytes
```

It equals the acquired file exactly.  The parser now preserves the recorded
32-by-22 field and exposes its one-based bounds `[3,34,3,24]`; it does not
invent halo values.  The acquisition script's frozen size is corrected to the
same value.  A new non-vacuity test appends the 232 missing halo values from
the old assumption and proves that the reader refuses the resulting file.

## Frozen prediction refuted by the header

The first real-record read passed the byte count, magic, and payload extent,
then stopped with:

```text
header is (1, 1080, 3, 2, 1, 36, 26, 31, 64),
expected (1, 1080, 3, 2, 3, 36, 26, 31, 64)
```

The record is right.  NEMO initializes `(Nbb,Nnn,Naa,Nrhs)=(1,2,3,3)` at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/nemogcm.f90:368`.
The three-stage program swaps `Nnn/Naa/Nrhs` after stages 1 and 2 and swaps
`Nbb/Naa/Nrhs` at the end of every step at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3.f90:200-222`.
Carrying that compiled rotation to even step 1080 gives stage-3 `Kmm=2` and
`Krhs=1`, exactly the stored header.  The reader and its direct regression
test now encode `(1,1080,3,2,1,36,26,31,64)`.

This is a second reader expectation defect, so preregistration prediction 1
(`only` the no-halo count was wrong) is **REFUTED**.  Its explicit falsifier
required stopping the walk on any header failure.  The corrected reader was
therefore not rerun against the real record in this round: replay equality and
the actual-increment ULP plant remain **UNMEASURED**, not silently inferred.

## Passivity evidence and scientific scope

Both acquired passive-control restarts are already byte-identical to the
uninstrumented year run:

- step 180:
  `853b3d41b2aa512e934430cc1fcbf36ea574c2148419d6c4b98a1e16db94cfc6`;
- step 1080:
  `6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976`.

This proves the writer did not change the trajectory, but it does not replace
the still-pending replay calibration.  No shortwave statement, operand, or
candidate is named in this receipt.

Because executable model code did not change, certified values remain:

- kt2 T/S/U/V: `1.4210854715202004e-14`,
  `2.1316282072803006e-14`, `8.326672684688674e-17`, and
  `9.714451465470120e-17`;
- kt3 T/S: `4.9403105251144552e-07` and `4.0085410546453204e-08`;
- day-30/day-240/day-360 T3D RMS: `2.3276772050683987e-06`,
  `6.5861718814795174e-05`, and `2.6709923853294689e-03 K`;
- first over bar: kt3.

GYRE, generic NEMO-GYRE, DINO, LOCK_EXCHANGE, OVERFLOW, and ORCA2 execute no
changed model statement.  ORCA2 remains **UNMEASURED-WITH-SPEC** for this
shortwave ownership claim: it needs its native developed operands, masks,
production trace, and endpoint projection.

## Tests, citations, and review

The focused reader tests report `4 passed`.  The direct citation gate and its
shifted-citation plant are run after this receipt is committed; their exact
results are appended below without changing any scientific claim.  The final
diff receives the required separate read-only Codex review; its verbatim
verdict is appended below.  No broad ocean battery is warranted because no
model implementation or production gate changed.

## OPEN — round 188

1. Run the corrected reader once against the existing passive record.  Admit
   it only if the header, payload, producer stamp, bit-exact replay, and
   actual-increment ULP plant all pass.  Do not rebuild or rerun NEMO.
2. If admitted, execute the Round-187 preregistered `qsr_2BD` walk using the
   recorded step-1080 operands.  Report isolated eager/JIT separately from
   production eager/JIT and name the first non-bit compiled statement or its
   first inherited operand.
3. Only a one-variable NEMO-source-exact production-JIT candidate may enter
   the complete Decision-43/45/55/59 ladder, month, year, card census, DINO,
   tank, generic-card, citation, plant, and dual-review gates.

No acquisition or configuration decision is requested.
