# NEMO testcase L2 GYRE round 153 — developed FCT acquisition

Date: 2026-09-22

Status: **STOPPED_FOR_RECORD**.  No production physics, configuration,
carried state, restart schema, or acceptance gate changed.  Round 152's first
active process boundary is still stage-3 FCT advection, with an isolated
one-step temperature discrepancy of `1.0974591404090626e-8 K` RMS at NEMO's
day-180 entry.  This round preregistered and built the passive NEMO acquisition
needed to locate the first internal non-bit statement.  The NEMO run is not
attempted in the sandbox because the campaign's PMIx socket refusal is
binding.

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round153.md`, committed at
`3834f2b58`; its corrected full incoming-tip stamp is committed at
`ee327b9ce`.  The acquisition implementation is commit `50960ea48`, the
compiled-source citation registry is `af5b7e987`, and the finite-payload
admission check is `29104f58f`.  Preflight evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round153/acquisition_card/`.

## Compiled program and record boundary

The exact source card is the admitted Round-148 build
`GYRE_OMIP_L2_P3_SM_R148LDF`; its `nemo.exe` and admitted run binary are
byte-identical with SHA-256
`9d758bf51d85a27b7692858697ddbc5fbd9b19d724f957c983a605085c89e250`.
Its resolved run ends at step 1081 with `nn_stock=180`, `nn_write=2160`, and
seed zero.  The new target is `GYRE_OMIP_L2_P3_SM_R153FCTD`; the script clones
`GYRE_PISCES`, copies the R148 `EXP00`, `MY_SRC`, and cpp card file by file,
then applies only additive source patches.

At
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:164-199`, the
compiled parent calls the two-step upstream operator before forming the
high-order-minus-upstream faces.  The upstream routine writes its first faces,
first divergence, midpoint, averaged faces, second divergence, and direct
`Krhs` accumulation at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:495-610`.
The parent exchanges the anti-diffusive faces, invokes `nonosc`, forms the
final divergence, divides by the live Kmm thickness, and adds to `Krhs` at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:306-330`.
Within `nonosc`, NEMO forms the beta operands at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:849-883` and applies
the sign-selected U/V/W coefficients at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:886-936`.

Those are the compiled statements being observed.  There is no ownership
claim yet: the first non-bit internal row is **UNMEASURED** until the new
record is admitted and compared through legoESM's production step.

## Frozen record and passivity contract

The writer activates only at `kt=1081`, `Kbb=1`, `Kmm=2`, `Kaa=3`,
`Krhs=3`, for the ordered T/S tracer pair.  It refuses tiling and non-fp64
execution.  The self-describing stream has magic `NEMO_L2_R153FCT`, a fixed
14-integer header, and exactly 61 fields.  Each field carries its name, rank,
extent, origin, and binary64 payload; the parser derives byte size from these
descriptors.

| group | fields | count |
|---|---|---:|
| common | `p2dt`; U/V/W transports; `e3t_3d`; Kbb/Kmm/Kaa `r3t`; T/W masks; `r1_e1e2t` | 11 |
| each tracer | Kbb/Kmm tracer and entry `Krhs`; first U/V/W faces, divergence, midpoint; averaged U/V/W faces, second divergence and upstream `Krhs`; pre-limiter U/V/W faces; actual U/V/W coefficients; post-limiter U/V/W faces; final divergence, divisor and final `Krhs` | 25 |
| total | 11 common + 25 T + 25 S | 61 |

The coefficient arrays begin as exact binary64 one and are filled immediately
after each compiled `zcoef` assignment.  Thus an exact word different from one
is a direct activity observation, not a flux-ratio inference.  Capture calls
occur after the production statements and no captured value is read back.

Passivity is fail-closed against the entire R148 source-run registry.  The
script derives the complete set of every `oracle*.bin`, restart, and mesh file
from that run (1,191 files at preflight), requires the target set to be
identical after excluding only the new record, and compares every inherited
file byte-for-byte.  A single changed byte exits with `REFUSE`; there is no
optimizer/materialisation waiver.  The admission also verifies the record
stamp, clean producing commit, source-card manifest, run binary, exact schema
and EOF, finite payloads, binary masks, `p2dt=14400`, and nonzero directly
recorded limiter activity for both tracers.

The operator entry point is:

```text
/tmp/autopilot-work-pYMFSn/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round153_developed_fct/run.sh
```

Run it with `--run`.  If NEMO completes but admission is interrupted, rerun
the unchanged binary with `--admit-existing`; that mode checks binary and
source ancestry before admitting.  A refused or perturbing writer requires a
new target name.

## Predictions and disposition

1. **PENDING RECORD:** every inherited R148 file remains bit-identical.  Any
   changed byte refutes passivity and refuses the record.
2. **PENDING RECORD:** both T and S have at least one coefficient different
   from exact one.  A zero count refutes the activity prediction and stops the
   comparison.
3. **CONFIRMED LOCALLY:** the source patch removes zero lines, applies with
   fuzz zero, and the preprocessed writer plus modified `traadv_fct` compile
   with `gfortran -fsyntax-only` against the R148 build modules.
4. **CONFIRMED LOCALLY:** the record-layout plant deletes the final capture,
   prints `STATUS PLANT-FIRED: layout`, and exits 69.
5. **PENDING RECORD:** the stamp, truncation, missing-field,
   all-coefficients-one, and inherited-byte plants must each print
   `STATUS PLANT-FIRED` and exit nonzero on the acquired record.  Their
   synthetic unit tests pass; this is not substituted for record admission.

No prediction was relabeled after measurement.  Items 1, 2, and 5 remain
explicitly pending because NEMO has not run.

## Campaign surfaces

No candidate exists, so there is no Rule-12 or Decision-43/45 landing table
and no trajectory row moved.  The current values are inherited, not measured
in Round 153:

| headline | inherited value | disposition |
|---|---:|---|
| kt2 T max abs | `1.4210854715202004e-14 K` | AT-BAR |
| kt2 S max abs | `2.1316282072803006e-14 g/kg` | AT-BAR |
| kt2 U RMS | `2.7377110452773967e-12 m/s` | first-over-bar |
| kt2 V RMS | `3.2849219221489645e-12 m/s` | first-over-bar |
| kt3 T RMS | `8.659371033559182e-7 K` | DEBT |
| kt3 S RMS | `7.027288972949464e-8 g/kg` | DEBT |
| day-30 T3D RMS | `6.888193513796918e-5 K` | DEBT |
| day-240 T3D RMS | `1.644671864406711e-2 K` | DEBT |
| day-360 T3D RMS | `1.1223450861560211e-2 K` | DEBT |

Only a private NEMO writer and its admission tooling changed.  No shared
statement executes differently in GYRE, DINO, LOCK_EXCHANGE, OVERFLOW, or the
generic recipe.  ORCA2 is **UNMEASURED-WITH-SPEC**: make the same developed
entry, active-branch FCT record and production-step comparison on the
ocean-only ORCA2 card before transferring a statement verdict.

## Controls, review, and verification

The focused parser suite reports `12 passed in 1.28s`.  It covers the prior
Round-111 reader plus Round 153's complete synthetic record, all five
admission plants, and a source-registry addition.  The preflight log contains:

```text
SYNTAX_PROOF_PASS l2_r153_fct.f90 traadv_fct.f90
ROUND153_DEVELOPED_FCT_PREFLIGHT_READY
```

The layout plant log contains `STATUS PLANT-FIRED: layout` and its recorded
exit is 69.  Evidence files are `parser_tests.log`, `preflight.log`,
`layout_plant.log`, and `layout_plant.status` under the evidence directory
named above.

The required read-only Codex review was attempted after the implementation.
It did not start a reviewer and emitted verbatim:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Its exit was 1.  Independent review was unavailable in-sandbox; no `SHIP` or
`DO NOT SHIP` verdict is fabricated.  This round lands no physics.

At clean receipt commit `175c8edfc`, the citation gate reports `PASS`: five
citations found, zero unmapped citations, zero citation failures, and zero
failing map entries.  Shifting the compiled parent range by two lines reports
`FAIL` and exits 1.  The final combined focused suite reports `28 passed in
2.98s`; it includes both FCT record readers, every Round-153 plant, and the
citation gate's own controls.  No full ocean tree was launched because this
round adds an unexecuted private NEMO acquisition and no production Python
path or physics.

## OPEN — round 154

First run the operator acquisition above and require
`ROUND153_DEVELOPED_FCT_READY`.  Do not use a record unless the ordinary gate
passes and all five record plants exit nonzero with their named markers.

Then extend the existing Round-111/112 production-step walk rather than
creating an isolated replacement closure.  Load NEMO's exact day-180 entry and
compare all 61 fields in compiled order under production JIT, production
eager, and isolated JIT.  The production-path ULP plant must flip the affected
row.  Report, for T and S, cells unequal and max absolute difference at each
row plus overlap with the directly recorded active coefficients.  The first
production-JIT non-bit row owns the next compiled statement.  Only a
source-exact candidate may proceed to the full ladder, month, year,
shared-card, DINO, and ORCA2-spec gates.
