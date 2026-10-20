# Vertical-mixing lane — V1 round 2 acquisition repair (vmix_r2)

Date: 2026-10-16. Base: `d8048d44f`. Instrument repair commit:
`01c79310a`. Status: **STOPPED_FOR_RECORD**. The operator's completed two-step
smoke is admitted in place; neither the writer-free reference nor the three
twins has run. No model code, card, deck, physical selector, TKE floor,
carried state, or score changed. Decision 129 remains pending, and this repair
continues to use the as-built option: NEMO's own TKE floors with IWM off.

## Diagnosis and verdict

The emitted stream header is 36 bytes: the 16-byte magic is followed by five
native four-byte integers. The actual kt-1 frame reads
`NEMO_V1_HMI_1`, version `1`, step `1`, header shape `184 x 152`, and element
width `64` bits; the remaining payload contains `26,640 = 180 x 148` fp64
values. Thus the refusal text's `1 1 64 26640` means version, step, element
width, and payload element count; it omitted the two header dimensions.

**The writer was wrong; the checker was right.** In the compiled enclosing
`SUBROUTINE v1_dump_hm_i`, the old header wrote haloed local-domain `jpi,jpj`
but the next statement wrote `l4_canon_2d(hm_i,'T')`
(`ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:449-466`). The repaired
writer describes the array it actually passes by writing `SIZE(hm_i,1)` and
`SIZE(hm_i,2)`; the payload statement is unchanged
(`scripts/validate/ocean_fidelity/testcases/nemo_testcase_v1_orca2tke/run.sh:62-94`).
The strict checker still requires payload count to equal the declared shape
(`scripts/validate/ocean_fidelity/testcases/nemo_testcase_v1_orca2tke/run.sh:102-127`).

## Resume and non-vacuity

The only exception is an explicit `legacy-v1-r1` admission for the already
completed smoke's exact `184 x 152` header and `180 x 148` payload. Admission
also pins the smoke binary, both frame hashes, all staged inputs, both decks,
resolved V1 switches, step-2 completion, and the step-2 restart
(`scripts/validate/ocean_fidelity/testcases/nemo_testcase_v1_orca2tke/run.sh:304-327`).
It does not make the strict checker accept the malformed header.

`--resume` admits that smoke without executing it, repairs/rebuilds the existing
record target, validates the compiled corrected header, and begins at the
writer-free boundary run. It refuses any pre-existing reference or twin
directory before doing so
(`scripts/validate/ocean_fidelity/testcases/nemo_testcase_v1_orca2tke/run.sh:368-408`).

The synthetic writer-bug frame is refused by the strict checker. The narrow
legacy admission accepts only the registered old layout, and a planted
`184 x 151` header exits nonzero
(`tests/ocean/fidelity/test_nemo_testcase_v1_orca2tke.py:246-260`). This proves
the repaired check refuses a truly bad frame rather than merely accepting any
payload whose count is `26,640`.

## Preflight, tests, and review

Clean committed preflight at `01c79310a` printed:

```
V1_PREFLIGHT_OK commit=01c79310a961de84be18972ad863b507b7aa59c9 reference=G5_writer_free_unbuilt record=ORCA2V1_OMIP_L4_R1FULLENTRY expected_boundary=kt11_after_10_completed
V1_SMOKE_ADMITTED_IN_PLACE .../smoke_2step (legacy header 184x152, payload 180x148; smoke not rerun)
V1_RESUME_PREFLIGHT_OK target=pinned-legacy next=boundary_then_twins command=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/vmix_r2-2557330026/scripts/validate/ocean_fidelity/testcases/nemo_testcase_v1_orca2tke/run.sh --resume
```

The final synchronous run reports **34 passed**: 11 focused V1 tests, 17
receipt-citation tests, and six private-cross-import tests. It includes the
strict mismatch and legacy-header plant. `bash -n` and `git diff --check` pass.

Independent review: operator (Claude).

## Scope and open item

ASKED: identify whether writer or checker was wrong; repair it; add a firing
bad-frame plant; resume from the completed smoke without rerunning it; preflight
and return the exact acquisition command. UNASKED choices: none. No NEMO
trajectory or fidelity result was measured this round.

OPEN: the operator runs:

```
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/vmix_r2-2557330026/scripts/validate/ocean_fidelity/testcases/nemo_testcase_v1_orca2tke/run.sh --resume
```
