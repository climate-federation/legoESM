# Grid ladder rung G5 — round 1 receipt (gridladder_r31)

Status: **STOPPED_FOR_RECORD.**  The first operator acquisition stopped during
the two-step smoke at kt=1 because an ORCA1-only diagnostic rejected ORCA2's
ice selectors.  The corrected acquisition is committed; no failed-smoke field
is admitted and no trajectory or bar verdict is claimed.

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g5_round1.md`
at `9970164eb`.  Card/acquisition: `888621bd3`; inherited-selector test
registration: `c608effd7`.  Acquisition evidence is specified under
`gridladder_rounds/round31/oracle_g5_r31`.

## Resolved-deck comparison (corrected 2026-10-15)

G5 is the certified G4 deck plus ORCA2's sea ice.  It is **not** the ORCA2
lane's Decision-103 OMT-0.  The direct G4→G5 assignment delta is exactly one
key:

| key | G4 | G5 / ORCA2 |
|---|---:|---:|
| `namsbc.nn_ice` | `0` | `2` |

ORCA2 states `nn_ice=2` at
`cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:77-95` (the assignment is :85), and
the running branch dispatches value 2 to SI3 at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcmod.f90:499-503`.

Decision-103's OMT-0 is the rung-0 deck: ORCA2 mesh and 31 levels,
climatological T/S under the `ln_tsd_dmp` guard, from rest, with no surface
forcing and no ice; advection, lateral diffusion, and drag are off while EEN,
split-explicit free surface, and constant vertical mixing remain.  Its binding
gate sets `ln_blk=false`, `nn_ice=0`, `ln_traqsr=false`, `ln_ssr=false`,
`ln_rnf=false`, and `nn_fwb=0`
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round82_rung0_deck_gate.py:21-43`).
G5 instead keeps every G4 forcing switch and sets `nn_ice=2`; it climbs the
forcing/ice axis toward the shipped ORCA2 program, not the OMT module ladder.

## Exact SI3 overlay

The committed `namelist_ice_cfg` is byte-identical to ORCA2's file, SHA-256
`9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8`.
ORCA2 states all sixteen overlay groups and no assignments, so every SI3 value
resolves from `cfgs/SHARED/namelist_ice_ref`: `nampar` :22-24, `namitd` :26-28,
`namdyn` :30-32, `namdyn_rdgrft` :34-36, `namdyn_rhg` :38-40, `namdyn_adv`
:42-44, ice `namsbc` :46-48, `namthd` :50-52, `namthd_zdf` :54-56,
`namthd_da` :58-60, `namthd_do` :62-64, `namthd_sal` :66-68, `namthd_pnd`
:70-72, `namini` :74-76, `namalb` :78-80, and `namdia` :82-84, all in
`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_cfg`.

The card also applies SI3's initial global snow+ice+pond mass correction to
G4's bump.  The shared helper preserves the already-tested full ORCA2 card's
value `0.09343505872684969 m`; G5's T/S/U/V remain G4-identical.

## How ice enters the ocean card

No sea-ice model was added to legoESM.  Decision 123 is represented by card
labels `nemo_post_si3_records` / `nemo_ocean_surface_input_v1`, and the card is
fail-closed on `g5_record_not_admitted`.  The ORCA2 lane's mapping was read at
`nemo_testcase_l4_orca2_round1_ladder_gate.py:765-825` on the read-only ORCA2
branch and is retained exactly:

| NEMO post-SI3 field | G5 ocean input | reason |
|---|---|---|
| `qsr`, `qns` | `sw_down`, source-rounded `q_net=qsr+qns`, `q_nonsolar=qns` | SI3 has already replaced the ocean heat fluxes (`iceupdate.f90:142-175`) |
| `emp` | freshwater `evap=emp`; separate `ice_fw=0` | `emp` already contains atmosphere-ocean plus ice-ocean mass exchange; adding `fwfice` again would double count (`iceupdate.f90:181-192`) |
| `sfx` | `salt_flux=1e-3*sfx` | SI3 has already accumulated all ice salt exchange (`iceupdate.f90:194-197`) |
| `fr_i` | `ice_concentration` | SI3's ocean-facing fraction (`iceupdate.f90:201-213`) |
| `utau`, `vtau`, `taum` | native ocean stress and modulus | these are the already-combined open-water/ice-ocean stresses (`iceupdate.f90:373-419`) |
| `rnf`, `rnf_tsc` | runoff water and tracer content | same ORCA2 record route; runoff is not inserted into tracer dilution twice |
| chlorophyll input | ORCA2 lane's verified per-step `fld_read` reconstruction | the surface record contains fluxes, while the RGB attenuation operand comes from the pinned `chlorophyll.nc` clock |

The reused writer is called immediately after `sbc`, not at injection:
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/MY_SRC/stprk3.F90:158-166`.  Its schema
records combined stress/heat/freshwater/salt at :451-470, `fwfice`, runoff and
`fr_i` at :471-472, snow/ice mass and `rCdU_ice` at :473-474, and `rnf_tsc` at
:481.  Thus no requested exchange is absent from the record.

One ocean implementation gap is intentionally not hidden.  ORCA2 resolves
`ln_drgice_imp=true` (`cfgs/SHARED/namelist_ref:810-819`); SI3 forms
`rCdU_ice` at `iceupdate.f90:399-404`, NEMO copies it to `rCdU_top` at
`zdfdrg.f90:178-188`, and the external mode first consumes it at
`dynspg_ts.f90:1426-1432`.  The writer records `rCdU_ice`, but legoESM has no
top-drag forcing input.  This is the preregistered predicted first non-bitwise
boundary and remains explicit in `ORCA2G5_UNMEASURED`; the acquisition can
measure it without inventing a second ice model.

## Acquisition

The first `run.sh` reused R5's complete MY_SRC set.  That set was built for the
ORCA1 ice deck: its extra Prather dump stops unless `nn_icesal=2` and ponds are
off (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/MY_SRC/icedyn_adv_pra.F90:504-515`).
G5 resolves `nn_icesal=4`, `ln_pnd_LEV=true`, and Prather advection from
`cfgs/SHARED/namelist_ice_ref:127-130,197-204,240-250`.  The operator's smoke
therefore wrote `time.step=1`, logged `l4_orca1ice_pra_dump: selector contract
changed` twice, and exited 123.  The failed directory was preserved as
`smoke_2step_failed_orca1ice_writer_20261015`; none of it is admissible.

The corrected acquisition replaces only `icedyn_adv_pra.F90` with the pinned
SHA-256 `d1be8a22cbb6dd2f7c695e31164ba3d09b4cd3f5bcbe5f71ed0cfec5749f3828`
from `ORCA2_OMIP_L4_P2VBND_R2`, whose writer ran the same ORCA2 ice overlay and
serializes the selected salinity and pond families behind their allocation
selectors (:502-543).  The complete hybrid MY_SRC manifest is pinned as
`f1bfa3ee99cf841e9dab64204b0967e91715084daf74b43489935c8c3181e59c`.
Preflight now resolves the three selector values from the reference plus
overlay and rejects the old writer before any build; the direct planted test
passes the ORCA2 writer and makes the ORCA1 writer exit nonzero.

Two fresh config names prevent reuse of any object from the failed build:
`ORCA2G5_OMIP_L4_R31REF_FIX1` is a writer-free G5 reference with only the bump
sources, and `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1` is the record build.  Both use
the same pinned CPP keys.  The record smoke runs first, the writer-free
reference discovers NEMO's boundary, then record twins run only to that
boundary.  The surface admission now checks the exact byte count implied by
the frozen schema as well as finite values and twin identity.

Execution order is fixed: two-step smoke first; a 100-step boundary run;
then twins A/B/C only to `min(10, completed)` rounded to `nn_fsbc=2`.  The
expected inherited boundary is a stop while beginning kt 11, hence completed
kt 1–10.  Across A/B/C, restart coverage is every writable step 2 through the
admitted boundary; step 1 is never requested.  Per-step surface frames must be
finite, schema-valid, and byte-identical across all three twins.

Corrected read-only preflight at `a8d113f38` passed:

`G5_PREFLIGHT_OK source=ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY commit=a8d113f38dda1829dca7be6f2233a6fff1d01000 expected_boundary=kt11_after_10_completed`

It would build the fresh writer-free and record targets named above, then run
the smoke, boundary reference, and twins.  No `makenemo` or `mpirun` was invoked
by the author.

## Preregistered prediction and tests

Predicted first non-bit statement: `dynspg_ts.f90:1431-1432`, the first
external-mode assembly containing `rCdU_top`.  Falsifiers are any earlier
state/supported-flux mismatch, absent or everywhere-zero wet-ice `rCdU_ice`,
an unexpected resolved-deck key, or exact agreement through that consumer.
The exact observed boundary supersedes the prediction next round.

- Focused G4+G5, CPU fp64 (`JAX_ENABLE_X64=1`): **38 passed, 2 deselected**.
- Split G2/G3 inherited-selector regression: **27 passed, 9 deselected**.
- Split G1 selection regression: **8 passed, 77 deselected**.
- The initial combined 148-test process was discarded after hitting the
  documented JAX per-process compiler limit (`SIGABRT`); splitting is the
  repository-prescribed remedy, and the affected selector assertions pass.
- `bash -n`, Python compilation, `git diff --check`, and acquisition preflight:
  PASS.
- Controls: changing `nn_ice`, modifying any byte of the SI3 overlay, selecting
  the ORCA1-only Prather writer, moving the surface writer above exact `CALL
  sbc`, truncating its payload, omitting a required field, requesting a step-1
  restart, a non-finite frame, or unequal twins is refused by a direct test or
  acquisition admission.  Each of the three new review controls was planted
  and observed to exit nonzero or fail its assertion.

independent review: Claude — HOLD on the first acquisition; all four findings
were incorporated.  Per the operator's resume instruction, no Codex review was
run on this correction.

## OPEN

The operator must run the acquisition.  Round 2 admits the record and tests
the preregistered top-drag boundary; G5 is not execution-ready before that.

ASKED: exact G4+ORCA2-SI3 deck, corrected non-OMT framing, ORCA2 surface/ice
record route, compatible SI3 writer acquisition, card/preregistration, focused
synchronous tests.
UNASKED: the card-side G1–G4 rejection of an unexpected `namelist_ice_cfg` is
retained and changes no run deck; no physics fix, new sea-ice model, NEMO source
edit, or author-run integration was added.
UNVERIFIED: G5 trajectory, actual stop boundary, nonzero wet-ice `rCdU_ice`,
and the first numerical mismatch.
