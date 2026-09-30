# ORCA2 round 79b — what the 1 m TKE mixing-length floor was compensating

Status at this report: **ACQUISITION_NEEDED**.  The reading half of the round is
complete and it already names a candidate; the measurement half needs a NEMO
record that does not exist on disk, and the launcher for it is committed and
preflight-clean.

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round79b.md`
(committed before any measurement).

## The finding from reading alone

**ORCA2's deck runs internal-wave-driven mixing and legoESM's ORCA2 card does
not.**  NEMO adds a wave-driven diffusivity on top of the turbulence closure's;
legoESM adds nothing.

Evidence, all read out of the pinned record and the compiled source:

* The deck selects it.  `namelist_cfg:397` sets `ln_zdfiwm = .true.`, and the
  run's own log confirms the resolved value and that the parameterisation
  initialised and read its six input fields (`ocean.output:1003`, `:1045-1093`).
  The same log line `:1039` is the two-arm choice round 78 named: with the wave
  arm on, NEMO forces the turbulent-energy minimum to 1e-10 and the
  mixing-length floor to 1.0e-3 m.
* NEMO applies it last in the vertical-physics chain.  In the compiled
  `zdfphy.f90`, the closure's coefficients are copied into the working arrays
  (`:349-350`), the river mouths add to them (`:355`), the convection arm runs
  (`:359`), the salt/heat split runs (`:363`), and then the wave arm adds to all
  three coefficients (`:372`).  The addition itself is
  `zdfiwm.f90:314-316`, with the added diffusivity clamped into
  `[1.4e-7, 1e-2]` m2/s and masked at `:294`.
* legoESM's ORCA2 card resolves the arm OFF, and says so.  The card's
  `unmeasured_features` tuple has carried `internal_wave_mixing` from the
  beginning and is machine-pinned by the round-1 ladder gate's
  `EXPECTED_UNMEASURED`.  The resolved card prints `iwm.enabled = False`.
* The inputs are on disk.  `zdfiwm_forcing_orca2.nc` is in the pinned input set
  of every ORCA2 run directory and in the shared input tree, with its hash in
  `input_files.sha256`.  So the missing arm is transcription work, not an
  acquisition.
* legoESM has the numerics already: `ocean/physics/vertical_mixing/internal_wave_mixing.py`
  carries the de Lavergne diffusivity, and the implicit K-profile path adds it
  to both the tracer and the momentum coefficient when the arm is enabled.  The
  ORCA2 card runs that implicit path.  Nothing is missing but the card's
  selection and the forcing fields it would need to read.

**Predictions P1 and P2 are CONFIRMED.**  P3, P4 and P5 need the record.

### Two further arms the same reading turned up

Neither was in the round's brief; both are in NEMO's chain between the closure
and the consumers, and both are reported, not acted on.

| NEMO arm | deck | legoESM ORCA2 card | statement |
|---|---|---|---|
| internal-wave mixing | `ln_zdfiwm = .true.` | OFF, declared unmeasured | `zdfiwm.f90:314-316` |
| double-diffusive salt/heat split | `ln_zdfddm = .true.`, and the log says "use double diffusive mixing: avs /= avt" (`ocean.output:1012`) | OFF, though its two parameters are transcribed at the deck's values | `zdfphy.f90:363` |
| river-mouth diffusivity | `ln_rnf_mouth = .true.`, `rn_avt_rnf = 1e-3` (`ocean.output:648-649`) | no such statement anywhere in the package | `zdfphy.f90:355` |
| enhanced-diffusion convection | `ln_zdfevd = .true.`, `rn_evd = 100`, `nn_evdm = 0` | transcribed and matching | `zdfphy.f90:359` |

The salt/heat split is the one that bears on round 79a's headline: it is the
only arm in the chain that gives salt a different diffusivity from heat, and
round 79a measured salinity's independent-month maximum moving +79.0% while
temperature's moved -2.2%.  That is a hypothesis, not a result.

## Hypothesis table, as preregistered

| id | hypothesis | status after the reading half | what settles it |
|---|---|---|---|
| a | the TKE chain itself (mixing-length recurrence, surface anchor, step-entry N-squared routing) | OPEN.  The card states `nemo_first_wzv_after_ssh = rk3_extrapolated` and takes the carried step-entry route, and it resolves `nn_mxl = 3` with the surface-anchor overwrite on, all matching the deck | NEMO's recorded `en`, `mxlm`, `mxld`, `dissl` substituted one at a time |
| b | the internal-wave arm | **CANDIDATE**, on the reading above | the recorded increment `avt_after_iwm - avt_after_ddm`, against the diffusivity the retired 1 m floor produced on the same record |
| c | the consumers | OPEN | the recorded end-of-chain coefficients substituted into legoESM's implicit solves |

The arithmetic that makes b plausible, stated as arithmetic and not as
evidence: with the turbulent energy sitting at its forced minimum the retired
1 m floor produced about 1e-6 m2/s of tracer diffusivity where NEMO's floor
produces about 1e-9, and the wave arm's own clamp starts at 1.4e-7 and runs to
1e-2.  Whether the two actually overlap where it matters is exactly what the
record measures; the floor also binds in the stratified thermocline, where the
closure's physical length is well under a metre, and the numbers there are not
known.

## What the record does not yet cover

The one TKE-walk stream that exists on this deck
(`oracle_tke_walk_kt00000002.bin`) is a single step on a single rank and it
stops at the closure's own output: it has no wave increment, no salt/heat
split and no end-of-chain coefficients, which is the whole discriminator.  The
acquisition below is the first record of the chain itself.

## The acquisition

Committed at
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round79b_vertical_mixing_acquisition/`
(writer, two additions-only patches, launcher, README) with its admission gate
at `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round79b_vertical_mixing_gate.py`.
Every artefact is referenced by a repo-relative path; nothing lives in `/tmp`.

* Write-only, additions-only: the removed-line census of both patches is zero,
  checked by the launcher before it builds.
* Ten steps, both ranks, twenty-three named arrays per file.
* Self-describing, per note BD: sixteen-byte magic, fifteen header integers,
  then per array a sixteen-byte name, a rank, three extents, three origins and
  the payload.  The checker reads every name and every payload length out of
  the file and predicts no byte count.
* Admission is note-AS: all four ten-step restart files byte-identical to the
  pinned record, then the five plants, then the gate.
* The launcher's own `PYTHONPATH` carries the repository root, which is the
  round-67 defect.
* The evidence directory is created before the free-space check, which is the
  round-50 defect.

Preflight proofs run in this round, in the sandbox:

* both patches apply at `--fuzz=0` against the pinned pristine sources;
* the writer module and both patched sources preprocess with the deck's four
  compile keys and pass `gfortran -fsyntax-only` against the source build's
  own module set — `SYNTAX OK` for `zdfphy_round79b_writer`, `zdftke`, `zdfphy`;
* the launcher's five `ocean.output` predicates all match the pinned record;
* the gate's non-vacuity test passes, 7 tests: the synthetic clean record is
  admitted, each of the five plants makes the gate refuse, and a record set
  missing a step is refused.

### Command for the operator

```
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/orca2_r79b/repo/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round79b_vertical_mixing_acquisition/run.sh --run
```

## Which cards execute the statements named here

Measured from the resolved cards, not from prose.  ORCA2 and GYRE both resolve
the wave arm and the salt/heat split OFF; the three tank cards carry no physics
configuration at all, so they run no closure and no chain.  Only ORCA2's deck
selects either arm, so a landing of the wave arm or the split is ORCA2-scoped
and cannot move GYRE — which must still be measured, not argued, when the
landing comes.

## Review

Codex is refused this round: the quota guard reports 84.0% used (reset
2026-10-01 21:50), which the round's order treats as refused.  One Claude
code-reviewer subagent is the review; its verdict is recorded below.

## OPEN

1. The operator runs the acquisition; the round resumes at the substitution
   table (predictions P3, P4, P5).
2. The double-diffusive split and the river-mouth diffusivity are named, not
   measured.  Neither is in this round's brief and neither is acted on.
3. Every ORCA2 magnitude measured before the fold-in remains superseded
   (round 78's note B16); nothing in this round changes that.
4. No model file is touched in this round, so GYRE, DINO and the tanks are
   untouched by construction and are not re-run.
