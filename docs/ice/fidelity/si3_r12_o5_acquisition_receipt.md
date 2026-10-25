# SI3 round 12 receipt — O5 operator acquisition tool

## Outcome

The O5 NEMO acquisition tool is complete and its read-only preflight passes.
No NEMO build or integration was run. The operator must run
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o5_orca2_sas_acquisition/run.sh --run`;
until that happens, O5 remains UNMEASURED against NEMO.

Resume 4 diagnoses and repairs the retained reference-run initialization
failure. The offending assignment was `ln_ice_embd` on line 28 of the original
committed O5 `namelist_cfg`: it was under `&namdom`, and `stage_run` copied that
file unchanged into the run directory. NEMO 5.0.2 declares `ln_ice_embd` only
in `&namsbc`; the fixed deck writes it there. `nn_itend=3` was and remains in
`&namrun`, matching NEMO 5.0.2's group ownership and the admitted O1/O2 decks.
The retained NEMO diagnostic was:

```text
  ===>>> : E R R O R
          ===========
 misspelled variable in namelist namdom (cfg) iostat =  5010
```

The complete nonempty-group comparison is below. Empty groups in both staged
files are also required to exist in their corresponding reference file.

| staged group | NEMO 5.0.2 reference ownership | admitted O1/O2 | failed O5 | fixed O5 |
|---|---|---|---|---|
| ocean `namrun` | all six staged variables belong | `nn_itend=2`, `nn_stock=1` | `nn_itend=3`, `nn_stock=3` | unchanged; `nn_itend` stays here |
| ocean `namdom` | `rn_Dt` belongs; `ln_ice_embd` does not | `rn_Dt` only | `rn_Dt` plus misplaced `ln_ice_embd` | `rn_Dt` only |
| ocean `namsbc` | `nn_fsbc`, `ln_blk`, `nn_ice`, and `ln_ice_embd` belong | first three; reference supplies `ln_ice_embd=F` | first three | explicitly adds `ln_ice_embd=F` |
| ocean `namcfg`, `namsbc_blk`, `namsbc_sas`, `namlbc`, `nameos` | every staged variable belongs to its named group | same assignments | same assignments | unchanged |
| ice `nampar` | `ln_icedyn`, `ln_icethd` belong | both false | dynamics true, thermodynamics false | unchanged |
| ice `namdyn` | `ln_dynALL`, `ln_dynRHGADV` belong | no overrides | true, false | unchanged |
| ice `namdyn_rdgrft` | `ln_ridging`, `ln_rafting` belong | no overrides | both false | unchanged |

Preflight now parses both staged decks and NEMO's own `namelist_ref` files by
group. Every written variable must exist in that exact group; resolved-value
and registered-deviation checks are also group-qualified. An added `nn_itned`
in `&namrun` must refuse specifically as an unknown variable. If a NEMO launch
produces `E R R O R`, the launch path invokes the same log verifier before
retention, and the verifier quotes the NEMO banner and following diagnostic
lines rather than emitting only a generic token error.

Resume 3 diagnoses the operator refusal as a gate defect, not a patch or build
defect. Both O5 builds compiled. The rheology patch contains 46 added raw
lines, seven of which are blank; NEMO's FCM preprocessing removes those seven
blank lines, leaving the same 39 nonblank observer lines in the record ppsrc.
The old same-program check compared all 46 raw additions against those 39
compiled additions and therefore refused a correct build. Its refusal cleanup
then removed both O5 build trees, so they were no longer present for the
requested postmortem despite the retained evidence directory. The repaired
comparison requires every nonblank compiled insertion exactly and ignores only
raw blank lines that CPP does not emit. Refusals now retain both built trees.

The frozen overlay is one MPI rank, `nn_it000=1`, `nn_itend=3`,
`ln_dynALL=T`, `ln_dynRHGADV=F`, `ln_ridging=F`, `ln_rafting=F`,
`ln_icethd=F`, H79 strength with smoothing, and aEVP with 100 inner
iterations. No campaign bar was relaxed.

## Source-derived observation contract

The observer patches quote the running ORCA2_SAS_ICE arm, not a dead branch.
A1 brackets NEMO's strength call
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:268`). A2 is
after H79's raw strength statement
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90:1081`), while A3
is after smoothing and its T exchange
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90:1114`). B1 follows
the levitating SSH statement
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:281`), whose active
arm returns ocean SSH (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1090`).

B2 follows the one-time F-mask exchange
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:238`). Each selected
inner iteration 1, 2, 3, 50, 99, and 100 has separate records after the
delta exchange (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:440`),
after the stress update, and after the parity-dependent velocity exchange
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:647` and `:754`).
The entry and exit around `ice_dyn_rdgrft` retain the preregistered
no-redistribution control.

The writer produces direct unformatted stream writes with a 112-byte common
header. In addition to `jpi`, `jpj`, `jpl`, halo, decomposition, step,
iteration, rank, and SAS time-level metadata, it stores `Nis0`, `Nie0`,
`Njs0`, and `Nje0`. The three A2D(0) diagnostics are consequently read at
their declared 180-by-148 interior extent. Fast and iceberg U/V masks are
recorded only on the 183-by-151 rectangle actually filled by their common
NEMO loop, and inner stresses use the exact runtime `ntsi`/`ntei` and
`ntsj`/`ntej` bounds of their two producing loops. There are 82 records for
the three-step, one-rank acquisition window; signed-zero bits are retained.
The required `Kmm` is derived from the SAS `Nnn` assignment and `sbc` call,
then checked in every record rather than hard-coded by the reader.

## Admission and controls

The admission gate imports O1's wet-mask, ULP, signed-zero, and real-log
helpers and round 328's config-root-normalized same-program comparator.
The comparator's committed-patch check is defined on compiled source: raw
blank additions removed by NEMO's CPP are excluded, while every nonblank
insertion remains exact and all replacements or deletions still refuse.
`--admit-existing` reads the expected producer commit from the retained run's
own manifest rather than current `HEAD`; the gate verifies that object with
`git cat-file` and requires it to be an ancestor of `HEAD`. It reads the two
retained executable digests from that manifest's relative paths, verifies every
raw record and individual payload checksum, and therefore permits an older
producer after a newer gate-fix commit. It requires `RUN_COMPLETE` and `STOP 0`
independently for the reference and record runs, then byte-compares both step-3
ocean and ice restart files before scoring. A failed admission leaves a
completed evidence directory in place.

Strength and aEVP replay run under CPU production JIT with fp64/libm policy.
Strength, B2 `fimask`, all delta/stress/velocity fields at B3 iterations 1,
2, 3, 50, 99, and 100, final stresses and velocities, SSH, fast-mask filled
ranges, metric areas and reciprocals, and the no-redistribution entry/exit are
required to be 0 ULP on wet masks and full recorded arrays; seam and fold
maxima are reported. The shared production solver supplies the trace, so
inner-frame scoring and final-state scoring cannot diverge into two solvers.

Preflight now applies all three observer patches to unmodified NEMO source,
preprocesses patched and unpatched copies with the admitted O2 record build's
real keys and FCM `-P -traditional` flags, and sends those real ppsrc files
through the same same-program comparator used after an O5 build. A changed
`fimask` observer expression in that preprocessed source must refuse. Other
plants cover each of the twelve observer landing sites, header,
truncation, checksum, SAS time level, swapped fields, signed zero, extents,
rank collapse, and the first wet-cell one-ULP change. The bounds-checked
Fortran fixture is a hand-written driver into which preflight splices the
observer CALL lines copied verbatim out of the patched ppsrc; it is then
compiled with bounds checking and minimal declarations/stubs. Its NaN-guarded
filled-range limits are hand-set from the producing NEMO loops rather than
executing those loop bodies. At real extents it drives every spliced observer
call with distinct two-rank payloads, planted unfilled NaN edges, a `-0.0` cell,
and checks all 668 payload digests and finiteness. The patched sources are
preprocessed with the real O2 keys, syntax-checked against the real NEMO
modules, and the placement checks inspect that ppsrc rather than raw `.F90`.

In the operator path, same-program proof occurs after both builds and before
either launch. The disassembly census parses complete objdump instruction
lines, reports nonzero per-function equality and totals of 2928, 2231, and
309 for `icedyn_rhg_evp`, `icedyn_rdgrft`, and `icedyn`, respectively. It
examines only the Fortran flags used for those objects, so the arch file's
unrelated `%CFLAGS -O0` is not a refusal. A genuinely compiled extra
binary64 multiply/add makes the census fail in preflight. All preregistered
fold-sign labels, exchange-before-smooth, unmasked-smooth, parity-swap,
swapped-field, and wet-one-ULP admission controls are wired into retained-run
admission.

The committed provenance-bearing real excerpt is from the earlier
dynamics-off acquisition. Preflight therefore checks its provenance, while
`verify_o5_log` is tested on a synthetic dynamics-on excerpt assembled from it;
admission checks the complete real O5 `ocean.output`. The default path remains
read-only, while `--run` has RAM, disk, timeout, ERR and EXIT gates.

## Verification

- Resume-4 read-only preflight: PASS, including exact-group validation against
  both NEMO reference namelists, the `nn_itned` refusal plant, quoted-error
  plant, ten checklist sections, real-preprocessed
  same-program comparison and insertion plant, 12 landing controls and 12
  misplacement plants, the nonzero real-object census, and its injected-op
  plant; no NEMO build/run.
- Resume-4 O1/O2/O5 acquisition and O5 model regressions: 64 passed, one
  deselected in 665.05 s, including older-producer admission,
  real-preprocessed same-program proof, refusal retention, exact-group
  preflight, and quoted error diagnostics.
- The verifier was also run directly on the retained failed
  `reference_3step/ocean.output`: expected refusal rc 2, with lines 193, 195,
  and 197 quoted verbatim.
- Private-import ratchet: 6 passed against a complete archived checkout (the
  operator work clone omits two tracked ML sentinel files).
- Citation gate: all nine SI3 receipts and the default gate PASS with zero
  failures, unmapped citations, or stale full-map entries; gate tests: 17
  passed.
- NEMO acquisition result: ACQUISITION_NEEDED.
- Previous re-reviews: Argon RUN; Claude HOLD on retained-run admission was addressed by
  using the manifest producer commit and verifying its object and ancestry.
  Resume-3 Codex-authored diff review is operator-owned under the lane rule.

## Choices

ASKED: three steps, one rank, the six preregistered inner frames, exact-input
strength plus aEVP admission, and operator-only acquisition.

UNASKED: the model now exposes a trace view of the same aEVP execution so the
preregistered inner frames can be scored when evidence arrives; the existing
final-state solver API and every configuration default are unchanged. No
NEMO agreement is claimed before operator evidence exists.
