# Preregistration: post-split momentum commits, round 33

Date: 2026-08-30. Frozen before the first round-33 numerical execution.
Session `01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Question and inherited release

The official round-32 receipt, SHA-256
`0b284d7c8880646850daee86b2b85fb13fb65540402e2e9a02760b0ed60ba86f`,
certifies SSH/U/V at all 68 split-explicit substep starts. There is no strict
failure and the final U/V point errors are `4.83e-16`/`4.44e-16`, about 30
times below the registered linear-accumulation bound. Rows 1--3 are therefore
released in the ordered chain. This round asks for the first failing operand
among rows 4--6, using the existing day-180 deterministic-writer streams.

No NEMO build or run is authorized. The admitted baseline is the retained
`RUN_ZDF21_COEFF_OFF.Nvczr8` stream set, whose 197-stream manifest is already
bound by `dino_zdf_row21_coeff_assembly_artifact.json`. Its row-4--6 files are
byte-identical to the corresponding faithful-writer arm in that artifact.

## Ordered rows and source operands

| Row | Operation | Executed NEMO source | Existing operands |
|---:|---|---|---|
| 4 | `dyn_zdf` after-velocity | `stpmlf.F90:396-409`; initial state/removal `dynzdf.F90:137-178`; active U/V matrices and recurrences `dynzdf.F90:180-337` | `stp_dump_07_dynspg_kt00005761_{u,v,ub,vb}`, `zdf_dump_{u,v}1_{pre,post}stress`, `dump_avm`, `stp_dump_08_dynzdf_kt00005761_{u,v}` |
| 5 | second `wzv` NOW velocity | `stpmlf.F90:411-412`; QCO bottom-up continuity `sshwzv.F90:168-228` | `seq_dump_hdiv_nnn_kt00005761`, QCO depth fields, `wzv_dump_ww_call2.bin`; call 1 is retained only as the ordering control |
| 6 | `mlf_baro_corr` after-level reconciliation | `stpmlf.F90:578`; depth sum and uniform correction `stpmlf.F90:709-765` | `baro_dump_{u,v}_{before,after}_kt00005761.bin` and certified round-32 final barotropic U/V |

The older `dyn_zdf_probe.py` statement that no true pre-solve dump exists is
withdrawn by the committed `s17_dynzdf_bracket.py`: during the step `Naa` and
`Nrhs` alias, so stage 7 is the true post-`dyn_spg` RHS. The reconstruction

`naa_B = (u(Kbb) + 2*dt*Krhs - u_b(Kaa))*mask`

must reproduce `zdf_dump_*1_prestress` at a maximum absolute error no larger
than `1e-14`; the wrong `dt` and zero-`dt` plants must fail. NEMO's own
`poststress-prestress` level-1 increment is then added before the production
implicit solve. This is the row-4 oracle-input arm. A null substitution must
be byte-identical to the unmodified production arm.

## Bars and disposition

Every directly comparable signed field is scored by
`fidelity_bar_gate.classify`, including correlation and mean-absolute-ratio
axes. Row 4 (`dyn_zdf`) and row 5 (`wzv`) retain their registered
`ACCUMULATING` maximum/RMS bar of `1e-12`; row 6 (`mlf_baro_corr`) retains its
registered `POINTWISE` bar of `1e-15`.

* Row 4 is `VERIFIED` only if both U and V in the oracle-input arm are AT BAR.
  Otherwise the first failing component/partition is the ordered stop.
* Only if row 4 verifies, row 5 is read. Its literal source reconstruction
  must reproduce the NEMO call-2 dump at arithmetic scale before the legoESM
  comparison is admitted. A call-1 substitution must be detectably worse or
  the call-order control has no power.
* Only if row 5 verifies, row 6 is read. The primary local arm substitutes
  NEMO's own pre-correction U/V into the production reconciliation while
  retaining the certified production barotropic target. Both components must
  be AT BAR. The unchanged-input null must be byte-identical.

Rows after the first non-bar result remain `ORDERED-BLOCKED`. Aggregate or
wall-band diagnostics may be recorded as non-dispositive localization only;
they cannot promote a failing pointwise/class gate.

## Admission and controls

The scorer runs from a tracked-clean checkout on CPU/fp64 with checkout-first
`PYTHONPATH`, `DINO_1226_LANE=d180`, `LEGOESM_NEMO_E3T=both`, and an exported
`CODEX_SESSION_ID`. It requires the full-halo stream sizes and exact wet
populations (U 9,758; V 9,868; T 9,920), hashes every consumed dump, the
retained-manifest artifact, restart, mesh, active NEMO sources, production
modules, scorer, preregistration, and round-32 receipt. It admits the retained
producer by stream hash plus model-diff-zero, not by requiring its historical
commit to equal current HEAD.

Identity must classify AT BAR; a five-`nextafter` plant must turn the same
path red; zero shift must win an actual alignment scan; hook counts and
restoration are mandatory; all captured arrays must be finite. Any failed
receipt, shape, reconstruction, control, or restoration makes the artifact
`INVALID`, not evidence. No SLOT block is allocated because all oracle
operands already exist.

## Instrument amendment after invalid attempt 1

Attempt 1 stopped at reconstruction control C1 and produced no artifact. The
scorer had used the retained run's `DINO_00005761_restart.nc`, which is the
output of the measured step (`kt=5761`, `adatrj=180.03125`), as though it were
the entry state. The required entry restart is the original sequential lane's
`DINO_00005760_restart.nc` (`kt=5760`, `adatrj=180.0`). The scorer now takes
that path explicitly, hashes it, and refuses a missing file. No operand,
statistic, bar, or disposition rule changes; the failed C1 receipt is invalid.

Attempt 2 stopped before model construction because the retained run already
contains a symlink to the same `5760` entry restart and the temporary view
tried to create a duplicate name. No numerical operation or score ran. The
view builder now admits an existing link only after its content hash equals
the explicit entry-restart binding; otherwise it stops. The science design is
unchanged.

Attempt 3 completed all four numerical arms and wrote a provisional JSON, but
the process then exited nonzero while printing a nonexistent display-only
metric key. Under the lane's exit-zero admission rule the provisional file is
not a receipt. The display now uses the classifier's actual
`per_element_max_error_over_nemo_rms` key. The scorer also records the
preregistered five-`nextafter` plant explicitly and requires it, rather than
relying on the older helper's four-ULP plant. Arms, comparands, bars, and
disposition logic are unchanged; attempt 3 is not cited.

Attempt 4 exited zero but classified the artifact `INVALID`, as required:
five ULPs at the largest-magnitude wet element give only `1.69e-14` maximum
error over reference RMS, below this row's `1e-12` ACCUMULATING class bar.
That plant is structurally incapable of turning this 336k-element row red.
It is retracted and replaced by adding `2e-12*RMS(reference)` at that same
wet element, through the same row classifier. The identity and actual
zero-shift controls remain. No measured arm, class bar, or verdict rule is
changed; attempt 4 supplies no science receipt.
