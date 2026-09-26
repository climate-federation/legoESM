# Preregistration: split-explicit / momentum-commit chain, round 1

Date: 2026-08-29. Frozen before running the round-1 receipt probe.

## Claim and scope

This round orders the active DINO MLF calls at the matched day-180 state and
uses only the existing `RUN_SEQDUMP_D180_1R`/deterministic-writer files. It
asks where the chain first diverges; it does not change physics, build NEMO,
run MPI, use a GPU, or make a climate claim.

The inherited lateral receipt is an admission fact, not a new score: its
production tracer-entry `uu(Kmm)` comparison diverged at all 9,758 wet U
columns (correlation 0.9845023189, RMS ratio 1.0296145800). The source-site
rewrite is active NEMO `cfgs/DINO/MY_SRC/dynspg_ts.F90:1170-1174`.

## Ordered registry

The order below is the executed `stpmlf` order, not a dependency guess.

| Row | Active call and output | Executed NEMO evidence | Existing day-180 evidence |
|---:|---|---|---|
| 1 | `dyn_spg_ts`: split-explicit solve, `uu_b/vv_b(Naa)`, `un_adv/vn_adv`, and `uu/vv(Kmm)` rewrite | `cfgs/DINO/MY_SRC/stpmlf.F90:332`; complete MLF forcing assembly `cfgs/DINO/MY_SRC/dynspg_ts.F90:316-445`; final rewrite `:1170-1174` | `spg_dump_{zu,zv,ssh}_frc`, loop seed, substep-1, final `puu_b/pvv_b/pssh`, and final `un_adv/vn_adv`; `stp_dump_07_dynspg_*` |
| 2 | second `div_hor`: `hdiv(Nnn)` | `cfgs/DINO/MY_SRC/stpmlf.F90:349-376`; active MLF kernel `src/OCE/DYN/divhor.F90:151-197` | `seq_dump_hdiv_nnn_kt00005761.bin` through `...5764.bin` |
| 3 | second `dom_qco_r3c`: `r3t/r3u/r3v(Naa)`, `r3f` | `cfgs/DINO/MY_SRC/stpmlf.F90:378-394`; active MLF QCO formulas `src/OCE/DOM/domqco.F90:140-186` | `seq_dump_r3{t,u,v}_aaa_kt*`, `seq_dump_r3f_kt*` |
| 4 | `dyn_zdf`: momentum `uu/vv(Naa)` commit | `cfgs/DINO/MY_SRC/stpmlf.F90:396-409`; active vector-form arm and split-explicit removal `cfgs/DINO/MY_SRC/dynzdf.F90:137-141,166-178` | `stp_dump_08_dynzdf_kt*_{u,v}.bin` and the completed ZDF chain receipt |
| 5 | second `wzv`: `ww(Nnn)` | `cfgs/DINO/MY_SRC/stpmlf.F90:411-412`; active MLF integration `cfgs/DINO/MY_SRC/sshwzv.F90:168-348` | `wzv_dump_ww_call2.bin` (call 1 retained as the ordering control) |
| 6 | `mlf_baro_corr`: restored after-level barotropic mean | `cfgs/DINO/MY_SRC/stpmlf.F90:578`; correction `:709-765` | `baro_dump_{u,v}_{before,after}_kt00005761.bin` through `...5764.bin` |

Rows after the first DIVERGED row are `ORDERED-BLOCKED` in this round. Their
dumps are inventoried, not silently promoted to chain verdicts.

## Row-1 operand ladder and gates

The committed wrapper
`scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round1.py`
reuses the production-state construction, NEMO readers, time-level registry,
and production barotropic call in the already committed
`spg_substep_chain.py`. It intercepts that probe's reports; it does not copy
or rederive the solver.

For every signed field, retain the diagnostic

`E = RMS(legoESM - NEMO) / RMS(NEMO)`

over the probe's registered wet mask. The decision itself now calls the
campaign's committed `fidelity_bar_gate.classify`: correlation must be at
least `1 - 1e-9`, mean-absolute ratio must be within `1e-6` of one, and the
maximum `|diff|/RMS(NEMO)` must clear the arithmetic-class bar below.
`MATCHED` requires `AT BAR` for every component; otherwise the subrow is
`DIVERGED`.

| Subrow | Operand/output | Bar | CONFIRM / REFUTE |
|---:|---|---:|---|
| 1.1 | slow momentum forcing `zu_frc`, `zv_frc` | POINTWISE `1e-15` plus aggregate axes | CONFIRM exact operand receipt / REFUTE |
| 1.2 | loop seed `sshn_e`, `un_e`, `vn_e` | POINTWISE `1e-15` plus aggregate axes | CONFIRM exact entry state / REFUTE |
| 1.3 | first substep `ssh`, `ub`, `vb` | ACCUMULATING `1e-12` plus aggregate axes | CONFIRM accumulated prefix / REFUTE |
| 1.4 | final `puu_b`, `pvv_b`, `pssh`, `un_adv`, `vn_adv` | ACCUMULATING `1e-12` plus aggregate axes | CONFIRM split-explicit output / REFUTE |

The first source-ordered failing subrow localizes row 1. `zu_frc`/`zv_frc`
are operands, not causes: a failure there transfers the next peel to the
vertically averaged explicit momentum RHS and must not be called a
barotropic-integrator defect.

## Controls and stop rule

The wrapper requires a clean Git worktree, CPU, fp64,
`LEGOESM_NEMO_E3T=both`, lane `d180`, exact registered populations, finite wet
values, and the probe's own registered time levels. It records and enforces a
zero-shift alignment scan on the actual legoESM/NEMO `zu_frc` pair. Its scorer
controls must all fire before an artifact is admitted: identical arrays score
zero, a planted `1e-6` RMS-scale offset turns an identical pair red, the same
perturbation changes the actual binding's score, and zero shift is the best
alignment. Any missing field, failed control, shape/population mismatch,
non-finite wet value, wrong lane, dirty tree, or nonzero inherited-probe exit
is `INVALID`, not evidence. The receipt hashes the Git commit, all 18 dumps
read by the inherited probe, restart, mesh, output/namelists, production
modules, recipe fingerprint, scorer, wrapper, and active NEMO sources.

## Instrument amendment after invalid attempt 1

Attempt 1 exited before producing an artifact because the inherited probe's
bridge did not carry native T-point latitude, now required by the production
card's `tke_htau_evaluation="nemo_literal"`. No row score was admitted. Before
attempt 2, `spg_substep_chain.py` is amended to resolve the card before the
bridge and pass the same `omega` and `carry_native_lat_deg` condition used by
`kamm_twin_90d.py:889-905`. This is a harness-input repair only; the ordered
rows, arrays, bars, controls, and stop rule above are unchanged.

## Instrument amendment after invalid attempt 2

Attempt 2 completed the inherited calculation but the receipt correctly
refused admission because it required `ssh_frc`. The NEMO dump exists, but
this production recipe has no legoESM `F_slow_eta` while freshwater closure
is inactive, so no paired score exists. Before attempt 3, `ssh_frc` is marked
`UNMEASURED` and removed from momentum subrow 1.1; the U/V fields and their
bars are unchanged. The inherited log also exposed superseded hard-coded
seed and accumulation prose inconsistent with its live values. That prose is
made conditional or withdrawn in the probe before attempt 3, as required by
the campaign retraction rule; no numeric calculation changes.

## Instrument amendment after attempt-3 adversarial review

Attempt 3 exited zero, but both independent reviews returned HOLD before the
bundle was made. Its receipt is withdrawn. Before attempt 4, the wrapper is
changed to call the campaign classifier with its correlation, mean-absolute
ratio, and arithmetic-class per-element axes; reject shape, population, or
wet non-finite changes; enforce and record the actual legoESM-to-NEMO
zero-shift scan; and stamp the complete runtime provenance listed above. The
active MLF source citations are corrected, and the result may only say an
earlier operand diverges: recurrence and rewrite correctness remain
independently unmeasured without held-forcing counterfactuals. Numeric model
code and the row ordering are unchanged.

## Instrument amendment after first re-review

Both re-reviewers retained HOLD because the planted arrays had not traversed
`fidelity_bar_gate.classify` itself and because runtime provenance still
fingerprinted the base recipe rather than the probe's effective replaced card.
Before the next run, identical and planted metrics are classified through the
same decision function used for row verdicts and must flip `AT BAR` to a
non-bar state. The actual replaced DINO config and derived model/barotropic
settings are captured at their production builder call; their fingerprints
and selected settings are recorded. Runtime Python/JAX/jaxlib/NumPy/platform/
CPU-device fields, the NEMO executable, and resolved `output.namelist.dyn` are
also stamped. Rows, bars, populations, numeric model code, and stop rule remain
unchanged.

## Instrument amendment after mechanism re-review

The instrument reviewer returned SHIP, but the mechanism reviewer retained
HOLD because inherited stage 6b still converted scalar RMS-magnitude agreement
into forcing ownership and skipped term attribution. Before the next run, that
claim is retracted in the executable probe: the ratios remain explicitly
noncausal diagnostics, cancellation is named, and stage 6c always reports term
ownership UNMEASURED pending a held-forcing or vector-residual test. No numeric
calculation, registered row, gate, or stop rule changes.
