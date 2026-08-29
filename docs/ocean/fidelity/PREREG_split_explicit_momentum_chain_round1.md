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
| 1 | `dyn_spg_ts`: split-explicit solve, `uu_b/vv_b(Naa)`, `un_adv/vn_adv`, and `uu/vv(Kmm)` rewrite | `cfgs/DINO/MY_SRC/stpmlf.F90:332`; MLF forcing `cfgs/DINO/MY_SRC/dynspg_ts.F90:316-370`; final rewrite `:1170-1174` | `spg_dump_{zu,zv,ssh}_frc`, loop seed, substep-1, final `puu_b/pvv_b/pssh`, and final `un_adv/vn_adv`; `stp_dump_07_dynspg_*` |
| 2 | second `div_hor`: `hdiv(Nnn)` | `cfgs/DINO/MY_SRC/stpmlf.F90:349-376`; dispatched kernel `src/OCE/DYN/divhor.F90:65` | `seq_dump_hdiv_nnn_kt00005761.bin` through `...5764.bin` |
| 3 | second `dom_qco_r3c`: `r3t/r3u/r3v(Naa)`, `r3f` | `cfgs/DINO/MY_SRC/stpmlf.F90:378-394`; active QCO formulas `src/OCE/DOM/domqco.F90:220-260` | `seq_dump_r3{t,u,v}_aaa_kt*`, `seq_dump_r3f_kt*` |
| 4 | `dyn_zdf`: momentum `uu/vv(Naa)` commit | `cfgs/DINO/MY_SRC/stpmlf.F90:396-409`; split-explicit removal and solve entry `cfgs/DINO/MY_SRC/dynzdf.F90:145-206` | `stp_dump_08_dynzdf_kt*_{u,v}.bin` and the completed ZDF chain receipt |
| 5 | second `wzv`: `ww(Nnn)` | `cfgs/DINO/MY_SRC/stpmlf.F90:411-412`; divergence/integration `cfgs/DINO/MY_SRC/sshwzv.F90:404-426` | `wzv_dump_ww_call2.bin` (call 1 retained as the ordering control) |
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

For every signed field, define

`E = RMS(legoESM - NEMO) / RMS(NEMO)`

over the probe's registered wet mask. `MATCHED` requires every component in a
subrow to satisfy its frozen bar; otherwise the subrow is `DIVERGED`.

| Subrow | Operand/output | Bar | CONFIRM / REFUTE |
|---:|---|---:|---|
| 1.1 | slow forcing `zu_frc`, `zv_frc`, `ssh_frc` | `E <= 1e-15` | CONFIRM exact operand receipt / REFUTE |
| 1.2 | loop seed `sshn_e`, `un_e`, `vn_e` | `E <= 1e-15` | CONFIRM exact entry state / REFUTE |
| 1.3 | first substep `ssh`, `ub`, `vb` | `E <= 1e-12` | CONFIRM accumulated prefix / REFUTE |
| 1.4 | final `puu_b`, `pvv_b`, `pssh`, `un_adv`, `vn_adv` | `E <= 1e-12` | CONFIRM split-explicit output / REFUTE |

The first source-ordered failing subrow localizes row 1. `zu_frc`/`zv_frc`
are operands, not causes: a failure there transfers the next peel to the
vertically averaged explicit momentum RHS and must not be called a
barotropic-integrator defect.

## Controls and stop rule

The wrapper requires CPU, fp64, `LEGOESM_NEMO_E3T=both`, lane `d180`, and the
probe's own registered time levels. It records the inherited zero-shift
alignment scans. Its scorer controls must both fire before an artifact is
admitted: an identical-array control must score zero and a planted `1e-6`
RMS-scale wet-field offset must breach the relevant bar. Any missing field,
failed control, non-finite metric, wrong lane, or nonzero inherited-probe exit
is `INVALID`, not evidence.

## Instrument amendment after invalid attempt 1

Attempt 1 exited before producing an artifact because the inherited probe's
bridge did not carry native T-point latitude, now required by the production
card's `tke_htau_evaluation="nemo_literal"`. No row score was admitted. Before
attempt 2, `spg_substep_chain.py` is amended to resolve the card before the
bridge and pass the same `omega` and `carry_native_lat_deg` condition used by
`kamm_twin_90d.py:889-905`. This is a harness-input repair only; the ordered
rows, arrays, bars, controls, and stop rule above are unchanged.
