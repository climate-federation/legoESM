# DINO split-explicit / momentum-commit chain: round-1 result

Date: 2026-08-29. Session:
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Verdict

The first executed row, `dyn_spg_ts`, is **DIVERGED**. The first
source-ordered failing operand is the slow momentum forcing constructed at
active NEMO `cfgs/DINO/MY_SRC/dynspg_ts.F90:316-370`:

| Operand | Population | `E = RMS(diff)/RMS(NEMO)` | Correlation | RMS ratio | Bar | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `zu_frc` | 9,758 wet U columns | `2.740311743349557e-4` | `0.9999999661960921` | `1.0000976381607058` | `1e-15` | DIVERGED |
| `zv_frc` | 9,868 wet V columns | `2.939336660178107e-5` | `0.9999999997944569` | `0.9999785828768776` | `1e-15` | DIVERGED |

This localizes row 1 to the operand fed into the split-explicit recurrence,
not to the recurrence itself and not to the downstream
`uu/vv(Kmm)` rewrite at `dynspg_ts.F90:1170-1174`. It is not yet a root-cause
claim: the next peel is the vertically averaged explicit momentum RHS and its
Coriolis/drag subtraction. The inherited probe's non-gating arithmetic is
consistent with this localization: the U forcing difference propagated
through one `rDt_e` predicts `1.1018e-5`, versus the independently measured
first-substep U error `1.1009e-5` (ratio `1.001`). The analogous V ratio is
`1.633`; neither ratio is used as the verdict gate.

The lateral-chain admission fact remains intact: its production tracer-entry
`uu(Kmm)` comparison diverged 9,758/9,758 (correlation `0.9845023189`, RMS
ratio `1.0296145800`). This round places an earlier divergence upstream of
that handoff.

## Ordered row result

| Row | Active call/output | NEMO execution evidence | Round-1 status |
|---:|---|---|---|
| 1 | `dyn_spg_ts`: `uu_b/vv_b(Naa)`, `un_adv/vn_adv`, `uu/vv(Kmm)` rewrite | `cfgs/DINO/MY_SRC/stpmlf.F90:332`; forcing `cfgs/DINO/MY_SRC/dynspg_ts.F90:316-370`; rewrite `:1170-1174` | **DIVERGED at `zu_frc/zv_frc`** |
| 2 | second `div_hor`: `hdiv(Nnn)` | `cfgs/DINO/MY_SRC/stpmlf.F90:349-376`; `src/OCE/DYN/divhor.F90:65` | ORDERED-BLOCKED |
| 3 | second `dom_qco_r3c`: `r3t/r3u/r3v(Naa)`, `r3f` | `cfgs/DINO/MY_SRC/stpmlf.F90:378-394`; `src/OCE/DOM/domqco.F90:220-260` | ORDERED-BLOCKED |
| 4 | `dyn_zdf`: momentum `uu/vv(Naa)` commit | `cfgs/DINO/MY_SRC/stpmlf.F90:396-409`; `cfgs/DINO/MY_SRC/dynzdf.F90:145-206` | ORDERED-BLOCKED |
| 5 | second `wzv`: `ww(Nnn)` | `cfgs/DINO/MY_SRC/stpmlf.F90:411-412`; `cfgs/DINO/MY_SRC/sshwzv.F90:404-426` | ORDERED-BLOCKED |
| 6 | `mlf_baro_corr`: restore after-level barotropic mean | `cfgs/DINO/MY_SRC/stpmlf.F90:578`; correction `:709-765` | ORDERED-BLOCKED |

The basin preregistration's alignment work is reused rather than repeated:
the temporary Kmm transport-mean installation is not an eta candidate because
`dyn_zdf` constructs Naa and `mlf_baro_corr` later restores the momentum mean.
The ZDF climate refutation likewise remains unchanged; this round makes no MLD
ownership or climate claim.

## Existing-dump measurements

All scores came from `RUN_SEQDUMP_D180_1R` at `kt=5761` and the existing
committed production probe. No NEMO build, new writer, SLOT block, MPI process,
GPU, or new integration was used.

| Subrow | Existing evidence | Result |
|---:|---|---|
| 1.1 | `spg_dump_zu_frc.bin`, `spg_dump_zv_frc.bin` | DIVERGED |
| 1.2 | `spg_dump_sshn_e_init.bin`, `spg_dump_un_e_init.bin`, `spg_dump_vn_e_init.bin` | MATCHED: `0`, `2.296259689e-16`, `2.288824335e-16` |
| 1.3 | substep-1 SSH/U/V dumps | DIVERGED: `6.204876300e-7`, `1.100939578e-5`, `5.706172277e-7` |
| 1.4 | final `puu_b/pvv_b/pssh/un_adv/vn_adv` dumps | DIVERGED: `4.372119219e-5`, `4.966985718e-5`, `1.859773733e-5`, `2.732316730e-5`, `2.737515993e-5` |

`ssh_frc` is explicitly UNMEASURED: its NEMO dump exists, but this recipe has
no legoESM `F_slow_eta` while freshwater closure is inactive.

## Controls and attempt audit

- CPU, JAX fp64, `LEGOESM_NEMO_E3T=both`, lane `d180` all held.
- Identical-array control scored exactly zero.
- Planted `1e-6` RMS-scale offset scored `1.0000000000008017e-6` and breached
  the `1e-15` bar.
- Attempt 1 was INVALID before scoring because the inherited bridge omitted
  native latitude required by the current `nemo_literal` TKE card. The probe
  was repaired to use the production harness condition.
- Attempt 2 was INVALID after calculation but before artifact admission
  because it required unpaired `ssh_frc`. The preregistration records the
  correction. No artifact from either invalid attempt was admitted.
- Attempt 3 exited zero and produced the committed receipt.

Machine receipt:
`docs/ocean/fidelity/dino_split_explicit_momentum_chain_round1_artifact.json`.
The receipt contains SHA-256 for all 14 read dumps, the wrapper, inherited
probe, and active NEMO `stpmlf`/`dynspg_ts` sources.
