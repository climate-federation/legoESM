# DINO split-explicit / momentum-commit chain: round-1 result

Date: 2026-08-29. Session:
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Verdict

The first executed row, `dyn_spg_ts`, is **DIVERGED**. The first
source-ordered failing operand is the assembled slow momentum forcing,
constructed before loop initialization at active NEMO
`cfgs/DINO/MY_SRC/dynspg_ts.F90:316-445`:

| Operand | Population | `E=RMS(diff)/RMS(NEMO)` | Correlation | Mean-absolute ratio | Max `abs(diff)/RMS(NEMO)` | Campaign gate |
|---|---:|---:|---:|---:|---:|---|
| `zu_frc` | 9,758 wet U columns | `2.740311743349557e-4` | `0.9999999661960921` | `1.0000909532419107` | `5.491879143984337e-3` | DEBT |
| `zv_frc` | 9,868 wet V columns | `2.939336660178107e-5` | `0.9999999997944569` | `0.9999986895970492` | `8.011401797623365e-4` | DEBT |

The forcing is the first divergent operand fed into the split-explicit
recurrence. This does **not** exonerate the recurrence or the downstream
`uu/vv(Kmm)` rewrite at `dynspg_ts.F90:1170-1174`; both remain independently
unmeasured without a held-forcing counterfactual. It is also not a root-cause
claim. The next operand peel is the full forcing assembly: vertically averaged
explicit momentum RHS (`:316-353`), Coriolis removal (`:358-370`), bottom drag
(`:372-383`), and centered wind (`:423-445`).

The inherited probe also reports a non-gating, magnitude-only diagnostic:
the U forcing-difference norm scaled through one `rDt_e` is
`1.1018e-5`, versus first-substep U `E=1.1009e-5` (ratio `1.001`); V gives
`1.633`. These scalar norms discard vector direction and cancellation, so they
assign no causal ownership and do not prove recurrence fidelity.

The lateral admission fact remains intact: production tracer-entry `uu(Kmm)`
diverged 9,758/9,758 (correlation `0.9845023189`, RMS ratio
`1.0296145800`). This round establishes an earlier upstream divergence.

## Ordered row result

| Row | Active call/output | Executed NEMO evidence | Round-1 status |
|---:|---|---|---|
| 1 | `dyn_spg_ts`: `uu_b/vv_b(Naa)`, `un_adv/vn_adv`, `uu/vv(Kmm)` rewrite | `cfgs/DINO/MY_SRC/stpmlf.F90:332`; complete MLF forcing `cfgs/DINO/MY_SRC/dynspg_ts.F90:316-445`; rewrite `:1170-1174` | **DIVERGED at `zu_frc/zv_frc`** |
| 2 | second `div_hor`: `hdiv(Nnn)` | `cfgs/DINO/MY_SRC/stpmlf.F90:349-376`; active MLF kernel `src/OCE/DYN/divhor.F90:151-197` | ORDERED-BLOCKED |
| 3 | second `dom_qco_r3c`: `r3t/r3u/r3v(Naa)`, `r3f` | `cfgs/DINO/MY_SRC/stpmlf.F90:378-394`; active MLF formulas `src/OCE/DOM/domqco.F90:140-186` | ORDERED-BLOCKED |
| 4 | `dyn_zdf`: momentum `uu/vv(Naa)` commit | `cfgs/DINO/MY_SRC/stpmlf.F90:396-409`; active vector arm/removal `cfgs/DINO/MY_SRC/dynzdf.F90:137-141,166-178` | ORDERED-BLOCKED |
| 5 | second `wzv`: `ww(Nnn)` | `cfgs/DINO/MY_SRC/stpmlf.F90:411-412`; active MLF integration `cfgs/DINO/MY_SRC/sshwzv.F90:168-348` | ORDERED-BLOCKED |
| 6 | `mlf_baro_corr`: restore after-level barotropic mean | `cfgs/DINO/MY_SRC/stpmlf.F90:578`; correction `:709-765` | ORDERED-BLOCKED |

The basin preregistration's alignment work is reused rather than repeated:
the temporary Kmm transport-mean installation is not an eta candidate because
`dyn_zdf` constructs Naa and `mlf_baro_corr` later restores the momentum mean.
The ZDF climate refutation remains unchanged; this round makes no MLD or
climate ownership claim.

## Existing-dump measurements

The wrapper ran one fresh CPU/fp64 legoESM production step and barotropic
replays against existing `RUN_SEQDUMP_D180_1R` files at `kt=5761`. There was
no new NEMO integration, build, writer, SLOT block, MPI process, or GPU use.

| Subrow | Existing evidence | Campaign result |
|---:|---|---|
| 1.1 | `spg_dump_zu_frc.bin`, `spg_dump_zv_frc.bin` | DEBT / DEBT; DIVERGED |
| 1.2 | loop-seed SSH/U/V dumps (`dynspg_ts.F90:572-573`) | AT BAR / NEAR-CLASS / NEAR-CLASS; DIVERGED under the exact class gate (`E=0`, `2.296259689e-16`, `2.288824335e-16`) |
| 1.3 | substep-1 SSH/U/V dumps | DEBT / DEBT / DEBT; DIVERGED (`E=6.204876300e-7`, `1.100939578e-5`, `5.706172277e-7`) |
| 1.4 | final `puu_b/pvv_b/pssh/un_adv/vn_adv` dumps | all DEBT; DIVERGED (`E=4.372119219e-5`, `4.966985718e-5`, `1.859773733e-5`, `2.732316730e-5`, `2.737515993e-5`) |

`ssh_frc` is explicitly UNMEASURED: its NEMO dump exists, but this recipe has
no legoESM `F_slow_eta` while freshwater closure is inactive.

## Controls, provenance, and attempt audit

- CPU, JAX fp64, `LEGOESM_NEMO_E3T=both`, lane `d180`, exact U/V/T
  populations, and finite wet fields held.
- `fidelity_bar_gate.classify` supplied all verdicts. The receipt records its
  correlation, mean-absolute-ratio, and arithmetic-class per-element axes.
- Clean Git commit before/after:
  `69446f08f9bad339a3566a4e8ca391926d77365e`.
- Identical arrays classified `AT BAR`; the planted `1e-6` RMS-scale offset
  traversed the same `fidelity_bar_gate.classify` path and classified `DEBT`;
  perturbing the actual legoESM binding also changed its score.
- Actual `zu_frc` alignment scan selected `(dj,di)=(0,0)` with
  `E=2.740311743e-4`; the next-best shift was `(1,0)` with `E=0.1287265`.
- Receipt hashes all 18 dumps read by the inherited probe, restart, mesh,
  NEMO executable, `ocean.output`, resolved `output.namelist.dyn`, both
  namelists, five production/scorer modules, wrapper/probe, and active NEMO
  `stpmlf`/`dynspg_ts` sources. It also stamps the effective replaced DINO
  config, derived model/barotropic settings, Python/JAX/jaxlib/NumPy versions,
  platform, and CPU device.
- Attempts 1 and 2 were INVALID before artifact admission (native-latitude
  bridge precondition; unpaired `ssh_frc`). Attempt 3's receipt was withdrawn
  after two adversarial HOLD reviews. The first attempt at the hardened commit
  was rejected by its clean-tree gate before calculation. The first re-review
  retained HOLD on classifier-control and effective-config provenance; the
  mechanism re-review then retained HOLD on stale scalar-norm ownership and
  accumulation claims in the inherited probe. The final run after retracting
  both claims exited zero, labeled both norm comparisons non-causal, and
  printed term attribution UNMEASURED.

Machine receipt:
`docs/ocean/fidelity/dino_split_explicit_momentum_chain_round1_artifact.json`.
