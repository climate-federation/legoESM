# Receipt — VORTEX_SMT round 27 (lane round 239): source-order trajectory veto

**Status: HELD.** NEMO's HPG-first source order is locally bit-exact under
production JIT, but it is trajectory-inert and fails Decision 96's net-
improvement criterion: 24 of 50 SMT-4 aggregate rows move, only 11 toward NEMO
and 13 away. Production is restored. The candidate is preserved only as
`manifests/nemo_testcase_l1_vortex_smt_round239_source_order_held.patch`.

Base: `fef682255` (round 238). Candidate measurement commit: `e9bafc987`.
Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round239/`.
Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smt_round27_source_order.md`.

## 1. Compiled statement and one-variable arm

The admitted SMT-4 stage program calls HPG first, VOR second, and vector
advection third at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:328-344`.
The vector dispatcher then calls KEG before ZAD at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90:134-139`.
The compiled z-level HPG routine assigns `Krhs`, rather than accumulating into
an earlier term, at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:314-335`.

The candidate changed one selection at the existing shared implementation:
vector velocity stages used the already-present HPG -> VOR -> KEG -> ZAD
source-order accumulator. No arithmetic, configuration, card, state field, or
public option changed. The production-JIT stage-3 HPG rows become bit-exact:

| boundary | U unequal / max | V unequal / max |
|---|---:|---:|
| HPG | `0 / 0` | `0 / 0` |
| VOR | `21,960 / 2.710505431213761e-20` | `21,731 / 1.355252715606881e-20` |
| ADV / pre-LDF | `21,979 / 4.732338229374326e-13` | `21,739 / 1.620657914366122e-13` |
| post-LDF | `21,950 / 4.732338229374326e-13` | `21,885 / 1.620657914366122e-13` |

The HPG one-ULP plant moves exactly one scored cell, prints
`STATUS PLANT-FIRED`, and exits 1. Thus P1 is **CONFIRMED**: the statement is
locally exact under the production closure, and the control is non-vacuous.

## 2. Certified 50-row trajectory

The candidate keeps all five kt=1 rows AT-BAR and first-over-bar at kt=2 on
T/u/v/ssh. It changes 24 aggregate rows, with 11 toward and 13 away; no row
changes AT-BAR/DEBT status. Every moved row is registered below.

| row | before | candidate | direction |
|---|---:|---:|---|
| kt2 V | `6.826891959729742e-10` | `6.826891951056124e-10` | toward |
| kt3 U | `1.020709123819574e-08` | `1.020709125554298e-08` | away |
| kt3 V | `1.655891325967607e-09` | `1.655891321630798e-09` | toward |
| kt3 SSH | `6.240860606077092e-10` | `6.240851724292895e-10` | toward |
| kt4 U | `2.308118721844332e-08` | `2.308118722191277e-08` | away |
| kt4 V | `3.490460296663722e-09` | `3.490460294928999e-09` | toward |
| kt4 SSH | `8.106746940406140e-10` | `8.106735838175894e-10` | toward |
| kt5 U | `3.639640225316931e-08` | `3.639640225663876e-08` | away |
| kt5 V | `6.094295553881607e-09` | `6.094295562555224e-09` | away |
| kt5 SSH | `1.105926239475252e-09` | `1.105926794586765e-09` | away |
| kt6 U | `5.428826766948336e-08` | `5.428826762784999e-08` | toward |
| kt6 V | `1.141989562948045e-08` | `1.141989563140490e-08` | away |
| kt6 SSH | `1.355555334647818e-09` | `1.355555667714725e-09` | away |
| kt7 U | `1.435806865693334e-07` | `1.435806866734168e-07` | away |
| kt7 V | `2.184259867200639e-08` | `2.184259867143718e-08` | toward |
| kt7 SSH | `3.376272808205960e-09` | `3.376272964331073e-09` | away |
| kt8 V | `4.044365841488970e-08` | `4.044365846606404e-08` | away |
| kt8 SSH | `7.740548139956172e-09` | `7.740547827705946e-09` | toward |
| kt9 U | `6.701961815409885e-07` | `6.701961809858770e-07` | toward |
| kt9 V | `1.093947741229867e-07` | `1.093947741342624e-07` | away |
| kt9 SSH | `1.053818847853230e-08` | `1.053818842172011e-08` | toward |
| kt10 U | `8.309995438229856e-07` | `8.309995440450302e-07` | away |
| kt10 V | `2.098501566084529e-07` | `2.098501565173799e-07` | toward |
| kt10 SSH | `1.297493420343576e-08` | `1.297493601448707e-08` | away |

The registry's missing-row plant removes kt10 V, prints
`STATUS PLANT-FIRED`, and exits 1. The oracle-relative cellwise gate also
fails: 30 violations, maximum worsening 14.75 row-scale oracle ULP, with no
row-status change and no earlier first-over-bar. P2's less-than-one-percent
magnitude prediction is **CONFIRMED**, but P3 is **REFUTED** because a strict
majority does not move toward NEMO. Decision 96 therefore does not admit the
candidate.

## 3. One-hundred-day magnitude

The existing scorer was extended rather than duplicated. Its optional
one-card reference binds both the ten-step sanity check and kt10 cross-check
to the candidate ladder produced on the same commit. Its default immutable
references are unchanged. The control accepts an exact candidate reference,
then changes kt5 T and reports `MISMATCH` with the exact changed tuple.

The candidate's complete daily curve is in
`day100_candidate/round210_scores.json`; the selected checkpoints are:

| day | T RMS K | U RMS m/s | V RMS m/s | SSH RMS m |
|---:|---:|---:|---:|---:|
| 1 | `1.405693974953757e-07` | `6.286792086259285e-08` | `7.431603294900065e-08` | `1.104148578800144e-08` |
| 2 | `5.100120075023583e-07` | `1.065039598445632e-07` | `1.162550943487766e-07` | `2.408124879322319e-08` |
| 5 | `4.424519998380317e-06` | `1.775484908088887e-06` | `1.557389852232555e-06` | `1.787936677422284e-07` |
| 10 | `2.454180679955823e-05` | `7.312516990411982e-06` | `7.914167800890280e-06` | `7.704382890672249e-07` |
| 20 | `7.226803498120785e-05` | `1.608965554380536e-05` | `1.741920379706436e-05` | `3.463151481712305e-06` |
| 30 | `1.450215283520363e-04` | `2.673795366390027e-05` | `2.985170460449299e-05` | `7.380713702873827e-06` |
| 60 | `2.003482609185436e-04` | `2.947599702481983e-05` | `2.982349827610741e-05` | `1.247144122648569e-05` |
| 100 | `2.552708052060162e-04` | `2.834815878571902e-05` | `2.715247913584464e-05` | `1.171711158357981e-05` |

At day 100, T RMS moves from `2.552708052055443e-04` to
`2.552708052060162e-04 K`, a worsening of `4.719532056829401e-16 K` or
`2.36e-06` of the `2e-10-K` run-to-run floor. P4 is **CONFIRMED**: the change
is physically inert at 100 days. That does not rescue P3's failed trajectory
criterion.

## 4. Disposition, scope, and open item

Production was restored byte-for-byte. The surviving code changes only extend
the existing 100-day measurement tool and add its fail-closed test; the held
manifest cannot execute. No package physics, card, configuration, or carried
state change remains. P5 therefore follows its frozen HELD arm: unchanged
cards are not re-certified and no DINO, GYRE-year, tank, or ORCA2 movement is
claimed.

The required read-only Codex review was attempted on the committed candidate,
scorer extension, evidence, restoration, and held manifest. It exited before
reading the diff. Its verdict is quoted verbatim:

> `WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)`
> `Reading additional input from stdin...`
> `Error: failed to initialize in-process app-server client: Read-only file system (os error 30)`

**Independent review unavailable in-sandbox.** No SHIP verdict is inferred.
The candidate is HELD independently by its mechanical trajectory veto.

### OPEN — next round

Do not walk the source-order arm's last-bit VOR remainder. Rank the SMT-4
100-day growth by the existing process-family substitution method. Open a
`dynldf_lev` operand acquisition only if that ranking points back to lateral
momentum diffusion; otherwise follow the largest measured family. ORCA2's
rung-0 uses the same level-Laplacian equations with a file-backed coefficient,
but this round licenses no ORCA2 fold-in.

**DECISION_NEEDED: NONE. ACQUISITION_NEEDED: NONE.**
