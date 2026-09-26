# Split-explicit / momentum-commit chain: round 16

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Measured package commit
`50e2095d13d`.

## Association rows 9.4--9.6

The existing day-180 dumps close all three multiplication-association rows:

| Row | NEMO expression | Result |
|---:|---|---|
| 9.4 | `(e2u * ua_e) * zhup2_e` | unique bit-exact U transport |
| 9.5 | `(e1v * va_e) * zhvp2_e` | unique bit-exact V transport |
| 9.6 | `(dU + dV) * r1_e1e2t` | zero mismatches on 9,920 wet T points |

For row 9.6, divide-after-sum differs at 2,655/9,920 points. These are the
executed source expressions at `dynspg_ts.F90:698-704,722-724`. The production
path now preserves these associations under
`barotropic_continuity_evaluation="nemo_literal"`.

Adversarial review caught and closed a selector-routing defect before any
climate arm: the first implementation let the selector alter face-depth/drag
physics while both values still took literal divergence. Production now keys
H_u/H_v only on `barotropic_face_depth` and keys generic versus literal
divergence only on `barotropic_continuity_evaluation`. A one-substep test with
nonzero drag and `g=0` requires both arms to return bit-identical transports
and velocities while SSH is bit-distinct; the combined continuity/drag/
partial-cell set passes 40/40 on CPU/fp64.

At the measured production commit, the hardened production acceptance reports
rows 9.1--9.7 all `AT BAR` with normalized error and maximum error both exactly
zero. It binds checkout-local production modules, the exact mesh/restart and
dump hashes, CPU/fp64, session, and controls. The production acceptance calls
the face-depth, metric-transport, and divergence helpers; its SSH value is the
registered formula applied to those production operands. The independent full
recurrence closes the first-substep SSH output.

## Ordered chain state

The hardened recurrence at this commit remains stopped at row 1.2 under the frozen
pointwise `1e-15` bar:

| Row | Operand | Status | E | max error / NEMO RMS |
|---:|---|---|---:|---:|
| 1.2 | `sshn_e` | AT BAR | 0 | 0 |
| 1.2 | `un_e` | NEAR-CLASS | 2.29625968916112e-16 | 5.57431633372995e-15 |
| 1.2 | `vn_e` | NEAR-CLASS | 2.223767115037694e-16 | 4.26422750960742e-15 |

The round-18 existing-dump peel now owns this roundoff residue exactly. At an
MLF restart, `istate.F90:149-155` reconstructs the barotropic Kbb seed with
BEFORE (`sshb`) live face thickness and source-ordered vertical accumulation;
that expression is bit-exact on every registered U/V face. The former
`dynatf_qco` attribution is corrected for this restart boundary. Production
still uses a fused reduction, so row 1.2 remains an ordered implementation stop
until the designed `barotropic_seed_evaluation="nemo_literal"` selector lands
and the recurrence is replayed. See the round-18 result.

Rows after the stop retain targeting evidence only:

| Row | Operand/output | Targeting E | Registry disposition |
|---:|---|---:|---|
| 1.3 | `ssh_substep1` with exact seed | 1.90418589978210e-20 | ordered-blocked |
| 1.3 | `ub_substep1` with exact seed | 4.18260787855496e-7 | ordered-blocked |
| 1.3 | `vb_substep1` with exact seed | 1.05114180302868e-6 | ordered-blocked |
| 1.4 | `puu_b_final` | 6.22167444653010e-6 | ordered-blocked |
| 1.4 | `pvv_b_final` | 1.10928061109024e-5 | ordered-blocked |
| 1.4 | `pssh_final` | 4.20792959287899e-6 | ordered-blocked |
| 1.4 | `un_adv_final` | 3.36866897283752e-6 | ordered-blocked |
| 1.4 | `vn_adv_final` | 9.17972945992041e-6 | ordered-blocked |
| 2--6 | all registered rows | not crossed | ordered-blocked |

If row 1.2 clears, row 1.3 resumes at the first velocity-update operands in
source order: back-interpolated SSH and pressure gradient
(`dynspg_ts.F90:766-780`), in-loop Coriolis (`:783-805`), explicit bottom
stress (`:818-825`), then final vector association (`:838-850`). Existing
substep-1 Coriolis dumps and exact SSH dumps cover the first comparisons;
bottom-stress dumps must be re-inventoried before instrumentation.

## Structural owner and climate release

NEMO's V-point latitude and zonal metric are explicit:
`zvj=j-nn_jeq_s+0.5`, `gphiv=asin(tanh(...zvj...))`, and
`e1v=R*cos(gphiv)*dlon` (`usrdef_hgr.F90:98,108,113`). The former legoESM
path averaged adjacent tracer latitudes before taking the cosine. The corrected
V-face metric plus literal continuity path is now production-selected and
passes the hardened day-180 operand acceptance exactly.

The climate intervention is therefore released independently of the ordered
registry stop. Its frozen question is:

> Does the NEMO-faithful V-face Mercator metric plus literal QCO continuity
> materially reduce the frozen day-360 southern-basin deficit (-0.951912 Sv)
> and the five-day wall flicker?

The committed preregistration uses a 2x2 metric x association design so the
headline contrast is not confounded and the main effects plus interaction are
reported. No GPU arm was run in this round.

## Bound artifacts

- acceptance: SHA-256
  `994acb1d382dcb221636895cebbc420c5e43da2e6ad710109658f54dc06a46a0`
- recurrence: SHA-256
  `b2dcb7e151436b36ade81f69ad6ed43ec56d25cdcb76f2e289ba26168a167935`
- combined adjudication: SHA-256
  `2ce41f04fc0620ba3e049b4b27fd399ee1ab1ef87452f5f06a0df7e54ac9a749`

The earlier round-13/14 artifacts are unbound diagnostics superseded by these
hardened receipts. No NEMO process, GPU, `mpirun`, push, or new instrumentation
was used.

## Post-run climate receipt adjudication (2026-08-29)

The four day-360 basin arms completed at clean producer
`e013e95ca54957a4454878ed7118e623da0a19ba`; the scorer stopped before opening
the climate statistic. Exact receipt replay confirms that the new stress hash
`cad9b34958cba58812f1dce2c6c441c8fd6b5f2733c20b91065f4d60507592b7`
is wholly attributable to the already-owned `nemo_literal` wind-profile
arithmetic. Re-selecting only `factored_smoothstep` reproduces the frozen
historical hash `b6a08b8395017c8e3f8df0b8b13eefa75fdfe7be3770d788beaaf1ca514127ae`.

The actual STOP was the scorer expecting the descriptive string
`BRIDGED_BEFORE` instead of the harness's canonical `twin_start_mode=bridged`.
Both factorial scorers now use the canonical stamp and additionally bind the
stress content hash to `dino_wind_profile_evaluation`, with red-capable plants
for the start label and content hash. Blocks 1--2 and their artifacts are not
rerun. The scoring blocks use a separate clean adjudication checkout so the
producer checkout remains frozen at the commit that made the artifacts.

The corrected basin scorer exits zero on those four retained artifacts. The
M0A0 gap is `-0.951979116371497 Sv`; M1A1 is
`-0.9388829260453218 Sv`, so the combined intervention improves the gap by
`0.013096190326175261 Sv` (`1.37568042207605%`). This is below the registered
`2F=0.12347312432091852 Sv` resolution threshold and is therefore
**`UNRESOLVED/FLOOR`**, not a refutation or confirmation. All frozen
compensation metrics pass. The metric alone at generic association worsens the
gap by `-0.02496024897123128 Sv`; the interaction is
`+0.03865013957391383 Sv`, leaving every registered component bounded at the
same floor. The score artifact SHA-256 is
`d1974563f01cfec03e317dbeba6de1210935b184d8bc8a1dd5fa34984a6d30a6`.
The independently preregistered five-day wall factorial is now complete. Its
artifact SHA-256 is
`c9be1e08aac6b501dca2d8634ee889df7e7f90e930c70d0c345af8cdc2819343`.
All four arms lie in the narrow ranges ratio `1.63496--1.65118` and wall share
`0.17020--0.17308`; continuity alone changes either statistic by less than
`1e-9`, while the V-face metric changes ratio by about `-0.0162` and wall share
by `+0.00288`.

The formal wall verdict is **`INVALID_CONTROL`**. The registered control band
was the earlier uncorrected-bridge epoch (`[2.60,3.18]`, `[0.38,0.59]`), while
the corrected-T basin lane had already moved to ratio `1.1607251697830108`,
wall share `0.11941867158491266`. The current M0A0 control instead measures
`1.6511846170859847`, `0.17020235431211447`. Therefore no registered wall
ownership verdict exists: all four metric/association effects are descriptive,
and any earlier wording that the current-epoch wall question was confirmed or
refuted by this factorial is withdrawn.

The basin result is a real registered null measurement, not an invalid
control. It rejects the working prediction that this structural pair would
produce a visible, material movement of the `-0.95 Sv` deficit: the observed
response is only 1.376%. Its formal label nevertheless remains
**`UNRESOLVED/FLOOR`**, because the decision tree checks `|delta|<=2F` before
the <=2% refutation branch. It must not be relabeled formal `REFUTED`; the
correct claim is that material ownership was not confirmed and the measured
pair does not explain the deficit at the registered resolution.

The wall control's rise from the corrected-T epoch (`1.1607/0.1194`) to the
current epoch (`1.6512/0.1702`) opened a separate attribution lane. Two stopped
partitions proved that its 16 selectors do not form an unconstrained cube: the
matrix, recurrence, Langmuir, entry-N2, and carried-slope dependencies leave
14,336 legal subsets. Those independence claims are retracted in the committed
preregistration; neither stopped arm produced an admitted NPZ.

The replacement legal-lattice bisect is complete. Its score artifact SHA-256
is `05dc4c7edefc8b6e705026812b3a4d6ae90098715b60a940094a3de421c45a16`.
The registered disposition is
**`LOCALIZED_TO_TKE_CORE_AT_FAITHFUL_SLOPE_N2`**: reverting the TKE matrix,
solver recurrence, and Langmuir association together restores ratio
`1.163579045567568` and wall share `0.11693494406953533`, closing `99.42%` and
`104.89%` of the epoch shifts and entering both historical bands. Every
primitive non-core contrast is `BOUNDED_SMALL`; the core x slope-N2 interaction
is only `-0.11%/-1.55%` of the two shifts. This is faithful-but-worse: the
legacy core supplied a compensating error that masked a remaining wall error.
The group arm does not yet distinguish matrix from recurrence from Langmuir or
name the compensated counterpart. The registered next discriminator is the
legal solver-only / Langmuir-only / both-legacy peel at faithful matrix, with
the existing full-core arm furnishing the conditional matrix contrast. Full
receipts and interpretation are in
`dino_zdf_wall_epoch_group_bisect_result.md`.

## COMPLETE-PR boundary

The momentum lane's finished PR unit, #1696, remains ready: the NEMO V-face metric and
literal QCO continuity production fixes; exact day-180 row-9.1--9.7 acceptance
and recurrence receipts; selector isolation/non-confound tests; corrected-T
merge-order dependency; and both completed climate outcomes with the formal
qualifications above. The PR must say COMPLETE and must not claim either basin
or metric/continuity wall-climate ownership. Its former `INVALID_CONTROL`
qualification may cross-reference the separately registered lattice result:
the epoch drift belongs to the pre-existing ZDF TKE-core selector group, not to
#1696's intervention. No TKE-core code or follow-up peel belongs in #1696.

The following remain open and are not merge conditions for that finished
unit: the ordered chain stop at row 1.2; rows 1.3--6; the unknown owner of the
`-0.95 Sv` basin deficit; and the still-unidentified feeder that the legacy
TKE matrix compensated at the walls. The follow-up sub-peel has now localized
the wall-epoch rise to the matrix conditional on legacy solver and Langmuir;
its WALL-TKE-MATRIX-FEEDER operand ladder belongs on a new follow-up branch/PR,
not as a tail appended after #1696 opens.
