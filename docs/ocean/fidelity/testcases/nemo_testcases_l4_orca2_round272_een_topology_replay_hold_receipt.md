# ORCA2 round 272 — EEN topology and replay hold

Date: 2026-10-13

Lane base: `b5374a49a2`

Labels: **independent** and **given NEMO's entry**, measured separately

Verdict: **HELD**. The coast-only hypothesis is refuted, but the offline
source replay misses the live EEN term above the campaign floor. No EEN leaf
is attributed and no model, card, deck, carried-state, initial-state, or
sea-ice change lands.

## 1. Frozen question and correction

The frozen preregistration is
`PREREG_nemo_testcases_l4_orca2_round272.md` at `2fdf6c01a`. Its P1 copied
round 271's private-arm maximum (`1.2863307536453391e-8 m/s2`) instead of the
landed-tree baseline (`1.2863307536456779e-8 m/s2`). P1 is **REFUTED**, not
silently edited; the post-measurement disposition is appended below the
frozen text. The argmax remains `[19,33,3]` under both labels.

The compiled build executes `np_CRV`: combined planetary and relative
circulation at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:762-770`, transports at
`dynvor.f90:784-785`, and triads plus accumulator at `dynvor.f90:790-802`.
The deck resolves `ln_dynvor_msk=.false.`; consequently the `fmask` product
inside the first cited branch is dead on this card. Although `rn_shlat=2`
modifies the stored mask, that inactive multiplication cannot own the
executing residual.

## 2. Passive measurement

The admitted OMT-4 kt=1 frame and stage-term records were read from round
230. The already-certified completed-stage trace passed its state passivity
check. An attempted in-executable EEN callback did not reproduce the live
accumulator and was discarded; the final instrument receives only completed
passive operands and replays them offline. All temporary package hooks were
reverted; `git diff b5374a49a -- packages` is empty.

The two labels are array-identical for every reported row:

| quantity | independent | given NEMO's entry |
|---|---:|---:|
| live after-VOR max vs NEMO | `1.2863307536456779e-8` | `1.2863307536456779e-8` |
| live after-VOR RMS | `1.2191816847794566e-10` | `1.2191816847794566e-10` |
| live after-VOR unequal | `249102/399600` | `249102/399600` |
| offline term replay max vs live | `2.5331686726199837e-9` | `2.5331686726199837e-9` |
| replayed accumulator max vs live | `1.736604536182156e-9` | `1.736604536182156e-9` |

The floor is `2e-10 m/s2`. The replay is therefore not calibrated: P3 is
**REFUTED** and P4 is **NOT REACHED**. In particular, the diagnostic equality
of the already-selected circulation helper is not a NEMO leaf substitution;
no leaf is named from it.

Evidence SHA-256:

- classification JSON: `bbd8d646298333e3b59665f0408f3d806711575dd35ed0380bcec999b07e5150`;
- independent JSON: `1bbfd0ac5be6b6c7f7cb31d8d3120500301ffe885b45583068140ac59d2484ee`;
- given-entry JSON: `11baf7571c1c46d2743e852c44ec55b42a62c2b5e15cb90885ecafadeb8c8c74`;
- both NPZ payloads: `9debe0f1920801a9267907f90e4ba7ab9251a6407f5d0d59286d6c955b7b056d`.

## 3. Topology discriminator

For each of the 100 largest native-U residuals, the committed probe reads the
four V transports in the two adjacent EEN triads and the three neighbouring U
circulation faces at that level. A point is coastal if any of those faces is
closed.

Both labels give **78 coastal and 22 open-ocean points**. The largest open
point is `[20,33,5]`, `4.700021238326384e-9 m/s2`; all seven tested faces are
open. R272-P2 therefore refutes a coast-only lateral-boundary owner. Combined
with the dead compiled mask branch, the next valid walk remains the full
executing EEN statement, not `rn_shlat`/`fmask`.

## 4. Mechanical controls and validation

`nemo_testcase_l4_orca2_round272_een_leaf_split.py` requires both labels,
the landed baseline and argmax, the mixed topology, the dead mask branch,
and—deliberately for this HELD gate—the replay miss above the floor. It emits
`PASS_R272_HELD_REPLAY_UNCALIBRATED`. Its planted false calibration sets one
label's replay maximum to zero and refuses with
`independent: held replay unexpectedly reached the floor`.

The focused round-266/272 controls pass **13/13**. The required single
`tests/ocean/fidelity -n 12` battery reached 99% with no emitted failure, then
stopped making progress at
`test_nemo_testcase_phase3_stage_sweep_gate.py::test_prediction_plant_is_fail_closed`.
That exact ID also failed to terminate in a 15-minute serial isolation; both
runs were interrupted and are registered as an unrelated pre-existing hang.
No round-272 test failed. Citation gate: **PENDING**. Separate read-only
review: **PENDING**.

## OPEN

First make the candidate-side replay source-complete and require its complete
vorticity term and accumulator to reproduce the passive live values below the
`2e-10 m/s2` floor under both labels. The present manual extraction changed
the compilation/association enough to miss by `2.53e-9`; no leaf score from
it is citable. Once calibrated, continue the compiled order from `ff_f` and
relative circulation through `r1_e1e2f`, live F thickness, `zwx/zwy`, the
four triads, final metric, and accumulator. Because 22 of the top 100 points
are fully open, do not start with a coastal mask or fold-only arm. If a
source-complete candidate replay cannot be made passive, acquire NEMO's
self-describing per-leaf stage-2 EEN stream rather than instrumenting the
executable with a materialising callback.

No acquisition or user decision is requested in this round.
