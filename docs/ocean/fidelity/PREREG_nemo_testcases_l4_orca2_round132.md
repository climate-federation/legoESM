# Preregistration — ORCA2 round 132 shared face-thickness merge

Date: 2026-10-03. ORCA2 base: `797a939b7`. Incoming GYRE/VORTEX tip:
`9cd35a16a`. This is the merge-only round ordered by operator note B42. No
configuration, carried state, stabilizer, sea-ice selector, threshold, or
`unmeasured_features` entry may change.

## Statement and scope

The incoming statement replaces legoESM's reference U/V face-thickness alias
to T-cell thickness with each NEMO testcase card's already-attached
NEMO-built `e3u_0` and `e3v_0`. The executed NEMO seamount build constructs
those arrays as the shallower-neighbour minima in
`VORTEX_SMT_R3_OMIP_L1_P3/MY_SRC/usrdef_zgr.F90:225,228,231-232`; the shared
consumer is `nemo_qco_resolved_mesh_operands`. ORCA2 carries the finished
partial-cell face arrays from `domain_cfg`, but the pre-merge qco consumer
aliases them to `e3t_0`. The incoming round-214 receipt measured 18,803 U and
18,300 V operand cells moving, by up to 916.98 m and 949.12 m.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R132-P1 | The merge contains both parents, no half-resolved conflict, and every conflict is listed. Citation maps retain the semantic union. | Both parent SHAs are ancestors; clean index; default citation audit has zero unmapped entries and its real-key plant fires. | **REFUTED**: hold the merge and name the exact conflict or dropped anchor. |
| R132-P2 | The GYRE certified ladder and year reproduce incoming tip `9cd35a16a`: ladder 0 moved rows; day-360 T RMS `5.4077419367442036e-05 K`; day-030/240/360 digests `4e36c106403b495e...`, `8b9cd60475626373...`, `e3e0a068346c7866...`. | Exact ladder comparison and certified eight-day year gate pass within Decisions 43/59. | **REFUTED**: hold with the first moved row/day and direction. |
| R132-P3 | ORCA2 rung-0 independent and rung-7 Decision-52 ladders each move 185/200 rows relative to the round-129 pre-merge baselines, lose zero bit-identical rows, and keep `kt=1 stage1 T` as first non-bit. A majority of RMS rows move toward NEMO. | Comparator reports those counts and unchanged first statement for both ladders. | **REFUTED**: retain the measured count; hold only for an exact-row loss, earlier first-over-bar/first-non-bit boundary, or gate refusal. |
| R132-P4 | Round 129's held complete barotropic boundary/transport arm no longer exposes the same rung-0 kt=10 stage-3 S maximum near `31.3617`; the corrected shared face thickness removes or materially reduces that compensation. | Replayed held arm reduces that row by at least 2x while the baseline shared statement remains active. | **REFUTED** if the row remains within 1% of `31.3617`; keep the arm held and name the surviving tracer association. |
| R132-P5 | The independent rung-0 month advances beyond step 16 before its first non-finite value, because the first measured step-16 boundary is stage-1 transport and the merged statement changes that transport operand. | First non-finite step is greater than 16 or no non-finite occurs through 240 steps. | **REFUTED** if step 16 still fails; report its field/index and do not infer ownership from the merge. |
| R132-P6 | GYRE, both flat VORTEX cards, DINO, LOCK_EXCHANGE, and OVERFLOW satisfy their standing landing gates; the seamount cards reproduce round 214's registered movement. | Every named gate emits its own PASS line; no unregistered exact-row loss. | **REFUTED**: hold with the first red line. |

## Round bar

Merge conflict resolution is reviewed as code. The receipt lists every
conflicted hunk and what was kept from each parent. Measurements use CPU,
fp64/x64/libm, the admitted records, and the existing committed gates. Every
ORCA2 moved row is registered by the comparator. The citation gate must pass
on its default receipt and this receipt, with a shifted-line plant that fires.
One focused battery and one `tests/ocean/fidelity -n 12` battery run only after
the corrected real-process census is zero. A separate read-only Codex review
is attempted. No NEMO acquisition is authorized or needed.
