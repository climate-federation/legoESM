# ORCA2 lane — preregistration for merging the GYRE lane, 2026-09-25

Committed BEFORE the merge is run.  Every prediction below is frozen; a
prediction that fails is recorded REFUTED and kept, never rewritten.

## Scope and provenance

Merge `github/fidelity/nemo-testcases-l2-gyre-codex2` (`d3631f884`, which
already contains GitHub `main` as of 2026-09-25 through the GYRE lane's own
merge commit `839e988c8`) into the ORCA2 lane tip `b03f78bb5`.  Merge base
`dda3f3257` — the GYRE lane tip the ORCA2 lane was cut from.  ORCA2 is 186
commits ahead of that base, GYRE 701.

Ordinary merge: no squash, no rebase, two parents.  Resolution is decided from
NEMO's compiled source, never from preference; each hunk's deciding statement
is named in the receipt.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/merge_gyre_2026-09-25/`.

## Expected textual conflicts — the operator's dry run named five

| # | file | why both lanes touched it |
|---|---|---|
| 1 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_decision36_nemo_face_shear_receipt.md` | receipt prose carrying citations into the two lat-lon C-grid modules; both lanes re-anchored them |
| 2 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` | same |
| 3 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` | ORCA2 landed runoff heat/water and the zero-width-wall reciprocal; GYRE landed the second continuity solve, LDF routings and the TKE shear routing arm |
| 4 | `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` | same two lanes, same NEMO paths |
| 5 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` | both lanes added `CITATION_MAP` entries and both re-anchored the shared ones |

If the conflict set differs from these five, the difference is reported and
each extra or missing file is explained before anything is resolved.

## Files edited on BOTH sides (auto-merge candidates, audited anyway)

Twelve files differ from the merge base on both sides; the five above plus:

- `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_round67_ldf_order_receipt.md`
- `packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py`
- `packages/ocean/legoesm/ocean/eos.py`
- `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py`
- `packages/ocean/legoesm/ocean/freshwater.py`
- `packages/ocean/legoesm/ocean/state.py`
- `packages/ocean/legoesm/ocean/vertical.py`

Every auto-merged hunk in the seven executable files is read and given an
execution verdict against the resolved ORCA2 card and the resolved GYRE card.

## The two invariants this merge must prove

### (i) ORCA2's ten-step ladder is unchanged, to the byte

`nemo_testcase_l4_orca2_round1_ladder_gate.py --max-step 10` on the merged
tree reproduces round 20's `LADDER_MEASURED` result with every content key
equal to round 20's `ladder.json` (only the `worktree` provenance block may
differ).  In particular the second continuity solve stays OFF for ORCA2: the
card sets `nemo_stage_momentum_wzv_split=False` EXPLICITLY, an unset card
raises, and Decision 58 turns it on later in its own measured round — NOT
here.

Falsifier: any content key differing; any ladder row moved; the field resolving
to anything but `False` for `orca2_vector_een_c2`; the gate refusing.

### (ii) GYRE's certified trajectory equals the GYRE lane's post-merge numbers

The ORCA2 lane has always kept GYRE byte-identical; its receipts state that as
day-30 snapshot digest `14a7e64b4512860e`, which is the PRE-round-163 value.
Round 163 landed the second continuity solve for GYRE and this merge brings it
across, so the ORCA2 lane's "GYRE unchanged" gate is RE-BASED here onto the
GYRE lane's own post-merge numbers, and the receipt says so explicitly.

Target values, from the GYRE lane's merge-main receipt of 2026-09-25:

| quantity | value |
|---|---|
| ladder document digest, provenance stripped | `cf06a8fc7d0e90f2` |
| day 30 T rms | `6.572574374770603e-05` K |
| day 240 T rms | `1.644836070117868e-02` K |
| day 360 T rms | `1.1225660018551306e-02` K |
| day030 / day240 / day360 snapshot sha256 | `a66143733bcc9e4e` / `0d4f16d0c51da705` / `c6b7e1523b9a6dbb` |
| certified 70-row compare | `rows=70 max_worsening_ulps=0`, first-over-bar `{T,S,u,v,ssh} kt 3` |

Falsifier: any of those differing on the merged tree.

## Predictions

| ID | prediction | confirms | refutes |
|---|---|---|---|
| M-P1 | Exactly the five preregistered files conflict, no more and no fewer. | the conflict list matches | any other file conflicting |
| M-P2 | Every conflict in the two model files is a UNION that keeps both lanes' NEMO identities; no hunk is dropped, no definition duplicated, no default moved. | per-hunk deciding statement named and the review agrees | any hunk decided by preference, or a dropped/duplicated definition |
| M-P3 | The ORCA2 ten-step ladder is byte-identical to round 20's (invariant i), with `nemo_stage_momentum_wzv_split=False` resolved for the ORCA2 card. | all content keys equal | any moved row |
| M-P4 | GYRE reproduces the GYRE lane's post-merge numbers exactly (invariant ii). | all six quantities equal | any differing |
| M-P5 | The citation gate passes on the union of both lanes' `CITATION_MAP` entries after a rigid re-anchor against the merged line numbers, and every self-test plant fires. | `status PASS`, 0 failures, 0 unmapped, all plants fired | a failure, an unmapped citation, or a silent plant |
| M-P6 | The DINO, lock-exchange, overflow and generic-GYRE card gates are unchanged: the ORCA2 lane's last count was 169, the GYRE lane's 170; the merged tree gives the GYRE lane's count. | count reproduced | any card gate red |
| M-P7 | Both push gates (ORCA2's five files, GYRE's six) pass on the merged tree. | pytest's own summary lines quoted | any red |
| M-P8 | The broad ocean-fidelity battery shows no failure absent from BOTH lanes' known-red lists. | every failing ID matched to a lane's known red | any new red, which is a merge defect |

## Protocol

Both measurement arms run from CLEAN, COMMITTED worktrees with no
`LEGOESM_GATE_ALLOW_DIRTY` escape.  One pytest battery at a time
(`ps -eo comm= | grep -x pytest` empty before each).  NEMO is never run; NEMO
numbers come only from the recorded acquisitions the receipts name.  Two
independent reviews — a Claude code-reviewer subagent and
`codex exec --sandbox read-only` — on the merge commit; every finding closed or
registered.

## Choices

ASKED: perform the merge the operator ordered, resolve from NEMO's compiled
source, keep both lanes' NEMO identities, keep ORCA2's second continuity solve
OFF.

UNASKED: none.  Decisions 54, 57 and 58 remain pending and untouched by this
merge.
