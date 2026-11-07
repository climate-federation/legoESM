# Preregistration — ORCA2 round 109 EEN three-build identity

Date: 2026-10-02. Base: `1ab7699e61a8876d6eb076cbf3dd5fdaa311e64e`.
Scope is instrumentation integrity on hierarchy rung 0. Every later ocean
number is **independent** because rung 0 starts from NEMO's own from-rest
state. No model physics, card field, configuration value, carried state,
threshold, sea-ice selector, or `unmeasured_features` entry may change before
the record is proven observationally passive.

## Observed refusal

The operator-run round-108 target reaches `STOP 0`, but its admission checker
refuses before restart comparison because the per-level record's terminal
`ffu_nw` differs bitwise from the admitted round-105 accumulator. This is not
an ocean result. Per the round order, the candidates are: (a) observational
perturbation by the per-level patch, (b) different decks/build inputs, or (c)
the round-105 record itself not being additions-only.

The committed round-107 patch adds seven full three-dimensional recording
arrays, stores every recurrence operand, and separately evaluates the product
written to `term_nw`. These additions make (a) a live possibility but do not
establish it. The build manifests, preprocessed sources, and terminal restarts
must discriminate the candidates mechanically.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R109-P1 | The base rung-0, round-105 accumulator, and round-108 per-level runs use the same resolved deck and numerical build inputs. | Their copied namelists, deck/input manifests, and relevant compiler/toolchain identities are byte-identical or differ only in explicitly enumerated recorder source/module entries. | Any numerical namelist, input, compiler, or unrelated source difference confirms candidate **(b)**; stop and reacquire from a content-identical base. |
| R109-P2 | The round-105 recorder is observationally passive relative to the admitted base rung-0 record. | Every rank's kt=10 restart variables and raw files are byte-identical between the base record and round 105. | Any difference confirms candidate **(c)**; withdraw round 105 and repair its recorder first. |
| R109-P3 | The round-108 per-level build is not observationally passive. | At least one rank's kt=10 restart differs from both the base and round-105 restart, while base and round 105 remain identical. | If all three restart sets are identical, the checker association or record extraction is wrong; repair the checker, not the writer. |
| R109-P4 | The perturbation is introduced by the round-107 patch rather than a hidden source/deck delta. | A diff of the round-105 and round-107 compiled `dynspg_ts`/recorder sources, after enumerating all additions, finds no deck or inherited numerical statement change; a minimal replacement that records already-computed scalars without re-evaluating the product restores kt=10 restart identity and the inherited final accumulator. | Any non-instrument numerical source delta confirms candidate **(b)**; if the minimal recorder still moves restarts, continue reducing it before acquisition. |
| R109-P5 | No signed-zero operand statement can be named from the refused record. | Admission remains blocked until the repaired build is bit-identical at kt=10 and its self-describing record plus all plants pass. | Quoting the refused record is a process failure; retract it immediately. |

## Repair bar

The offending instrument is repaired under a new target and run directory.
The repaired writer must be additions-only, use absolute pre-created per-rank
paths, and parse its own field headers. Before a physics walk it must prove:

1. base, round-105, and repaired kt=10 restarts are bit-identical on both
   ranks (raw-file identity where metadata permits, otherwise every restart
   variable with signed-zero-sensitive array equality);
2. the inherited round-105 accumulator record is byte-identical to its
   admitted source;
3. the repaired per-level terminal accumulator is bit-identical to round 105;
4. every header, field, recurrence, rank-coverage, and restart plant fires.

If the sandbox cannot run MPI, the round ends `STOPPED_FOR_RECORD` with a
fresh fail-closed `run.sh`; no physics statement lands.

## Frozen per-level walk after admission

The existing record may be walked only after the repair bar passes. The
source-ordered comparison is `zpvo_nw`, live U thickness, live V thickness,
neighbor V mask, stored product, accumulator before, accumulator after.

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R109-P6 | The first remaining signed-zero difference is already in NEMO's three-term `zpvo_nw` sum. | `zpvo_nw` has at least one unequal bit against the literal legoESM replay; report signed-zero and magnitude counts before inspecting later fields. | If `zpvo_nw` is bit-exact, mark **REFUTED** and advance to the first later unequal operand without skipping fields. |
| R109-P7 | Both live thickness operands and the neighboring V mask are bit-exact. | Zero unequal bits for `e3u_live`, `e3v_live`, and `neighbor_mask` over the rank-complete owned domain. | The first unequal one owns the walk; stop there and do not attribute the product or accumulator. |
| R109-P8 | NEMO's recorded `acc_after` is bitwise equal to recorded `acc_before + term_nw` at every executed level. | Zero recurrence-bit differences on both ranks. | Any difference makes the instrument or arithmetic association insufficient; stop without a model statement. |

### Pre-measurement correction after a refused probe

The first attempted per-level invocation was invalid and is not admitted: it
compared legoESM's constructed operands at all 30 physical levels against
NEMO recording arrays initialized to zero and written only inside
`DO jk=1,mbku`. Its large counts are retracted before interpretation. The
correct source order begins with the loop bound `mbku`, which is predicted
bit-exact, and all later fields are compared only on executed levels; both
arms are explicitly zero outside the loop. A non-bit `mbku` refutes this
correction and owns the walk before `zpvo_nw`.
