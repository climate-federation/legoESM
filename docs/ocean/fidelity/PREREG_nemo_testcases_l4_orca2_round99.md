# Preregistration — ORCA2 round 99 frozen EEN coefficient admission

Date: 2026-10-02. Base: `0b9e9d07b31c443de73b535fbb3d085f1c931ccd`.
Scope is ocean only. Every model number is labelled **independent** because
hierarchy rung 0 starts from NEMO's own from-rest state.

## Frozen source order and record contract

The operator completed the round-98 acquisition, but its checker refused the
first `ffu_nw` group because the group's declared dimensions differed from the
domain dimensions in the file header. The compiled record writer accepts the
actual allocated coefficient arrays and writes each group's own
`SIZE(value,1:2)`. The compiled solver allocates those arrays over
`Nis0:Nie0,Njs0:Nje0`, not over `jpi,jpj`, in
`ORCA2_OMIP_L4_R98EENCOEFF/BLD/ppsrc/nemo/dynspg_ts.f90:86,127` and calls the
writer immediately after `dyn_cor_2D_init` at lines 301-303. The checker must
therefore parse and validate the self-described group shapes rather than force
them equal to the separate local-domain header shape.

NEMO constructs the eight coefficients in the executing `np_EEN` arm at
`ORCA2_OMIP_L4_R98EENCOEFF/BLD/ppsrc/nemo/dynspg_ts.f90:1213-1265` and consumes
them in the four-product pairwise order at lines 1373-1396. No solver source,
deck, package file, configuration choice, threshold, stabilizer, carried state,
sea-ice selector, or `unmeasured_features` entry may change this round.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R99-P1 | The record is sound and the refusal is checker-only. | Both records parse as eight unique named fp64 groups with one common positive 2-D shape declared by the groups; that shape is consistent with the compiled `Nis0:Nie0,Njs0:Nje0` allocation; every payload is finite and nonzero; no trailing bytes exist. | Any malformed header, inconsistent group shape, missing field, invalid payload, or shape inconsistent with compiled allocation is **REFUTED**; repair the writer and request a fresh target instead of admitting this record. |
| R99-P2 | The completed acquisition is write-only and rank-complete. | The two records declare ranks 0/1 and their owned slabs cover the 148x180 global domain exactly once; all 20 terminal ocean restart shards are byte-identical to round 96; recorded stamps and producer-content hashes pass. | Any coverage, stamp, provenance, or restart mismatch is **REFUTED**; stop without using coefficient values. |
| R99-P3 | The checker dimension plant can target the declared group contract without hard-coding a payload shape. | The unplanted record admits; mutating one group's declared dimension makes its byte count or common-shape relation inconsistent and the plant fires. | If the plant remains green, the checker is not a valid instrument; stop before measuring coefficients. |
| R99-P4 | NEMO's frozen coefficients discriminate construction from application. | After owned-domain assembly, all eight NEMO arrays are compared bit-for-bit against the production-JIT source-divisor arrays. If all are exact, the compiled product/sum application owns the existing first non-bit row; otherwise the first unequal coefficient and cell own the next source-ordered walk. | Any inability to map the groups from their declared shape/bounds to owned global cells leaves the discriminator **UNMEASURED** and requires a new acquisition. |

## Landing and refusal bar

Admission is metadata repair, not a physics landing. The first non-bit
statement remains owner of the walk, including signed-zero differences. A
model statement may land only if it is bit-exact given NEMO operands and all
shared-card gates pass. If this existing record is sound, it is admitted
without rerunning NEMO; otherwise the round stops with a new fail-closed
acquisition path.
