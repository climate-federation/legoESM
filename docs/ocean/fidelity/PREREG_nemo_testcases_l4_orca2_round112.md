# Preregistration — ORCA2 round 112 southern EEN Coriolis association

Date: 2026-10-02. Base: `0016541792483c42843c1a69094487bb92848b97`.
Scope is one hierarchy-rung-0 source statement, measured **given NEMO's
recorded entry**. No configuration, forcing, initial state, carried state,
threshold, stabilizer, sea-ice selector, or `unmeasured_features` entry may
change.

## Executed oracle path and correction

Round 111 left the first non-bit item at the southern `ff_f` operand in
NEMO's EEN `zpvo_nw` expression: 180 magnitude differences, all at global
row `j=0`, level `k=0` (`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/
dynspg_ts.f90:1244-1248`). Its receipt pointed to the `dom_hgr` copy-fill at
`domhgr.f90:143`. That particular call is a **dead arm on this deck**:
`namelist_cfg:36` and `ocean.output:116,275,303` prove
`ln_read_cfg=.true.` and that NEMO reads `ORCA_R2_zps_domcfg`.

The executed source is `domhgr.f90:101-109,233-236`: `dom_hgr` calls
`iom_get(..., 'ff_f', ..., cd_type='F', kfill=jpfillcopy)`. The generic
`jpfillcopy` southern-boundary assignment copies the nearest inner-domain
value rather than wrapping the northern row (`lbclnk.f90:1198-1225`). This
round cites that executed read path. The dead-arm wording in round 111 will
be retracted explicitly in the round-112 receipt; its measured 180-cell
boundary remains valid.

The production discrepancy is local and explicit: the literal coefficient
builder's generic `jnp.roll(..., 1, axis=0)` wraps the final northern row into
the southern operand. The candidate will replace only the southern `ff_f`
association in the EEN quotient with the oracle's copy-filled association;
the independently open `e3f_0vor`, `r3f`, and `fe3mask` southern operands
remain untouched.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R112-P1 | The unmodified replay reproduces round 111: `south_ff` is first, with exactly 180 magnitude differences at `j=0,k=0`. | Exact census and first index match the admitted round-110 record. | Any mismatch means the baseline moved; stop and reconcile before testing a candidate. |
| R112-P2 | NEMO's executed `jpfillcopy` association is `ff_south[0,:]=ff[0,:]`, with `ff_south[j,:]=ff[j-1,:]` for `j>0`. | That one-variable candidate makes `south_ff` bit-exact on every executed cell. | Any remaining `south_ff` bit rejects the association and forbids a landing. |
| R112-P3 | After only `south_ff` closes, the first remaining item in registered source order is `south_e3f0`: 7 cells, first `(j,i,k)=(0,28,0)`. | The existing downstream counts remain registered and the probe stops at `south_e3f0`. | A different first item owns the continued walk; preserve this prediction as REFUTED. |
| R112-P4 | The changed production statement executes on ORCA2 rung 0/rung 7 and the literal EEN VORTEX cards; the ENE GYRE path and flux-form tanks do not execute this EEN-only association. | A resolved-card scope gate prints this exact routing and its planted route mutation fires. | Any additional card executes the statement: add that card's binding gate before landing. |
| R112-P5 | No certified row leaves AT-BAR and no first-over-bar row moves earlier. | Complete rung-0 and rung-7 200-row comparators pass; every moved row is registered. Shared-card gates pass under their standing predicates. | Any red landing predicate holds the change and names the exact row. |

## Measurement and landing bar

Extend the existing round-111 fraction walk rather than build a second
reader. It must compare baseline and the single `ff_f` association candidate
against the admitted self-describing round-110 record, on executed levels and
rank-complete owned cells, under production JIT on CPU with fp64/x64/libm.
Both oracle-bit and model-bit plants must fire. A direct unit control must
fail when the old cyclic association is restored.

The statement may land only if it is bit-exact given NEMO's recorded
operands on every executing card, both ORCA2 ladder gates satisfy the frozen
landing predicate, and the required GYRE/DINO/tank/generic-card/citation/push
gates pass. Otherwise the round is HELD at the first red line. No downstream
denominator, mask, product, accumulator, fold, or later-substep item is in
scope.
