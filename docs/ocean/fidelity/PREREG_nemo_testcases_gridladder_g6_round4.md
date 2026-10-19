# PREREG — grid ladder G6, round 4 (gridladder_r39)

Frozen before the KEG/ZAD split is compared against NEMO.

## Instrument

NEMO records only the combined stage-2 KEG+ZAD accumulator at kt 1
(`after_advection`, `stprk3_stg.f90:412-420` in the record build).  The split
is made OFFLINE: a literal numpy transcription of `dyn_keg` nkeg_C2
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynkeg.f90:61`, statements :118-130; deck
`nn_dynkeg=0`) applied to NEMO's OWN recorded stage-2 `uu/vv(Kmm)` with their
two-cell halos (`oracle_rkstage2_operands`, `stprk3_stg.f90:526-535`) on top
of NEMO's recorded `after_vorticity`, and `r1_e1u = 1/e1u`, `r1_e2v = 1/e2v`
from NEMO's mesh file.  Calibration: the replayed NEMO after-KEG plus the
card's own ZAD increment must reproduce NEMO's `after_advection` at every
cell where the card's after-ZAD already equals NEMO; it is not quoted
otherwise.  Card side: write-only hooks `after_vor`, `after_keg`, `keg`,
`zad`, `after_zad` at stage 2 (no operator reimplemented in the card).

## Predictions

P1  The replay calibrates (above) and NEMO's after-KEG is then compared to
    the card's after-KEG cell by cell.
P2  PLAUSIBLE owner: KEG on the V pivot row and the row below.  NEMO forms
    `zhke` on the halo T row directly from the halo (fold image) velocities
    (`dynkeg.f90:120-125`) and the pivot V reads `zhke(ji,jj+1)` (:130);
    the card differences a T-cell KE through its fold pad / fold-partner
    gradient overwrite.  Predicted: card after-KEG V unequal concentrated on
    the two top V rows; after-KEG U on the pivot T row.
P3  ZAD (`dynzad.f90:59`, :105-127) is equal on the fold rows once KEG is
    removed, OR it owns the scattered interior last-place set.
P4  Whatever the owner, the landing (if a transcription defect) is selected
    only by the G6 card through `rk3_vector_raw_vmask`; other cards unchanged.

## Falsifiers

Replay fails calibration (instrument, not physics — no claim); after-KEG
equal everywhere (then ZAD owns the fold rows and its operands are walked);
any non-G6 card affected by the landing.
