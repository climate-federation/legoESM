# NEMO testcase Lane 4 — ORCA2 Phase-2s preregistration

Date: 2026-09-06

Starting parent: `65dcf0adc25e396758cc018128c1334e8d9c8547`

Status: **PREREGISTERED BEFORE MEASUREMENT.**  This round restores the ORCA2
card's resolved NEMO shear selector, prepares a self-describing non-vacuous
ZDF acquisition, and stops for user-shell MPI twins.  Shared SH2/TKE
arithmetic is outside Lane 4 and will not be changed.

## P2S-1 — card-selector boundary

The resolved ORCA2 deck selects `ln_zdftke=.true.` and no CST/OSM/RIC closure
(`namelist_cfg:378-404`).  NEMO therefore sets `nzdf_phy=np_TKE` and
`l_zdfsh2=.TRUE.` (`zdfphy.F90:207-223`), calls `zdf_sh2` before `zdf_tke`
(`zdfphy.F90:264-286`), and evaluates the no-Stokes face-native statements
(`zdfsh2.F90:78-100`).  Under this RK3 executable, `stprk3.F90:164-165`
passes `Kbb=Nbb, Kmm=Nbb`; both velocity and QCO face-metric factors use the
Nbb whole-step-entry slot.

The single ORCA2 card changes only this explicit selector tuple:

| field | current | corrected NEMO identity |
|---|---|---|
| `tke_shear_production` | `squared_centered` | `nemo_face_native_nbb2` (renamed in Phase 2u; same preregistered arm) |
| `tke_shear_avm_weighting` | `tpoint` | `nemo_face` |
| `tke_shear_metric_source` | `tpoint_jacobian` | `nemo_qco_live_face` |
| `tke_shear_evaluation_stage` | `step_entry` | `step_entry` (unchanged) |

Admission requires the instantiated ORCA2 tuple to equal that table, while
GYRE, LOCK_EXCHANGE, OVERFLOW, and C1D resolved cards remain byte-identical
before/after construction.  Their existing kt=1 gate rows must remain 0 / n.
The gate includes a selector plant that substitutes the old ORCA2 tuple and
must exit nonzero.  This is a user-requested NEMO-identity restoration, not a
new physical choice and not a default change.

## P2S-2 — version-2 ZDF acquisition

The accepted kt=1 frame is retained but is mechanically non-discriminating:
all recorded Kbb/Kmm velocities are zero and `sh2` is therefore zero.  The
WRITE-only `zdfphy.F90` override will write the same 25-array operand frame at
both `kt=nit000` and `kt=nit000+1`.  The second frame is admitted as
non-vacuous only if both Kbb and Kmm velocity-gradient operands have at least
one nonzero owned wet face.  Filenames carry the actual eight-digit `kt`.

The old `NEMO_L4_ZSH2_1` record is superseded by
`NEMO_L4_ZSH2_2`.  Each stream contains:

1. 16-byte magic;
2. the existing 13 little-endian int32 values: version, kt, Kbb, Kmm, Krhs,
   jpi, jpj, jpk, real storage bits, n3, n2, payload-value count, and the
   resolved `l_zdfsh2` flag;
3. 25 allocation triples `(n1,n2,n3)` as little-endian int32, in exact payload
   order (2-D fields use `n3=1`); and
4. the 25 canonical zero-first binary64 arrays in that same order.

The payload count is the sum of the products of the written extent triples.
The validator must decode the real file using only the header, walk all 25
arrays to exact EOF, verify each extent against its payload view, and reject
wrong magic/version/count/extent, truncation, trailing bytes, non-canonical
slots, and a zero-gradient kt>=2 frame.  Every plant is routed through that
same validator and must exit nonzero.

The Phase-2m EEN WRITE-only block is restored to `dynspg_ts.F90` unchanged:
`e3f_0vor`, live `e3f_vor`, `q`, and eight `zpvo` arrays at kt=1.  Its own
zero-first storage and rank-zero guard remain authoritative.  One rebuilt
binary must emit all four EEN streams plus both ZDF SH2 frames.  Twin admission
requires every stream raw-identical.  Relative to the Phase-2q root, all old
streams except the deliberately versioned kt=1 SH2 stream must be raw-identical;
the new kt=2 SH2 and four restored EEN streams are additions.  Phase-2m becomes
an EEN witness rather than a separate pin.

The build uses the unchanged `conda-scalarmath` architecture and accepted CPP
key set; `nm -D` must find zero `_ZGV*` symbols.  No NEMO MPI command runs in
the sandbox.  Two `(jpni,jpnj)=(2,1)` CPU run roots will contain the same
binary, copied namelists, the existing absolute read-only input symlinks,
hash guards, and the unchanged Bash-time launcher pattern.

## P2S-3 — TKE entry pre-walk

Before the user-shell run, the kt=1 records may score only non-vacuous TKE
inputs/statements that do not depend on velocity shear.  Each row is labelled
CONFIRMED, PLAUSIBLE, or UNMEASURED and reports unequal / n.  The kt=2 frame is
registered to discriminate the face-gradient, face-`avm_k`, live-QCO divisor,
coast weighting, `p_sh2`, and its first consumption in `tke_tke`.  The walk
stops at the first non-bit statement; a shared statement routes to GYRE, while
an ORCA2 forcing operand (`taum`, `fr_i`, `rCdU_bot`) remains Lane 4.

## Controls and stop conditions

- Production JIT, CPU, fp64, and the card's explicit scalar-libm policy are
  mandatory for legoESM scores.
- One-variable selector and record-format arms are used.
- Any moved non-ORCA2 card, nonzero cross-card kt=1 row, nonbinding plant,
  `_ZGV*` symbol, non-WRITE-only source diff, malformed real record, or missing
  run prerequisite stops the phase.
- A successful preparation still stops for the user to run the two launchers.

## ASKED / UNASKED at preregistration

| action | classification | disposition |
|---|---|---|
| correct the ORCA2 SH2 tuple | ASKED | preregistered current -> corrected values above |
| choose `kt=nit000+1` as the requested kt>=2 frame | ASKED | minimum requested non-vacuous acquisition |
| choose explicit extent triples rather than an allocation hash | ASKED implementation option | self-describing triples selected; no physics effect |
| restore four EEN streams to the same binary | ASKED | preregistered unchanged from Phase 2m |
| keep BBL defaults 0 / 0.0 | ASKED, Decision 10 | resolved; unchanged, no silent on |
| add prognostic `uu_b/vv_b` here | Decision 8 assigns GYRE | not done in Lane 4 |
| extend scalar libm here | Decision 9 assigns ice-thermo | not done in Lane 4 |
| change shared SH2/TKE arithmetic | forbidden in Lane 4 | none planned |
| sandbox MPI/NEMO, shipped edit, deletion, push | forbidden | none planned |
