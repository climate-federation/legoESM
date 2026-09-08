# Preregistration amendment: row-1.3 vertex-f coefficient owner, round 23

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Frozen before evaluating the
face-latitude arm against the retained round-22 dumps.

## Bound stop

The authoritative round-22 scorer artifact is
`/tmp/dino_split_explicit_momentum_chain_round22_corcoef.json`, SHA-256
`c770d68c352682d15464c0bda01cb1c9cde22a857600e237f2658e8c6a97d1e3`.
Its literal reconstruction of NEMO's four-term application is bit-exact on all
9,758 U and 9,868 V points. Production diverges in all eight frozen coefficient
families, each at normalized RMS `3.4059e-5` to `3.6047e-5`.

NEMO forms each EEN corner triad from three `ff_f/e3f_vor` values before its
source-ordered vertical recurrence (`dynspg_ts.F90:1517-1563`) and applies the
local face metrics only after that recurrence (`dynspg_ts.F90:1536-1563`).
legoESM currently supplies the arithmetic mean of adjacent T-row Coriolis
values. NEMO's `ff_f` is instead `2*Omega*sin(gphif)`. The campaign previously
measured the median relative gap between these two constructions as `3.6e-5`;
the already-shipped `coriolis_placement={cell_average,face_latitude}` selector
exists for exactly this distinction.

## Frozen arm and bars

The single arm changes only `coriolis_placement` from `cell_average` to
`face_latitude`, rebuilds the bridged geometry with that selector, and reuses
the retained round-22 coefficient, velocity, mesh, and restart dumps. No NEMO
run is authorized.

The scorer must report:

1. bridged `f_v[1:-1]` versus NEMO `ff_f` on every finite point;
2. all eight checkerboard-materialized effective coefficients versus their
   retained NEMO dumps on 9,758 U or 9,868 V wet points; and
3. the production live Coriolis output versus `zu_trd/zv_trd` on those same
   populations.

Every row uses the POINTWISE bar: normalized RMS and maximum error divided by
NEMO RMS must both be at most `1e-15`. Bit mismatches are reported separately.
A one-ULP wet-point coefficient plant must classify DEBT. The legacy
`cell_average` control must reproduce the bound round-22 coefficient metrics
within `1e-15` relative and must remain DEBT.

Disposition is `VERTEX_F_OWNS_ALL_EIGHT_COEFFICIENTS` only if the face-latitude
arm puts all eight coefficient families and both output components at bar while
the legacy control remains DEBT. A reduction of at least 90% that misses the
bar is `VERTEX_F_PARTIAL_OWNER`; less than 90% reduction is `VERTEX_F_REFUTED`.
Either non-exact verdict authorizes no default change and requires a literal
`e3u/e3v/e3f/r3` operand peel.

If exact, `face_latitude` becomes the faithful default only on
`nemo_dino_kamm` and `nemo_dino_kamm_mlf`; all other cards retain byte-pinned
`cell_average`. Every canonical DINO/NEMO bridge must pass the resolved card
selector when constructing geometry. Existing conservation tests remain red:
the known face-latitude solid-body-curl trade is accepted for oracle fidelity,
while EEN energy/enstrophy identities must remain at roundoff.

Rows 1.4, 2--6, free-surface filter, momentum RHS, and tracer tail remain
ordered-blocked until this coefficient row reaches a registered disposition.

### Pre-measurement control amendment

The phrase "one-ULP wet-point coefficient plant" above is not a decisive
control for a `1e-15` normalized bar: one binary64 ULP of an order-`1e-4`
coefficient, divided by the field RMS, is itself below that bar.  Before any
arm was evaluated, the control was therefore made red-capable by advancing one
wet coefficient by the smallest repeated `nextafter` count whose independently
computed normalized maximum exceeds `1e-15`.  The scorer records that count,
requires the unmodified oracle identity to pass, and requires the planted copy
to fail.  This amendment changes no scientific arm, population, or bar.
