# Split-explicit momentum chain: round 23 result

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Verdict

`VERTEX_F_REFUTED` under the frozen sole-owner rule.  The retained round-22
artifact was scored by the committed round-23 probe; its artifact is
`/tmp/dino_split_explicit_momentum_chain_round23_vertex_f.json`, SHA-256
`4e9c6ee0a172ff5f5005186be1f1f7080927643ad88e1e7d044f00167c168e2e`.

NEMO's `ff_f` itself is reproduced at the POINTWISE bar (normalized RMS
`1.406e-16`, maximum/RMS `2.744e-16`, 10,348 points).  Substitution removes
66.3889% of the aggregate eight-coefficient normalized RMS, but every corner
remains DEBT at `7.793e-6`--`1.488e-5`; the live output remains DEBT at
`1.535e-5` U and `1.076e-5` V.  Thus face-latitude evaluation is a measured
partial contributor, not the owner registered for this arm.  No recipe default
change is authorized.  The canonical bridge call sites now pass the card's
selector explicitly, which is behavior-neutral while every card remains
byte-pinned to `cell_average`.

The decisive control advanced one wet coefficient by four binary64
`nextafter` steps; identity passed and the plant failed the unchanged `1e-15`
bar.  The legacy arm reproduced every bound round-22 coefficient metric within
the registered relative tolerance and retained all eight DEBT classifications.

## Source-owned next split

The remaining structure lies inside NEMO's live QCO thickness composition:
`dynspg_ts.F90:1331-1342` constructs each three-corner `ff_f/e3f_vor` sum,
`:1344-1347` left-accumulates `e3u(Kmm)*e3v(Kmm)*mask*sum`, and `:1349-1352`
applies the U-face depth and metrics.  The V analog is `:1358-1379`.
legoESM instead constructs live T thickness first, derives face/F thicknesses
through generic C-grid helpers, then reuses the 3-D AL81 operator
(`barotropic_latlon_cgrid.py:737-762,821-842`).  Round 24 factorially
substitutes the retained NEMO `e3u/e3v/e3f/r3` composition while keeping the
operator and all other inputs fixed.

Future held dumps use the canonical `__MEASURED_<name>__` SLOT prefix.  Round
22's shorter names remain bound history and are not silently renamed.
