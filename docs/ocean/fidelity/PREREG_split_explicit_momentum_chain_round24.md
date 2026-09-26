# Preregistration: row-1.3 QCO-thickness composition factorial, round 24

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Frozen before evaluating any QCO
thickness substitution. Bound predecessor: round-23 artifact SHA-256
`4e9c6ee0a172ff5f5005186be1f1f7080927643ad88e1e7d044f00167c168e2e`.

## Existing-dump inputs

No NEMO instrumentation or run is authorized. The retained round-22 run
provides the eight oracle coefficients, restart SSH, `mesh_mask.nc` reference
`e3u_0/e3v_0/e3f_0`, and deterministic `r3u/r3v` dumps.  The scorer first
transcribes `dom_qco_r3c` from restart SSH and mesh metrics. Both reconstructed
`r3u` and `r3v` must match their dumps at normalized RMS and maximum/RMS
`<=1e-15`; otherwise no `r3f`-dependent result is reported.

NEMO's executed EEN coefficient arithmetic is `dynspg_ts.F90:1331-1379`:
three source-ordered `ff_f/e3f_vor` terms, vertical left accumulation of live
`e3u*e3v*mask`, then local depth and horizontal metric factors. legoESM's
generic construction is `barotropic_latlon_cgrid.py:737-762,821-842`.

## Frozen 2x2 arms and bars

The two factors are:

- `F`: cell-average vertex f (0) or NEMO face-latitude f (1);
- `Q`: production generic live face/F thicknesses (0) or NEMO literal QCO
  fields (1): `e3u_0*(1+r3u*umask)`, `e3v_0*(1+r3v*vmask)`,
  `e3f_0vor*(1+r3f*fe3mask)`, and their live `hu/hv` denominators.

All four arms are materialized through the same production AL81 operator and
checkerboard inversion.  Metrics `e1u/e1v/e2u/e2v` are also scored directly
against the mesh and must be at the POINTWISE bar before Q ownership can be
claimed. Each of the eight coefficients and both applied outputs uses normalized
RMS and maximum/RMS `<=1e-15`; bit mismatches are diagnostic. Controls require
the `F0Q0` metrics to reproduce round 22, `F1Q0` to reproduce round 23, an
identity pass, and a decisive repeated-`nextafter` plant.

Main effects and interaction are reported as signed changes in aggregate
eight-coefficient normalized RMS:
`F=(F1Q0-F0Q0)`, `Q=(F0Q1-F0Q0)`, and
`F×Q=F1Q1-F1Q0-F0Q1+F0Q0`. `F1Q1` exact on all ten rows dispositions
`JOINT_VERTEX_F_QCO_THICKNESS_OWNS_COEFFICIENT_COMPOSITION`. If it removes at
least 90% but misses a bar, disposition is `JOINT_COMPOSITION_PARTIAL`; below
90% is `QCO_THICKNESS_REFUTED`. Only the exact verdict authorizes a production
fix, behind a new faithful selector and defaulted only on the two exactness
cards; every other card remains byte-pinned. A non-exact verdict advances to a
literal source-order/metric-placement scorer before any new NEMO dump.

Rows 1.4, 2--6, free-surface filter, momentum RHS, and tracer tail remain
ordered-blocked until row 1.3 reaches an exact or bounded disposition.

### Pre-measurement time-level clarification

The deterministic `r3u/r3v` streams are written after `dom_qco_r3c`, so their
control input is the retained `sshnxt_dump_ssh_after.bin`, exactly as in the
committed `een_e3f_mechanism.py` precedent.  The coefficient arm separately
uses restart Kmm SSH, the state consumed by `dyn_cor_2D_init(Kmm)`.  This
clarifies the two SSH time levels before evaluation; it changes no bar or arm.
