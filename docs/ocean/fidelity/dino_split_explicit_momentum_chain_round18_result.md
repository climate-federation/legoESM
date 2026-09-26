# Split-explicit momentum chain: row-1.2 owner, round 18

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Registered verdict

Row 1.2 is **`OWNED_BY_BEFORE_THICKNESS_TIME_LEVEL_AND_LEFT_ACCUMULATION`**.
Using the existing restart's BEFORE SSH `sshb`, the exact MLF initialization
expression is bit-identical to NEMO on all 9,758 wet U faces and all 9,868 wet
V faces: normalized RMS error, maximum error/NEMO RMS, and mismatch count are
all zero. The NOW-SSH control remains above the frozen POINTWISE `1e-15` bar
(U maximum `5.5743163337299506e-15`, V maximum
`6.396341264411133e-15`), as do static-weight and reverse-order controls.

The active source is not `dynatf_qco` at this restart boundary. MLF
`restart.F90:331-352` reads the 3-D `ub/vb` BEFORE state but no persistent 2-D
barotropic state; `istate.F90:149-155` reconstructs `uu_b/vv_b(Kbb)` from
live `e3u/e3v(Kbb)`, surface-to-bottom accumulation, and live reciprocal face
depth. `dynspg_ts.F90:571-579` then copies that field into `un_e/vn_e`.
Round 17's use of NOW SSH is reclassified as the registered time-level control,
not an input defect requiring a new writer.

The authoritative artifact is
`dino_split_explicit_momentum_chain_round18_seed_artifact.json`, SHA-256
`001ac4ab57c89b8d762d25e7c171e3ef5a231411e692e5bb38ed648c7c6f808a`.
The pre-adjudication external run copy was SHA-256
`4a95fccfa966a006b4573788f2337a3ba25b5ab052d52ea79063af5245226e29`;
the numerical arrays and scores are identical, while the committed copy emits
the preregistered combined disposition and binds adjudication commit
`ed4c9f7ba1cb735e7aee91bec502b2dd4964ad5c`.
It binds the round-17 control artifact, round-16 recurrence, restart, mesh, seed
dumps, CPU/fp64, populations, script, and planted controls.

## Production design and ordered stop

The production fix is **DESIGNED, NOT BUILT**. Add a dispatch-hardened
`barotropic_seed_evaluation="nemo_literal"` selector. At the MLF before-level
entry it must use the already-passed `eta_init`/BEFORE velocity, construct the
live QCO U/V face weights in NEMO operand order, accumulate the numerator from
surface to bottom, and apply the corresponding reference-depth reciprocal and
QCO stretch separately as `istate.F90:149-155` does. The generic fused stacked
reduction remains the legacy option. Tests must cover unknown-selector failure,
JIT and gradient finiteness, a planted nonuniform BEFORE/NOW SSH separation,
and bit identity of the literal U/V seed against this artifact's oracle.

After implementation, rerun the hardened row-1.2 recurrence at the production
commit. Only an exact U/V result releases row 1.3, which must then resume at
pressure gradient, in-loop Coriolis, explicit bottom stress, and final update
association (`dynspg_ts.F90:766-850`). Row 1.4, registry rows 2--6, and the
remaining free-surface-filter, momentum-RHS, and tracer-tail registry chains
remain **ORDERED-BLOCKED**. Existing dumps suffice for the present owner; no
SLOT block is emitted until the post-fix row-1.3 inventory proves an operand is
missing.
