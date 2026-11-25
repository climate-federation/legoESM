# ORCA2 round 255 preregistration — substep-2 SSH writer split

Date: 2026-10-12. Frozen base: `23f43cf2d`. Scope: OMT-4, both
**independent** and **given NEMO's entry** labels, with round 254's complete
private fold/transport unit. This is an offline measurement round. No deck,
selector, carried-state contract, stabiliser, sea-ice field, or
`unmeasured_features` entry may change.

## Compiled statement and admitted operands

The compiled OMT-4 oracle forms `zhdiv` from `zhU`, `zhV`, and
`r1_e1e2t`, then writes

`ssha_e = (sshn_e - rDt_e * (ssh_frc + zhdiv)) * ssmask`

at `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:550-558`.
It next associates `ssha_e` with the other external fields at
`dynspg_ts.f90:721-739` and rotates it into `sshn_e` at
`dynspg_ts.f90:781-793`. The admitted round-253 rank-complete record carries
the substep entry, metric transports, forcing, exit SSH, and all geometry
needed to replay this statement. The candidate trace carries the same named
boundaries at `barotropic_latlon_cgrid.py:2050-2096,2336-2374,2403-2408`.

## Frozen predictions and falsifiers

1. **R255-P1 — provenance and passivity.** Both labels run production JIT on
   CPU in fp64/libm, use the production D114 resolved-mesh route, reproduce
   completed kt=1..7, and retain round 254's exact candidate midpoint replay.
   The round-253 admission remains `PASS_R253_OMT4_SPG8_ADMISSION`. A dirty or
   wrong tree, moved completed state, stale route, missing label, or unadmitted
   record invalidates every number.
2. **R255-P2 — boundary ownership.** At kt=8 stage 1 substep 2, the candidate
   continuity output before association, `boundary_post_eta`, and the
   substep-3 current history are array-identical to each other. All three are
   non-bit against NEMO's `j002_ssha_e`. This CONFIRMS continuity as the first
   boundary and exonerates the association and history rotation. Any first
   change introduced only by association or rotation REFUTES this prediction
   and activates that cited statement instead.
3. **R255-P3 — continuity operand split.** In NEMO's written order, score the
   incoming `sshn_e`, `zhU`, `zhV`, `r1_e1e2t`, reconstructed `zhdiv`,
   `ssh_frc`, `rDt_e`, `ssmask`, and literal output. The first non-bit input is
   predicted to be incoming `sshn_e`, because round 254 measured the same
   substep-2 exit as the first non-bit substep-3 current history. The literal
   NEMO replay must reproduce `j002_ssha_e` bit-for-bit; the literal candidate
   replay must reproduce the traced continuity output bit-for-bit. A replay
   miss invalidates attribution.
4. **R255-P4 — disposition.** If incoming `sshn_e` is first, the round is
   HELD and the next walk moves to substep 1's continuity output; no downstream
   statement lands. If every incoming operand is bit-exact but the output is
   not, the arithmetic statement is the candidate and may be transcribed only
   after a one-variable arm closes its own output and the complete unit passes
   Decision 96. If an association or rotation is first, only that cited
   statement is eligible for the same atomic rescore.
5. **R255-P5 — controls.** Label coverage, round-254 dependency, admitted
   record, source order, NEMO replay, candidate replay, boundary identity,
   first-nonbit, and one-bit plants must each refuse when planted.

## Stop conditions

No acquisition is planned and no package change is expected. Decision 116
remains pending and is not re-emitted. No floor, clip, damping, alternate
configuration, or partial held association may land.
