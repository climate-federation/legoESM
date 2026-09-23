# Preregistration: row-1.4 final split-explicit outputs, round 27

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Frozen before the current-production
CPU replay. User admission releases row 1.3 as fully AT BAR. Bound held
instrument: round-26 artifact SHA-256
`16a03c0aca5cc1f4ea36c22b144e7ea2ddc339c69777abffddd059e65ffcc8cd`.

## Existing evidence and question

No NEMO run is needed. `RUN_SEQDUMP_D180_1R` and the byte-identical retained
round-26 run contain the complete 68-substep `substep_dump.bin` trajectory and
the five row-1.4 outputs written at active `dynspg_ts.F90:1033-1074` and the
post-loop momentum rewrite at `:1170-1174`:

- `spg_dump_puu_b_final.bin`, `spg_dump_pvv_b_final.bin`;
- `spg_dump_pssh_final.bin`; and
- `spg_dump_un_adv_final.bin`, `spg_dump_vn_adv_final.bin`.

The committed full-recurrence wrapper runs the current production solver on
CPU/fp64 at the matched day-180 restart, holds the already-owned NEMO slow
forcing exactly, and compares those outputs on the registered 9,758 U, 9,868
V, and 9,920 T populations. It uses the unchanged campaign gates for
`dyn_spg_ts puu_b`, `dyn_spg_ts pssh`, and `dyn_spg_ts un_adv`; no new
tolerance is introduced. The primary arm is `forcing_only`; seed-hold results
are controls because row 1.2 is already fixed in production.

Row 1.4 is AT BAR only if all five primary outputs are AT BAR. If so, rows
2--6 are released in order. If any output is DEBT, row 1.4 is the ordered stop
and rows 2--6 are not scored.

## First-DEBT localization

If row 1.4 is DEBT while the admitted substep-1 row remains AT BAR, the next
measurement uses the existing full `substep_dump.bin`; it does not instrument.
The first `jn>1` whose SSH/U/V end state crosses its unchanged campaign gate is
the next ordered subrow. At that substep, the operand ladder is NEMO source
order: continuity/transport, back-interpolated PGF, live EEN Coriolis, explicit
bottom stress, then vector update. A missing production capture at that first
substep stops with a SLOT handoff rather than attributing from a final norm.

The replay must bind checkout-local imports, SHA-pinned mesh/restart/dumps,
CPU/fp64, and a tracked-clean tree. Untracked campaign outputs are ignored by
cleanliness checks. Identity, actual-binding perturbation, zero-shift, and
planted gate controls must pass.

## Stopped implementation replay and authorized correction

The first current-production replay (artifact SHA-256
`2b6685f986492f3b22b4b2c813dd2da2a2c5cc2b990ce0afe6e7f34daed3125f`)
stopped at the already-admitted row 1.3: the branch still selected
`cell_average` Coriolis on the two fidelity cards. That is the temporary
round-24 retraction, made while the corner-coefficient owner was only partial;
round 25 subsequently closed all three registered association axes exactly.
It is therefore an implementation lag, not a new science arm.

Before repeating row 1.4, restore the already-tested `face_latitude` selector
on exactly `nemo_dino_kamm` and its `nemo_dino_kamm_mlf` child. Keep
`DINOConfig` and every other recipe byte-pinned to `cell_average`, with a
red-capable routing test that enumerates the selected recipes. The corrected
replay uses the same inputs, populations, controls, and unchanged bars above.
The stopped artifact is diagnostic only and cannot release row 1.4.

## Post-replay adjudication (not part of the frozen bars)

The selector-only replay did not meet row 1.3. Its artifact SHA-256 is
`74f4bb05827190a9d5bffad0a885374f175d270413a967937e98e95e80bd2b07`;
it retained U/V substep-1 DEBT at normalized RMS
`1.9256402731006323e-7` and `1.0000704249865956e-7`. The registered exact
round-25 arm also changed source association (triad, vertical recurrence, and
post factors) and consumed NEMO's live QCO F-point operands. Restoring only
the vertex-latitude selector was therefore an invalid partial implementation
of that design. The card change is retracted before packaging, while both
diagnostic artifacts remain bound. Row 1.4 was printed only as downstream
diagnostic output and is not admitted.
