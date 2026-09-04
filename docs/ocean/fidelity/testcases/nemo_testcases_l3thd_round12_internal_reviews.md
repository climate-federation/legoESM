# SI3 lane 3b round 12 — Codex-internal review artifact

Date: 2026-09-05  
Final reviewed commit: `746ded3a783e16a8f6506ba580be97a95c6ad936`

These are two independent, read-only Codex-internal reviews. They are not
external-review artifacts.

## `/root/review_r36_source`

Initial verdict on `aa7d2f6a144efd1aaea5099e81dba5356be6eaba`:
**HOLD**. The reviewer confirmed the physical stop, then required the gate to
bind `ssh_post` directly to `snwice_mass_b*r1_rho0`, narrow its initial-state
coverage, cite the executing QCO `e3t` expansion, correct the ratchet count,
and assert that each named plant row becomes DEBT.

Final verdict on `746ded3a783e16a8f6506ba580be97a95c6ad936`:
**SHIP**. The reviewer independently reproduced the unplanted construction
stop, canonical exchange-layout value
`snwice_mass_b=1710.0000000000002 kg m-2`, printed
`r1_rho0=0.0009746588693957114 m3 kg-1`, bit-identical source-order SSH owner,
bit-identical `domzgr_substitute.h90:126` thickness expansion, and final
`e3t=-0.6666666666666667 m` DEBT. The two focused files gave 11/11 passing
tests and the external JSON hash reproduced as
`c518a5b8bdbec8945f45874f7ba2ed8eb9d102490a8149f1c6d5ffdb888eaa5f`.

Limitation: the reviewer did not create a replacement geometry or run a
full-year oracle; those actions correctly remain behind the user decision.

## `/root/review_r36_gate`

Initial verdict on `aa7d2f6a144efd1aaea5099e81dba5356be6eaba`:
**HOLD**. The reviewer reproduced the real gate and controls, then corrected
the hardcoded-constant ratchet count and required the positive-thickness row
to be identified as a post-preregistration construction invariant.

Final verdict on `746ded3a783e16a8f6506ba580be97a95c6ad936`:
**SHIP**. The reviewer reproduced the external JSON byte-for-byte, baseline
exit 1 at `INITIAL_STATE.positive_wet_layer_thickness`, all three plants exit
1 with their specifically targeted row DEBT, the 3/3 construction tests, the
8/8 canonical exchange-layout tests, all quoted hashes, and zero dynamic
`_ZGV*` symbols.

Limitation: the reviewer did not repeat the NEMO compile or invalid 36-step
integration; the immutable executable and retained runtime evidence were
independently inspected and hash-checked.
