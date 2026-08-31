# Preregistration: Treguier Kmm geometry/N² operands, round 68

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 67 refutes the vertical-association hypothesis: source-ordered left
accumulation changes `zah` by only `7.94e-16` normalized and leaves `zn`,
`zhw`, `zRo`, `aeiu`, and row 8.8 unchanged. `zhw` is therefore decisive:
its only variable operand is `e3w(Kmm)*wmask` (`ldftra.F90:753-754`), and it
still fails at maximum normalized error `4.6767241595e-8`.

Replay the exact production `_nemo_wpoint_e3w_wmask_n2` call nested inside
the Treguier builder and compare its direct `e3w` and `pn2` outputs to the
existing full-halo `eiv_dump_e3w.bin` and `eiv_dump_rn2b.bin`. Use the
unchanged `1e-12` operand bar, the registered wet W population, and identity,
wet-point, and level-roll controls. If `e3w` fails and `pn2` passes, own the
coefficient residual to Kmm geometry time level; if `pn2` fails first, stop
there. No new NEMO run is permitted.

Frozen round-67 receipt SHA-256:
`2c1c1819f08cc07d9aa90fd43624285fe276558bbb5bba84aa97b3b83b345d95`.
