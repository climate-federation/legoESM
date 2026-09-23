# Preregistration: conditional tracer-entry continuation, round 61

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 60 proves the round-59 accumulator arithmetic is locally exact on
oracle substep operands: raw+metric is bit-exact, while raw+cancelled,
normalized+metric, and normalized+cancelled are red on 8, 134, and 140 U
faces. Production row 8.3 nevertheless retains 104 red columns with maximum
normalized error `1.4948039082e-15`; that is an explicit accumulated upstream
substep-operand qualification and is not reclassified as at bar.

Run one CPU replay with the registered NEMO `zu_frc/zv_frc` hold and replace
only the completed `Hu_avg/Hv_avg` returned by the split-explicit solver with
retained `spg_dump_un_adv_final.bin` / `spg_dump_vn_adv_final.bin`. The solver
must still execute once; only its returned transport tuple is substituted.
This is the same conditional frame that made the Kmm execute/undo cycle
bit-exact in round 54, now routed through the complete production tracer-entry
path.

Score rows 8.3--8.10 sequentially with their unchanged bars (`1e-15` for
8.3--8.6; `1e-12` for 8.7--8.10). CONFIRM the conditional release only if
8.3 reaches bar, the prior 104-column production debt remains admitted by
hash, exactly one forcing and one transport substitution occur, and all
identity/wet-point/roll/sign plants fire. Stop on the first subsequent failed
row. If all rows pass, release Redi temperature next. This conditional replay
does not retract or hide the production row-8.3 qualification.

Frozen receipts:

- round-59 held SHA-256
  `85ea27cce4804d98f281940fe472e798d9fa64c741c23bb55e3fca40ee9ca677`;
- round-60 SHA-256
  `b3ef5c0534ff1348dbdb581686aa602cc1d9eca9ef61336ca0b4130217e54e2d`;
- NEMO U/V final transport SHA-256
  `4d8e7a6445ba465c8229954b443c467862381409805163f2045f5854017917ab` /
  `ed1e28aa27e1c49d237e07a07707220251b3a1845a599d21b06df18b8425d088`.
