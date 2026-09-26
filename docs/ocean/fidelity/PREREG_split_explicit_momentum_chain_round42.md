# Preregistration: row-5 call1-to-call2 operand factorial, round 42

Date: 2026-08-30. Frozen before the first round-42 numerical execution.
Session `01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Bound finding and source correction

The official round-41 artifact, SHA-256
`e4b8b814746476d4b19cf31d05a65e2ee145632c07822a56591550cf157c410d`,
reports `ROW5_WZV_CALL2_DIVERGED`: normalized RMS `0.1712886978`, correlation
`0.9893368095`, and RMS ratio `1.0795734102` under the unchanged accumulating
bar `1e-12`. All controls and the duplicate bracket passed.

The suggestion that `dyn_zdf`-updated velocity enters call 2 is rejected by
the executed interfaces, before measurement: `dyn_zdf` writes `uu/vv(Naa)`
at `stpmlf.F90:396-409`, whereas the second `wzv(kstp,Nbb,Nnn,Naa,ww)` at
`:411-412` reads its horizontal divergence at `Kmm=Nnn`
(`sshwzv.F90:184,221`). The Kmm velocity changed earlier, at
`dynspg_ts.F90:1170-1174`, and the second `div_hor` was already evaluated
before `dyn_zdf` (`stpmlf.F90:350-379`). The only changed recurrence operands
between calls are therefore:

1. **H**: first-call `sshnxt_dump_hdiv.bin` versus second-call
   `seq_dump_hdiv_nnn_kt00005761.bin`, the latter using the
   `dyn_spg_ts`-corrected Kmm velocities; and
2. **Q**: first-guess Kaa `r3t` in `r3c_dump_r3t_kt00005761.bin` versus the
   barotropic Kaa `r3t` in `seq_dump_r3t_aaa_kt00005761.bin`.

Kbb/Kmm SSH, `e3t_0`, T mask, and `r1_Dt=1/(2*2700 s)` are held fixed from
the admitted entry restart and mesh. The literal recurrence is exactly
`sshwzv.F90:218-227`, including bottom zero and bottom-to-surface left
accumulation.

## Frozen 2x2 arms and bars

The four offline arms use existing full-halo streams only:

| arm | H divergence | Q Kaa `r3t` |
|---|---|---|
| H0Q0 | call 1 | call 1 |
| H1Q0 | call 2 | call 1 |
| H0Q1 | call 1 | call 2 |
| H1Q1 | call 2 | call 2 |

H0Q0 must reproduce `wzv_dump_ww_call1.bin` and H1Q1 must reproduce
`wzv_dump_ww_call2.bin` at the registered `wzv` accumulating bar `1e-12`.
Failure of either reconstruction is `INVALID`, not evidence. Identity must
be AT BAR; sign, zonal-roll, and `2e-12*RMS` point plants must turn red. The
factorial interaction closure must be no larger than `1e-12` normalized RMS.

Ownership is evaluated against the actual call1-to-call2 squared-error
energy. A single factor owns the delta only if its single substitution removes
at least 95% of that energy and the other removes below 5%. If both main
effects lie between 5% and 95%, or either changes sign under the other factor,
the result is `CALL2_HDIV_X_KAA_COMPOSITION`. If the full arm closes but these
rules do not identify an owner, disposition is `CALL2_OPERANDS_BOUNDED`.

This factorial localizes the *change between NEMO calls*. The much larger
production-vs-call2 residual is fixed only by extending the already faithful
literal QCO W path to this second boundary; no result may instead route
call-1 W into the tracer tail. Row 6 remains ordered-blocked until the
production call-2 replay is AT BAR.
