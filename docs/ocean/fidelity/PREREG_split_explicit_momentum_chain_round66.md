# Preregistration: row-8.8 GM coefficient ladder, round 66

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 65 localizes row 8.8 to `aeiu`: the consumed `wslpi(k+1)` face sum is
at the `1e-12` bar, `aeiu` fails in 9,185 U columns, and oracle-`aeiu` alone
makes the literal streamfunction pass with maximum normalized error
`1.3219229158e-15`. The normalized-half-sum and NEMO `-r1_4` associations are
identical on the still-red own coefficient, so association is exonerated.

Reuse the existing direct 2-D `eiv_dump_{zn,zah,zhw,zRo,zaeiw}.bin` and 3-D
`eiv_dump_aeiu.bin` streams. Capture the production call to
`compute_treguier_kappa_gm_nemo_native`, replay that exact call with its public
diagnostics return, and score this NEMO order (`ldftra.F90:687-706,715-718`):

1. `zn = SUM(sqrt(max(rn2b,0))*e3w)`;
2. `zah = SUM(rn2b*(wslpi^2+wslpj^2)*e3w*wmask)`;
3. `zhw = 5 + SUM(e3w*wmask)`;
4. `zRo = clip(0.4*zn/abs(f),2km,40km)`;
5. T-point `zaeiw` after timescale, tropical taper, and cap;
6. U-face `aeiu = 0.5*(zaeiw_i+zaeiw_ip1)*ssumask`.

Every operand uses the unchanged `1e-12` bar. Stop at the first failing row.
Also form the U-face average from oracle `zaeiw`; it must reproduce direct
oracle `aeiu`, and a zonal roll plus a wet-point perturbation must fail. If
rows 1--5 pass and row 6 alone fails, own the face average. Otherwise name the
first failing coefficient operand; no downstream promotion is allowed.

Frozen round-65 receipt SHA-256:
`6b4a80ddaf020916770dd0eb0005a6ce9dd69ffe79a6604a0e2125cfbadf4c60`.
