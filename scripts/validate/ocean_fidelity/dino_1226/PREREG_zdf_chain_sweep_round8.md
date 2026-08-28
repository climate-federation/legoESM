# Preregistration: row-12 TKE diffusion matrix

Date: 2026-08-28. Status at commit: **row 11 is VERIFIED; row 12 has not
been measured**.

Row 11 closed with 0/9,920 failing columns at the registered `1e-15` bar and
4/4 southern focus columns passing. The ordered sweep therefore advances to
the first unmeasured operation, the TKE lower, upper, and diagonal matrix
coefficients at `cfgs/DINO/MY_SRC/zdftke.F90:499-510`.

## Registered measurement

The NEMO reference is reconstructed in source order from independently stored
day-180 operands: `tke_dump_avm_in.bin`, `tke_dump_dissl.bin`, `mesh_mask.nc`
`tmask/e3t_0`, the live reciprocal-first qco stretch, and the already verified
live `e3w(Kmm)`. For `jk=2..jpkm1`, the ordered checkpoints are:

1. `zcof = (-0.5*rn_Dt)*tmask(jk)`;
2. `max(p_avm(jk+1)+p_avm(jk),2e-5)` and
   `max(p_avm(jk)+p_avm(jk-1),2e-5)`;
3. `zd_up = zcof*up_sum/(e3t(jk,Kmm)*e3w(jk,Kmm))`;
4. `zd_lw = zcof*low_sum/(e3t(jk-1,Kmm)*e3w(jk,Kmm))`;
5. `zdiag = 1-zd_lw-zd_up + (1.5*rn_Dt*rn_ediss)*dissl*wmask`.

The legoESM candidate is the actual `(a,b,c)` passed by the production TKE
path to its first tridiagonal solve, captured write-free at the solver call.
Its virtual surface row is excluded; the following 34 rows map exactly to
NEMO `jk=2..jpkm1`. Boundary-pinned dry rows are excluded by the registered
wet-W mask.

Each coefficient and each intermediate checkpoint is scored per column over
all 9,920 wet columns and the four registered southern-basin focus columns.
VERIFIED requires zero failing columns at maximum normalized per-column error
`<=1e-15`, 4/4 focus passes, and finite values. A planted coefficient
perturbation and a one-column horizontal roll must both fail. The first
checkpoint over bar is DIVERGED; later substitutions may localize but cannot
move ownership upstream.

If row 12 diverges only by arithmetic association, the next-round fix design
will be a selectable NEMO-literal matrix assembly, faithful by default only on
the two complete DINO NEMO cards and legacy opt-in there. No production fix is
implemented until the first operand is measured. If row 12 verifies, row 13
inherits the canonical preregistration and the sweep continues in order.

Climate arms remain unauthorized until row 32 is VERIFIED.
