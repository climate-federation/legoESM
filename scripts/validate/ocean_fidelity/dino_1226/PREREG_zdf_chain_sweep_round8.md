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

## Dated amendment: measured row-12 owner and next-round design

Date: 2026-08-28. Written after the registered row-12 measurement and before
any row-12 production change.

Row 12 is DIVERGED in 9,920/9,920 wet columns, with all four southern focus
columns failing. The first coefficient in source order, `zd_up`, fails with
maximum normalized column error `8.829184`; `zd_lw` and `zdiag` also fail in
all columns. The operand walk localizes the first failure to the denominator
at `zdftke.F90:504`: production supplies no live `e3t(jk,Kmm)` to the solve.
Its legacy branch places the incoming `e3w` in the face-gradient slot used as
that `e3t` operand; the slot fails all 9,920 columns with maximum `0.2484444`.
The preceding `zcof`, `p_avm`, and clipped upper viscosity sum are exact at
zero. The lower `e3t(jk-1,Kmm)` slot fails all columns (max `0.2394656`), and
the shifted/repeated `e3w` control-volume slot fails all columns (max
`0.4662894`). The independently scored incoming `e3w(Kmm)` is exact at zero,
but is assigned to the wrong coefficient role.

**LOUD RETRACTION (2026-08-28):** the first post-measurement draft scored one
shifted `dz_int_eff` approximation against `e3t` and reported max
`0.6806799`. Review found that this was not coefficient-aware and omitted
registered `zcof` and viscosity-sum checkpoints. Before any production fix,
the committed probe was corrected and rerun. The figures above replace that
draft; the owner remains the line-504 `e3t(jk,Kmm)` slot.

The independently checked `dissl` operand also fails all columns (maximum
`0.1088969`), but it occurs later, in `zdiag` at line 510, and cannot displace
the earlier line-504 owner. It remains the next row-12 operand after the
metric fix is rerun.

The next-round production design is a new
`tke_matrix_evaluation="nemo_literal"` path. It carries live raw-mesh
`e3t_0*(1+r3t)` at the same step-entry lifetime as the verified N2 geometry,
then evaluates `zcof`, `zd_up`, `zd_lw`, and `zdiag` in NEMO source order
before calling the unchanged differentiable Thomas solver. The two complete
DINO NEMO cards select `nemo_literal` by default; their explicit legacy opt-in
is `factored`. Generic `TKEConfig`, DINO `nemo_paper`/`veros`, ORCA recipes,
ACC recipes, and MPAS retain `factored` byte-for-byte. This is deliberately a
matrix selector, not a blanket activation of `veros_dz_slots`, because that
flag owns additional Veros N2, surface-volume, and mixing-length semantics.

Required red tests are a hand-computed nonuniform column where `e3t != e3w`,
a planted shifted-`e3w` denominator that fails, exact source-order coefficient
values, JIT/gradient finiteness, and byte-identity pins for every unchanged
card. The post-fix rerun must first close `zd_up` and `zd_lw`; it then resumes
at `dissl/zdiag`. Rows 13--32 remain ordered-unmeasured. Climate arms remain
unauthorized.
