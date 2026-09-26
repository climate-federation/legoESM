# Preregistration: ZDF row-30 `zau` metric/evaluation peel

Date: 2026-08-29. CPU-only matched day-180 state. Frozen before the offline
operand substitutions below. This peel follows the first red stage in
`PREREG_zdf_chain_sweep_round30_uv_operands.md`; it does not change that
measurement or inspect any later row-30 stage.

## Source-order localization

The active NEMO line is `ldfslp.F90:242`:

```fortran
zau = zgru(ji,jj,iik) * r1_e1u(ji,jj)
```

NEMO constructs `r1_e1u = 1._wp / e1u` in `domhgr.F90:140`. The measured
`zgru(iik)` operand is exact in all 9,758 wet U columns, so its contributing
BEFORE-level `rhd` EOS field (`stpmlf.F90:219,234`), `umask`, and horizontal
difference (`ldfslp.F90:203,217`) are exonerated at this stage. The
`eos_rab(ts,Nbb)` alpha/beta call at `stpmlf.F90:205` feeds `rn2b`, and hence
the later `zdzr` operand; it does not enter line 242. This peel therefore
tests only (a) the U-face `e1u` value and (b) division versus NEMO's stored
reciprocal followed by multiplication.

## Frozen substitutions and bars

The committed probe consumes only the certified existing row-30 state and
dumps. It scores, in this order:

1. captured production `zau` (must reproduce 190/9,758 failures);
2. exact NEMO `e1u` versus legoESM's selected U metric;
3. verified `zgru` divided by legoESM `e1u`;
4. verified `zgru` multiplied by a separately evaluated legoESM reciprocal;
5. verified `zgru` divided by dumped-mesh NEMO `e1u`;
6. verified `zgru` multiplied by a separately evaluated dumped-mesh NEMO
   reciprocal, in NumPy, JAX eager, and JAX-jitted forms.

Every `zau` candidate uses the registered normalized whole-column bar
`1.0e-15`, must report failures out of 9,758 wet U columns, and separately
scores the four southern focus columns `(11,1), (12,1), (13,1), (13,23)`.
The metric-value check reports exact-unequal points plus max absolute and ULP
distance; it is not a substitute for the column bar.

`CONFIRM-NEMO-RECIPROCAL` requires the captured production reproduction to be
190/9,758, all four focus columns to pass, and the JAX-jitted NEMO-metric
reciprocal/multiply candidate to be 0/9,758 with all focus columns passing.
The lego-metric reciprocal and NEMO-metric division arms apportion metric
value versus evaluation form; neither may be silently promoted if the full
literal arm remains red. Any nonzero full-literal count is `REFUTE` and row
30 remains at a still-unowned line-242 interval.

Controls are red-capable: the verified `zgru` baseline must be exact; a
bar-scale perturbation, one-i roll, and wet NaN must fail; advancing one
nonzero reciprocal by one ULP must change the resulting product at that
element. All inputs, dumps, source, probe, artifact, clean repository HEAD,
CPU backend, fp64 policy, lane, binary, and restart receipts are SHA-bound.

## Scope and climate adjudication

The measured four southern focus columns passed `zau`. That bounds the row-30
deviation outside the registered MLD focus support, but it does not waive the
whole-domain row bar. The frozen parent preregistration says explicitly:
“Climate remains unauthorized until rows 30--32 are VERIFIED or waived.” It
contains no focus-only or `OPEN-BOUNDED` exception. Accordingly row 30 remains
formally `DIVERGED` until fixed and remeasured (or separately waived under a
preregistered admissible reason); rows 31--32 remain promotion-blocked, and
the climate arms remain unauthorized.
