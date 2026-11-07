# Preregistration: GYRE LDF last association, ZAD retest, and barotropic memory, round 50

Date: 2026-09-11. Frozen at legoESM `2fd3560d85f3` before any round-50
measurement or production edit. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round50/`. CPU/fp64/libm only.
No NEMO executable will be built or run. The concurrent TKE file and GYRE
YEAR harness are excluded from this round.

## 1. LDF statement walk

The admitted inputs are the round-46 kt=1/2 stage records and the round-40
stage-3 term records already copied into that acquisition. The executed source
is `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90:121-140`.
GYRE resolves `nn_ahm_ijk_t=0`, `rn_Uv=2.0`, and `rn_Lv=100.e3`
(`EXP00/namelist_cfg:189-191`); the compiled initializer forms
`zah0=(1/2*rn_Uv)*rn_Lv` and masks it into `ahmt/ahmf`
(`BLD/ppsrc/nemo/ldfdyn.f90:312-322,387-394`).

The gate will first classify all round-49 residual cells by wet-boundary ring,
corner, native j row, equatorial relationship, and signed ULP distance. It will
then replay, in scalar `numpy.float64`, every source assignment and written
association: the four curl products and differences; `zwf`; the four
thickness-weighted divergence products and differences; `zwt`; both `zwf`
and `zwt` face differences; stored reciprocals; Kmm output divisions; masks;
and final RHS addition. The replay is calibrated only if post-LDF differs from
the recorded NEMO accumulator in **0 wet cells** for U and V at kt=2 stages 1
and 3. Any nonzero calibration row is GATE-ERROR and no attribution may print.

The same intermediates are exposed through legoESM's ordinary model path.
The first intermediate with nonzero cells is the owning statement. One
nextafter operand plant at that statement must exit nonzero and move a scored
wet cell. CONFIRM requires a single shared source-order transcription to make
all four post-LDF rows exact; REFUTE is any nonzero row. No card-name guard,
new selector, threshold, or fallback is allowed.

## 2. Conditional ZAD retest and trajectory

If and only if LDF lands exact, apply the preserved round-47 ZAD patch to a
temporary scratch copy and rerun the unchanged two-sided kt=1..10 comparison.
CONFIRM requires the former 57 worsened rows to become zero and no scored row
to worsen; otherwise REFUTE and do not land ZAD. For each landed statement,
report GYRE given-input kt=1/2, LOCK_EXCHANGE and OVERFLOW `ln_dynldf_OFF`,
DINO's distinct resolved branch and shared-statement risk, and ORCA2 as
UNMEASURED-WITH-SPEC unless native evidence already exists. The unchanged
trajectory gate supplies kt=1..10 first-over-bar before and after.

## 3. Barotropic-memory record and model score

The round-48 acquisition produced both records but stopped before its dedicated
gate: the inherited admission scanner rejected the registered round-46 magic
`NEMO_L2_R46STG1`. Fix only that reader/registry defect and prove the planted
unknown magic remains red. The dedicated reader must then pass header,
truncation, boundary, seed, reset, and false-stamp plants.

Run legoESM's ordinary GYRE card through kt=1 and decode its deviation-form
`bt_hist` through the production reconstruction path
(`barotropic_latlon_cgrid.py:2053-2084`). Compare raw `ubb_e/ub_e/vbb_e/vb_e/
sshbb_e/sshb_e`, `uu_b/vv_b`, current external mode, Kmm/Kaa U/V/SSH, and
reset accumulators against the NEMO kt=1-end/kt=2-start records. The bar is
zero unequal cells per field. A non-bit row names the first owning compiled
statement, including history rotation (`dynspg_ts.f90:749-761`) or SSH time
filter (`:593-600`), and remains debt; no inferred correction lands.

## Decisions

| status | item | disposition |
|---|---|---|
| ASKED | literal LDF walk/fix, conditional ZAD retest, trajectory and Rule-12 tables, round-48 reader repair and memory score | this round under gates above |
| UNASKED | NEMO run/build/source edit, TKE/YEAR edit, bar/config/card/threshold choice, per-card implementation | forbidden |
