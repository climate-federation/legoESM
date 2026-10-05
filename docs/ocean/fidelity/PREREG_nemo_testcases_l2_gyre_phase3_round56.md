# Round 56 preregistration — derived TKE floor and corrected operand record

Date: 2026-09-11. Frozen parent: `b8b62fa636d1`. CPU, production JIT,
fp64/scalar-libm only. NEMO source and records are read-only. No NEMO
integration is run by the agent.

## Acquisition acceptance fixed before the record

The round-55 record never existed. The replacement target is
`GYRE_OMIP_L2_P3_SM_R56TKE`, writing under
`phase3/round56/oracle_tke_operands`. The compiled card declares the row work
arrays `zdiag/zd_up/zd_lw` on `T1Di(0),jpk` and `p_pdlr` on `T2D(0),jpk`
(`R46KT2/MY_SRC/zdftke.F90:233,218`; preprocessed `:238,223`), while
`T1Di(0)=ntsi:ntei` and `T2D(0)=ntsi:ntei,ntsj:ntej`
(`BLD/inc/do_loop_substitute.h90:72-73,87-89`). The card resolves full
`jpi,jpj=36,26`, `nn_hls=2` (`round46/oracle_kt2_stage/ocean.output:149-155`),
so the record is explicitly the 32x22 inner domain, not the full halo domain.
The seventh plant changes a stamped extent and must exit nonzero.

## Frozen floor hypothesis and falsifiers

NEMO's standard TKE initialization evaluates, in this association,
`rmxl_min = 1.e-6_wp / (rn_ediff * SQRT(rn_emin))`
(`R46KT2/BLD/ppsrc/nemo/zdftke.f90:815-817`), then overwrites `rn_mxl0` with
that value when `ln_mxl0` (`:829-833`). legoESM's corresponding card fields
are `TKEConfig.c_k` (`rn_ediff`) and `TKEConfig.tke_background` (`rn_emin`).
For 0.1 and 1e-6 the fp64 expression is
`9.99999999999999847e-03`, one ULP below the literal `0.01` printed by the
landed GYRE and DINO cards. The shared NEMO TKE path will derive the value;
the raw `mxl0_min_m` field is removed because legoESM exposes no
`ln_mxl0=.false.` NEMO path that could consume it.

Prediction P1: direct construction and every GYRE/DINO NEMO mixing-length
floor are bit-identical to the fp64 expression above, including the surface
anchor and interior floors. Any unequal bit refutes P1. NEMO initializes both
length arrays with `rmxl_min`, floors the buoyancy length with it, and relies on
that initialization through the sweeps (`ppsrc zdftke.f90:593-595,619-621,
626-675`); `avm/avt/dissl` then consume those lengths at `:682-693`.

Prediction P2: with the floor as the sole coefficient owner, the after arm's
kt=3 maximum T/S residuals are exactly the round-55 frozen predictions
`1.6275031290e-4 K` and `6.3278533133e-6 g/kg`, from before values
`8.7413164119e-3 K` and `1.3568885211e-3 g/kg`. Any larger after kt=3 maximum
REFUTES the quantitative prediction; any remaining unequal acquired `avt`
cell REFUTES the floor as sole coefficient owner.

Prediction P3: day-30 T RMS after is smaller than the same-commit before arm.
Equality or worsening REFUTES only this trajectory-direction prediction.
Days 1 through 30 are all scored; no endpoint-only substitution is allowed.

## Frozen measurement and Rule 12

The before arm is the clean commit containing this preregistration and the
instrument repair, before the floor transcription. The after arm differs only
by the shared derived-floor transcription, dead-field deletion, surface-mask
factor, matching zero-step anchor, citations, and focused tests. Both run the
existing kt=1..10 ladder gate and one 30-day member with six-step snapshots;
the day-gap scorer uses the fixed `year_owners` NEMO root and days 1..30.

| card | preregistered disposition |
|---|---|
| GYRE | Measured before/after: kt=1..10, kt3 T/S reported, days 1..30 scored. |
| DINO | Shared NEMO TKE path: resolved floor predicted to move down one fp64 ULP; register as a moved operand. No DINO trajectory is inferred. |
| LOCK_EXCHANGE | Unaffected: resolved test card selects constant mixing (`LOCK_EXCHANGE_OMIP_L1_P3_R33ZDF/EXP00/namelist_cfg:131`). |
| OVERFLOW | Unaffected: resolved test card selects constant mixing (`OVERFLOW_OMIP_L1_P3_R33ZDF/EXP00/namelist_cfg:129`). |
| ORCA2 | UNMEASURED-with-spec: bit-score its compiled TKE closure using that card's own `rn_ediff`, `rn_emin`, masks, and operands before discharge. |

ASKED: corrected acquisition; derived shared floor; GYRE program; Rule-12
audit; citation/test repairs. UNASKED: none. The named evidence directories
are bookkeeping only, not scientific choices.
