# NEMO testcase L2 GYRE — round 63 Krhs-split receipt

Date: 2026-09-12. This round writes and compile-checks the acquisition, then
stops. No NEMO build or integration was run.

## Round 63 source verdict and preregistration

The active stage-3 order at kt=2 is:

1. zero T/S Krhs;
2. FCT advection, split into its two-step upstream first guess and its limited
   second-order correction;
3. surface boundary forcing;
4. penetrative solar heating;
5. isoneutral Laplacian lateral diffusion;
6. tra_zdf content assembly.

The zero/advection/surface calls and inactive ice-shelf arm are in
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:825-870`;
the stage-1/2 branch is
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:886-908`;
the complete stage-3 remainder, including QSR, LDF, inactive optional calls,
and ZDF, is
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:912-958`.
The transport prerequisite has two compiled Shuman arms
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:773-800`;
the resolved run selects the first.

FCT calls fct_up1_2stp and then its horizontal order-2 flux at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv_fct.f90:164-198`,
then the vertical order-2 flux, limiter, and corrected accumulation at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv_fct.f90:260-327`.
NEMO's content expression is recorded directly from the existing zl2_rhs and
calibrated against
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:545-564`.

The frozen predicted owner is complete FCT advection. It is confirmed only by
a like-for-like replacement that leaves zero unequal final content values;
any residual, or another first exact boundary, refutes it. No acquisition
exists yet, so the prediction remains UNMEASURED.

## Round 63 record schema and bounds

The tracer stream contains p2dt; zero, upstream-first-guess, after-advection,
after-SBC, after-QSR and after-LDF cumulative T/S Krhs; T/S(Kbb);
e3t(Kbb/Kmm); r3t(Kbb/Kmm); e3t_3d; tmask; and actual content T/S.

All 3-D payloads are the explicit T2D(0) x 1:jpkm1 inner domain,
32 x 22 x 30; 2-D r3t is 32 x 22. The macro declarations are
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/inc/do_loop_substitute.h90:72-89`.
The full FCT tracer dummy is declared at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv_fct.f90:94-111`;
the full tra_zdf dummy at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:86-88`.
The effective-thickness operands are declared at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dom_oce.f90:170-180`.

The TKE stream contains rn_Dt, zfact3, en at RHS entry, shear, avt, rn2,
dissl, avt*rn2, zfact3*dissl*en, wmask, and en immediately after the RHS.
Its reduced/full declarations are
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdftke.f90:219-223`;
the executing expression is
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdftke.f90:409-427`.
The allocation distinction is independently visible at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdf_oce.f90:85-87` and
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/oce.f90:40-42`.

Both records carry schema counts; adjacent stamps bind SHA-256, producer
commit, and filename. The extended round-54 parser refuses either stream
unless reconstructed e3t, T/S content, both TKE products, and the complete TKE
RHS statement have zero unequal fp64 values. One-ULP and truncation plants
exit nonzero.

## Round 63 compile proof

Dry-applied sources and preprocessed products are retained at
/tmp/gyre-r63-syntax.UEmrZP. The exact preprocessing commands were:

```text
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional -I /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/WORK -I /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/inc scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round63_krhs_split/l2_r63_krhs.F90 -o /tmp/gyre-r63-syntax.UEmrZP/l2_r63_krhs.f90
cpp [same -D/-I flags] /tmp/gyre-r63-syntax.UEmrZP/stprk3_stg.F90 -o /tmp/gyre-r63-syntax.UEmrZP/stprk3_stg.f90
cpp [same -D/-I flags] /tmp/gyre-r63-syntax.UEmrZP/trazdf.F90 -o /tmp/gyre-r63-syntax.UEmrZP/trazdf.f90
cpp [same -D/-I flags] /tmp/gyre-r63-syntax.UEmrZP/zdftke.F90 -o /tmp/gyre-r63-syntax.UEmrZP/zdftke.f90
cpp [same -D/-I flags] /tmp/gyre-r63-syntax.UEmrZP/traadv_fct.F90 -o /tmp/gyre-r63-syntax.UEmrZP/traadv_fct.f90
```

Then:

```text
/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran -fsyntax-only -ffree-line-length-none -I /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/inc -J /tmp/gyre-r63-syntax.UEmrZP /tmp/gyre-r63-syntax.UEmrZP/l2_r63_krhs.f90
/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran -fsyntax-only -ffree-line-length-none -I /tmp/gyre-r63-syntax.UEmrZP -I /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/inc -J /tmp/gyre-r63-syntax.UEmrZP /tmp/gyre-r63-syntax.UEmrZP/stprk3_stg.f90
[the identical command for trazdf.f90, zdftke.f90, and traadv_fct.f90]
```

Every cpp and gfortran command exited 0 with empty output.

## Round 63 validation, disposition, and handoff

At clean tool stamp a7688af736af, all 27 focused tracer/parser/citation tests
pass; the new schema/gate subset passes 11/11. The receipt citation gate passes
15 citations and its shifted-line plant exits 1. The clean-stamp compile was
repeated at /tmp/gyre-r63-clean-syntax.ZiJHwS with empty cpp/gfortran output.
The primary checkout remains uncommitted because its .git directory is
read-only; validation used the same-tip writable clone named below.

Exact operator command after commit:

```bash
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round63_krhs_split/run.sh
```

| kind | item | disposition |
|---|---|---|
| ASKED | New two-step R63 writer, runner, calibration gate, plants, twin admission | Implemented; acquisition not run |
| UNASKED | Production physics, carried state, existing round-62 candidates | Unchanged |
| UNASKED | NEMO build/integration | Not run; operator handoff |

Open: which Krhs term owns the content residual; which TKE product owns the
238-cell residual; issue 1455 could not be read because GitHub was unreachable.
