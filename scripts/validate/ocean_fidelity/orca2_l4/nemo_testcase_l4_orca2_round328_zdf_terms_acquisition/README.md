# Round-328 ZDF term-boundary acquisition

Status: `ACQUISITION_NEEDED`. This directory now contains the operator-only,
additions-only acquisition. Its `run.sh` selects the `zdfphy` target in the
previously reviewed round-328 `dyn_zdf` acquisition base; the shared script
retains the record build's `MY_SRC` (including `stprk3_stg`), input deck,
two-rank kt-1 trajectory, refusal retention, EXIT trap, `work_cfgs.txt` guard,
150-GB memory guard, timeout, and default read-only preflight. It does not
change a namelist, model option, timestep, decomposition, or input.

The additions-only observers instrument `zdf_phy` at these source-ordered
boundaries on both ranks in the record build's compiled
`BLD/ppsrc/nemo/zdfphy.f90`:

1. after `zdf_tke` at line 336 and the `avt_k`/`avm_k` copies at lines 348–351;
2. after the `zdf_ddm` block at lines 362–368; and
3. after `zdf_iwm` at lines 370–372, before `zdf_mxl_turb` at line 374 and
   `lbc_lnk` at line 380.

At every boundary write the native fp64 `avm` and `avt` arrays for
`jk=2:jpkm1`. Boundary 1's `avs` has not yet been assigned during this step, so
it is named `b1_avs_previous` and excluded from scoring. At boundary 1 also
write `en`, `avm_k`, `avt_k`, `rn2`, `r3t(Kmm)`,
`ht_0`, `mbkt`, and `wmask`; at boundary 2 write the DDM increments
`zavft+zavdt`, `zavfs+zavds`, and `MAX(heat,salt)`; at boundary 3 write
`zav_wave`, `zemx_iwm`, and `zReb`. The writer is write-only and may not feed
any payload back into NEMO state.

Admission must require both ranks, a producer commit, exact field order and
extents, finite fp64 payloads, and byte identity of all inherited round-328
oracle frames and terminal restarts. One-ULP plants in each of the three
boundaries, plus header, field-order, truncation, and producer-stamp plants,
must refuse through the final-bit or source-order chain comparisons after the
writer checksum has passed. The clean rank-0 record must reproduce the already
admitted final `avm`/`avt`/`avs` bits. Rank 1 must reproduce its own admitted
round-328 `dyn_zdf` `avm_input`; rank-1 `avt` and `avs` have no admitted
reference frame and are explicitly excluded from final-field scoring.

The gate imports the checksum verifier and configuration-root-normalised
same-program proof from the `dyn_zdf` gate. The ZDF writer imports the existing
Fortran checksum functions, and its gfortran fixture reuses the existing
fixture stubs. The fixture writes deliberately different payloads for ranks 0
and 1.
The gate admits both ranks only after the inherited `dyn_zdf` record and all
deterministic ocean/restart outputs remain identical, the post-IWM wet-cell
`avm`/`avt`/`avs` bits reproduce the admitted final record, and the header,
field-order, truncation, producer, and three boundary one-ULP plants refuse.

Operators run the read-only preflight by default:

```sh
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_zdf_terms_acquisition/run.sh
```

After it reports `ORCA2_ROUND328_ZDFPHY_PREFLIGHT_READY`, the operator-only
NEMO acquisition mode is:

```sh
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_zdf_terms_acquisition/run.sh --run
```

No NEMO executable was run while building or preflighting this acquisition.
