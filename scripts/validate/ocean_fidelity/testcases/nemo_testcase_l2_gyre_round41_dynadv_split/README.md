# Round 41 — GYRE stage-3 `dyn_keg` / `dyn_zad` split

The operator runs exactly:

```bash
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round41_dynadv_split/run.sh
```

The script creates
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round41/oracle_dynadv_split/`
and prints that path only after the additions-only checks, parent-record raw
comparison, consumed-field admission, source replay, closure, stamp, and their
plants pass.

`oracle_dynadv_split_kt00000001_s3.bin` is a named/ranked stream with magic
`NEMO_L2_ADVSP_1`.  Its decoded header pins kt 1, stage 3,
`(Kbb,Kmm,Krhs,Kaa) = (2,2,3,3)`, `nn_dynkeg=0`, the domain bounds, and 64-bit
`wp`.  The first name is local to `dyn_adv`: the compiled caller passes `Kmm`
in both index positions (`stprk3_stg.f90:469`), and the writer records those
two local arguments (`dynadv.f90:184-185`).  It records the
RHS immediately before KEG, after KEG, and after ZAD; the Kmm velocities;
`ww`; a defined all-zero `wsd_effective` plus the false vortex-force flag;
live and reference T/U/V/W thicknesses; T/U/V areas and reciprocal metrics;
and T/U/V/W masks.

On the model side, the one-sided halo convention substitutes the last owned
column in x and supplies zeros in y before calling the shared production paths.

The effective Stokes field is deliberate.  GYRE resolves `ln_wave=F`, so the
compiled initialization returns after forcing `ln_vortex_force=F` and before
allocating `wsd`.  The executed ZAD branch reads `ww` alone; the writer never
touches the unallocated object.
