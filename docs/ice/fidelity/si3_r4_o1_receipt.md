# SI3 Ladder Round 4 (`si3_r4`) Receipt — Rung O1

## Outcome

Resume 3 fixes all ten Claude/Argon HOLD findings and leaves O1 at `ACQUISITION_NEEDED`. NEMO was not built or run. The CPU acquisition preflight is `PREFLIGHT_OK`; operator acquisition and the operator-run independent review remain outstanding.

The original preregistration remains byte-identical to commit `345df680bf`; its amendment still moves live Prather NetCDF restart certification to O4. No scientific option, bar, window, or model default changed except the requested restoration of the lane-3 bridge salinity default from 34.7 to 34.0 (`packages/ice/legoesm/ice/si3_orca2_state.py:51-62`).

## Restart fidelity and scoring

NEMO reads every `iom_get(..., jpdom_auto)` payload into the no-halo interior and immediately applies `lbc_lnk(..., ldfull=.TRUE.)` (`src/OCE/IOM/iom.F90:1347-1431`). The reader now validates each known field's complete rank/category shape, and the compiled fixture writes `u_ice` and `v_ice` as two-dimensional fields, matching NEMO's declarations and reads (`src/ICE/icerst.F90:275-282`; `scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_nemo_restart_fixture.F90:7-34`).

`v_il` is read whenever its own variable exists, independently of `a_ip` and `ln_pnd_lids`, exactly as NEMO's separate probe requires (`src/ICE/icerst.F90:318-324`). `snwice_mass` and `snwice_mass_b` are read later by `update_rst`; their own `iom_get` calls therefore receive a post-read halo fill rather than retaining the earlier zeros (`src/ICE/iceupdate.F90:455-461`; `packages/ice/legoesm/ice/si3_orca2_state.py:2037-2047`). Direct tests cover all 960 mass-halo cells, malformed velocity/category rank, and the independent `v_il` branches.

Scoring includes every wet cyclic-seam and north-fold halo cell NEMO defines. Only dry cells and halos NEMO leaves undefined are excluded: `t_si`, `snwice_fmass`, equivalent variables, and aggregate diagnostics are written by interior-only loops; uncoupled `t1_ice` remains wholly unscored (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:447-481`). One-ULP plants on both the cyclic seam and north fold now refuse.

## Acquisition and admission

The restart deck is now an actual continuation: `ln_rstart=.true.`, `nn_it000=nn_itend=2`, and both the ocean and ice step-1 restarts from `rec_2step` are copied into the isolated input directory (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:195-208`; `:373-401`). The deck-deviation gate and a `restart-disabled` plant cover those settings. Admission reads legoESM with `ln_rstart=True`, scores the NEMO restart-entry snapshot, advances it at kt=2, and compares it with both the restart run's kt=2 record and the continuous run's kt=2 record.

`--admit-existing` now requires the recorded producer to be a real commit object and an ancestor of the current checkout, verifies per-run stamps, and binds `binaries.sha256` to exactly the reference and instrumented `nemo.exe` paths recorded by the post-build same-program proof (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1566-1616`; `:1619-1650`). The `ref-rec-diff` and `rst-missing` plants now perturb the operands/paths and reach the real byte and existence comparisons rather than raising from the plant selector.

The duplicate same-program implementation is deleted. O1 imports the established round-326 ELF comparator and round-328 multi-observer helper. Preflight performs and plants the planned source/build-flag comparison before any NEMO build; admission retains the post-build preprocessed-source/object comparison. The receipt no longer calls the preflight check a compiled same-program proof.

The citation map is a single static literal again: all 87 stale O1 entries and the `.pop()`/`.update()` mutation are removed. `test_no_citation_map_key_is_written_twice` and the default citation gate pass.

## Verification and interfaces

The regenerated `o1_preflight_receipt.json` has SHA-256 `b84544c5f5fa0c1ee78d6089bd12ac9616cb34018c288092d069d3971bb2bbb6`. The focused O1 suite passes 15/15. The preflight's Fortran/NetCDF fixture confirms `(180,148,5)` categorical arrays, `(180,148)` velocities, allocated `(184,152,5)` / `(184,152)` results, and nonzero defined halos. All 34 configured acquisition plants fire.

Public interfaces remain source-compatible. Tightened behavior: `si3_rst_from_netcdf_dict` now refuses known restart fields with the wrong full rank/category shape; the gate CLI adds `--same-source-flags` and `--expect-rstart`; restart evidence additionally requires the ocean input/output restart pair and same-program binary targets. `scripts/validate/ocean_fidelity/same_program.py` is removed in favor of the restored ORCA2 modules.

Choices: `ASKED` = all ten Resume 3 repairs in the user instruction. `UNASKED` = none. Search-before-build: searched the repository and later ORCA2 lane for same-program implementations; restored and imported the existing round-326/328 functions rather than retaining the O1 copy.
