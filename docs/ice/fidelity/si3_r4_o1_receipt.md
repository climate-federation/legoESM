# SI3 Ladder Round 4 (`si3_r4`) Receipt — Rung O1

## Outcome

Resume 2 resolves the Codex and Claude HOLDs and leaves the round at `ACQUISITION_NEEDED`. NEMO was not built or run. The CPU-only acquisition preflight is `PREFLIGHT_OK` with 33 negative controls; the focused SI3 suite passes 13/13, the private-import/default-path set passes 17/17, and the SI3 inline-coefficient ratchet passes. Codex has no GPU, so the corrected commit still needs an operator GPU spot-check.

The original preregistration is restored byte-for-byte to commit `345df680bf`. The source-discovered reason for moving live Prather NetCDF restart certification to O4 is recorded separately in `docs/ice/fidelity/si3_r4_o1_preregister_amendment_1.md`.

## Corrected model and restart contract

NEMO IOM restart payloads contain no halos. `si3_rst_from_netcdf_dict` now accepts NEMO's halo-free spatial layout or an already allocated mapping, rejects other shapes, embeds the payload into the true interior, and applies ORCA2 boundary conditions (`packages/ice/legoesm/ice/si3_orca2_state.py:1678-1686`). The NetCDF writer and reader expose that same contract (`packages/ice/legoesm/ice/si3_orca2_state.py:2022-2030`, `:2090-2099`). A compiled gfortran/NetCDF fixture writes `(x, y) = (180, 148)` with NEMO IOM dimension order and domain attributes; preflight proves it maps to `(184, 152, 5)` and receives halos (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:697-725`).

Restart branching now matches NEMO: uncoupled `t1_ice` is undefined and excluded from O1 scoring; fresh lane-3 state uses layer-1 ice temperature; `snwice_mass` is read only for `ln_rstart`; and `v_il` requires both pond data and `ln_pnd_lids`. Scoring uses only defined wet interior T/U/V cells and treats full-grid ULP as diagnostic (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1320-1336`). Dry signed zero and halo differences pass, while a wet one-ULP plant refuses.

The production `step_sea_ice` SI3-O1 branch requires explicit configuration and exact `dt == rDt_ice`, then refuses until the NEMO `ice_update_flx` heat/freshwater/salt forcing contract exists (`packages/ice/legoesm/ice/sea_ice.py:590-610`). The direct public `si3_o1_step` remains the O1 state transform; no silent zero flux or ignored timestep remains.

## Corrected acquisition and admission contract

The restart deck uses distinct `input_restart` and `output_restart` directories and a distinct output suffix (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:202-206`). Each staged run records `RUN_PRODUCER_COMMIT`; restart input is copied into its isolated directory (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:1039-1056`). Admission cross-checks the per-run producer stamp, verifies the copied input, and requires the distinct restart output.

`inputs.sha256` no longer hashes the gate. `gate.sha256` records gate provenance separately so repaired gates can re-admit retained producer evidence. `--admit-existing` trusts the producer commit recorded at run time instead of comparing evidence to the current checkout. Binary/input SHA and missing-completion/STOP plants now reach their real comparison. Admission and every plant are driven through one path (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:949-958`; `scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1622-1650`).

No O1-deck `ORCA2_SAS_ICE` output existed on disk. The committed text is explicitly a namelist/source-derived fixture, never rewritten run evidence; its verifier accepts only the derivable O1 values (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1072-1100`). Actual acquisition logs still require normal completion and error checks.

Generic same-program ELF comparison now has one implementation in `scripts/validate/ocean_fidelity/same_program.py`; the accidental byte-identical ORCA2 tool copy was removed. Preflight performs source/build-flag comparison before any NEMO build, and admission retains the post-build object proof.

## Public interfaces for the parallel O2 rebase

* `read_si3_netcdf_restart` / `write_si3_netcdf_restart`: files are halo-free; returned `SI3State` remains allocated with halos.
* `si3_rst_from_netcdf_dict`: accepts halo-free or allocated mappings and refuses other spatial shapes.
* `step_sea_ice` with `thermo_scheme="si3_orca2_o1"`: validates explicit config and timestep, then raises `NotImplementedError` until flux coupling is implemented; call `si3_o1_step` for the O1 state operator.
* Evidence adds per-run `RUN_PRODUCER_COMMIT`, `manifest/gate.sha256`, and separate `rst_1step/input_restart` / `output_restart` directories. The gate CLI adds `--verify-deck-fixture`.

## Verification and disposition

The regenerated `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round4/o1_preflight_receipt.json` has SHA-256 `275025cfeca61118fcbec2e06ffb83fe67ab74fa0c270ae5de05c2962564b6ee`. Its wet maximum is 0 ULP; dry/halo fixture metrics are `ulp_dry_negzero=0`, `ulp_dry_val=0`, `ulp_halo_val=0`, and `ulp_wet_1ulp=1`.

All eleven Resume 2 findings are fixed. Choices: `ASKED` = DECISION 153/159 and the explicit permission to refuse incomplete O1 production fluxes; `UNASKED` = none. Independent review remains operator acquisition and GPU spot-check.
