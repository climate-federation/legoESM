# SI3 Ladder Round 4 (`si3_r4`) Receipt — Rung O1

## Outcome

Resume 4 closes the five remaining acquisition findings and leaves O1 at `ACQUISITION_NEEDED`. NEMO was not built or run. The CPU acquisition preflight is `PREFLIGHT_OK`; operator acquisition and the operator-run independent review remain outstanding. The original preregistration remains byte-identical to commit `345df680bf`, and no scientific option, bar, window, or model default changed.

## Restart record and forcing clocks

The writer derives every step filename from the actual `kt`, so the `nn_it000=nn_itend=2` continuation writes `o1_si3_state_step0002_rankNNNN.bin` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_si3_state_writer.F90:94-97`). Admission now requires exactly that filename and metadata `kt=2`; its fixture contains no restart `step0001` file, and the `rst-step-name` plant proves the old lookup is refused (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1759-1824`; `tests/ocean/fidelity/test_nemo_si3_o1_orca2_sas_gate.py:1231-1252`).

NEMO updates the mean surface fields before dispatching SI3 in `sbc`: `sbc_ssm` is called at the forcing boundary and `ice_stp` later in the same routine (`src/OCE/SBC/sbcmod.F90:397-403`; `src/OCE/SBC/sbcmod.F90:474-477`). Admission therefore reads `sss_m` from the kt=2 snapshot for both the continuous and restart kt=2 advances, never from kt=1 (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1729-1742`; `scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1811-1833`). The direct fixture gives kt=1 and kt=2 different salinities at an ice-free wet category; `kt2-sss-step1` fails (`tests/ocean/fidelity/test_nemo_si3_o1_orca2_sas_gate.py:1112-1122`; `tests/ocean/fidelity/test_nemo_si3_o1_orca2_sas_gate.py:1301-1325`).

## Defined halo coverage

NEMO computes the full aggregate arrays with `DO_2D(nn_hls,...)` and the category-equivalent thickness/age fields over the same halo extent (`src/ICE/icevar.F90:105-252`; `src/ICE/icevar.F90:255-428`). The scorer now includes the wet cyclic-seam and north-fold halos for those full equivalent and aggregate fields, while retaining interior-only masks only for reduced `A2D(0)` diagnostics (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1261-1357`). One-ULP plants on `h_i` and `at_i` fire at both halo classes; the corrected test comment identifies only `diag_set0` and reduced `A2D(0)` outputs as undefined there (`tests/ocean/fidelity/test_nemo_si3_o1_orca2_sas_gate.py:822-879`).

## Pre-run executable identity and restart dependency

`--run` now builds both real reference and instrumented executables, runs the compiled same-program comparison, and only then permits the first NEMO launch (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:1055-1061`; `scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:1204-1212`). The preflight explicitly checks this operator-call order and plants a missing comparison; its earlier synthetic trees are retained only as a checker fixture (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:199-230`; `scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:905-913`). A real mismatch returns `REFUSE` before any `mpirun`.

The shipped SAS path opens the step restart from `day`, and `sbc_ssm` writes `sss_m` and the other SAS surface fields into the ocean restart (`src/OCE/DOM/daymod.F90:276-277`; `src/OCE/IOM/restart.F90:81-145`; `src/OCE/SBC/sbcssm.F90:177-191`). O1 sets `nn_stock=1`, so `rec_2step` produces `ORCA2_SAS_00000001_restart.nc` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/namelist_cfg:21-24`). Before staging `rst_1step`, the acquisition now requires both that SAS ocean restart and its ice peer with a dedicated refusal; the preflight's missing-ocean-restart plant fires (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:180-194`; `scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:915-926`; `scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:1173-1181`). Admission independently plants the same missing dependency.

## Verification and interfaces

The regenerated preflight receipt has SHA-256 `77dddab1ff9c6966142b8b8216425b4bd68037296ca7c1e410e8acee98c802e9`. It records `PREFLIGHT_OK` and 36 preflight plants. The focused O1 suite passes 15/15, including 13 admission plants. The default citation gate and private-import ratchet pass. Public model interfaces are unchanged; only acquisition/admission behavior and scoring coverage tightened.

Choices: `ASKED` = the five Resume 4 findings. `UNASKED` = none. Search-before-build: searched the existing O1 writer, gate, tests, and imported ORCA2 same-program helpers; extended those existing paths and added no duplicate implementation. Codex review: operator.
