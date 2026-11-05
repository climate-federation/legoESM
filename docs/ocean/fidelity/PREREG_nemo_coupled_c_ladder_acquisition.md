# Preregistration: coupled ocean+SI3 C-ladder acquisition

This file must be committed before the operator starts either NEMO executable. It fixes the measurements and verdicts for the fresh `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY` REF/REC clones using the exact R339/R5 deck, SI3 on, TOP off, two CPU ranks, `nn_fsbc=2`, and 10,800 s ocean steps.

## Fixed runs and retained outputs

- C2 REF and REC: steps 1--2; retain four step-2 ocean/ice restart shards per arm, logs, resolved namelists, binaries, patched/preprocessed sources, manifests, and every REC record.
- C3 REC: steps 1--100 (50 SI3 calls); retain 1,200 C records (kinds 1--6, 100 steps, two ranks), 200 update-flux records (kinds 15/16), 400 stress records (kinds 19/20), 200 private stress records (kind 34), logs, resolved namelists, and step-100 restarts.
- C4 REF: steps 1--2,920; `nn_stock=8`; retain exactly 1,460 daily ocean+ice restart shards (365 days × two products × two ranks), logs, and resolved namelists. No observer stream is permitted.
- Retain all raw material until the C0/C1 score and later C2--C4 composition admission are complete. A refusal moves evidence under `refused/`; it does not delete it.

## Preregistered measurements and verdicts

1. Source placement/passivity. CONFIRM only if every inserted executable statement is a writer call, placement is exactly before/after the registered live-R5 boundaries, and all four C2 REF/REC restart shards are byte-identical. REFUTE on any observer assignment/reordering, dead-arm placement, missing shard, or one-byte difference.
2. C0 `sbc_ssm`. Score all seven exit fields on both parities, both ranks, steps 1--100 against `nemo_sbc_ssm_step`. CONFIRM only at zero unequal IEEE-754 words (0 ULP). REFUTE at one unequal word. The planted one-ULP change must REFUTE.
3. C1 update flux/stress. Score all kind-16 outputs on the 50 SI3 calls against `si3_ice_update_flx`. On `MOD(kt-1,2)==0`, score kind-20 and private kind-34 outputs through the unchanged production `si3_ice_update_tau` (`nn_fsbc=1` contract). On the intervening ocean steps, the acquisition scorer applies NEMO's literal outer cadence guard (preserve `tmod_io`, `taum`, `utau_oce`, and `vtau_oce`) before the every-step stress blend; this is scorer-only and adds no production-model knob. Score final native-face stresses with `nemo_t_stress_to_uv`. CONFIRM only at 0 ULP for every owned value. REFUTE at one unequal word, wrong parity, or wrong time level.
4. C2, C3, C4 composed-step fidelity. The only permitted current status is `NOT_YET_SCORABLE_NO_COMPOSED_LEGOESM_COUPLED_STEP`; no match/fidelity verdict may be printed from these acquisitions.
5. Schema/integrity. CONFIRM only for magic `CLAD0001`, version 1, fp64, exact kind/field counts, complete payloads, exact expected record counts, and manifest digests. Kind 1 contains 19 fields when `remove_inverse_barometer=0` and 20 fields, including `ssh_inverse_barometer`, when it is 1. REFUTE on any flag/count/payload disagreement, truncation, trailing bytes, wrong magic/version/dtype/count, or digest mismatch.
6. Resources. CONFIRM only if C2 ≤ 6 GiB and ≤ 900 s per integration, C3 ≤ 150 GiB and ≤ 21,600 s, C4 ≤ 30 GiB and ≤ 10,800 s, and the pre-run RAM/free-disk gates pass. REFUTE any excess. Disk and RAM planted shortages must REFUTE.
7. Deck. CONFIRM only for `nn_it000=1`, C2 `(nn_itend,nn_stock)=(2,2)`, C3 `(100,100)`, C4 `(2920,8)`, `rn_Dt=10800`, `nn_fsbc=2`, `nn_ice=2`, two MPI ranks, unchanged ice namelist, and C4 frequency-mode daily ocean+ice restarts. REFUTE any other resolved value or restart count.
8. Control flow/retention. CONFIRM only if the stub exercises the acknowledged `--run` branch through staging and reaches the exact `mpirun -np 2` start point, while the retention plant leaves refused evidence recoverable. REFUTE if only `--preflight-only` is exercised, real NEMO starts during preflight, or refused evidence is deleted.

No statistic, tolerance, output count, budget, parity rule, or verdict above may be changed after the first run; any amendment requires a new committed preregistration before rerunning.

## 2026-11-05 refused-run receipt and schema repair

The approved operator run completed C2 REF+REC with byte passivity, C3 REC with 100 steps and 1,200 C records, and C4 REF with 2,920 steps and 1,460 restart shards. The scorer then refused the first kind-1 record as truncated. The retained refusal is `round30/refused/oracle_coupled_c_ladder_20261105T183812Z_rc1_3724699`; it remains read-only.

The record header advertised 20 fields while its three flags were `0,0,0`. With `ln_apr_dyn` false, `sbc_apr_init` did not allocate `ssh_ib`; the unconditional stream write emitted no bytes, leaving a 1,828,932-byte file instead of the advertised 1,943,236 bytes. The missing 114,304 bytes are exactly one `94x152` fp64 array. Every following field was therefore decoded one position early.

The repair does not change the deck, run lengths, output counts, cadence, scoring bar, or verdicts. The writer now emits `ssh_ib` iff it is allocated, records 19 or 20 fields accordingly, and refuses an allocation/flag disagreement. The reader accepts only the corresponding flag/count/payload layout and retains independent truncated and trailing-byte refusals. The allocation-off preflight stub writes the 19-field layout; a planted copy of the old unconditional 20-field layout must be refused before any real run.
