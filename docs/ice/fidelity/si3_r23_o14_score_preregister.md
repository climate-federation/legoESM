# SI3 Ladder Round 23 (`si3_r23`) — O14 Model Preparation and Scoring Preregistration

Date: 2026-10-28. Parent: `87a1c8843803`. Scope is rung O14 only.

## Frozen identity and source boundary

- Oracle: the admitted 100-step `ORCA2_SAS_ICE` master record under
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master`.
- Projection: O14 manifest SHA-256
  `8130e5ac516cf8c1549badbab2caf90cf5cce03f3167c317a10532b836572cd6`,
  exactly 200 records with entry kind 12 and exit kind 13 for `kt=1..100`.
- Resolved selectors are read from the admitted deck and must be exactly
  `ln_icedO=T`, `ln_leadhfx=T`, `rn_hinew=0.1`, `ln_frazil=F`,
  `nn_icesal=4`, `rn_sinew=0.75`, five categories, ten ice layers, and
  `rDt_ice=21600 s`. No default may supply any of them.
- The executing statement boundary is `ice_thd_do`: kind 12 is after
  `ice_itd_rem` and before `ice_thd_do`; kind 13 is after `ice_thd_do` and
  before the post-thermodynamic `ice_cor` call.
- The production implementation must execute the compiled-source order for
  category area, ice volume, bulk salt volume, ice enthalpy, layer salt
  volume, total concentration, and the open-water heat/freshwater/salt fluxes.
  Existing pass-through fields remain bit-identical.

## Frozen predictions and falsifiers

1. **Admission prediction.** The run, inventory, and O14 projection digests
   pass the existing master-record admission verifier. Any digest, schema,
   scalar-header, dtype, record-count, or resolved-namelist mismatch refutes
   admission and stops scoring.
2. **O14 prediction.** Feeding every kind-12 entry through the single
   production `ice_thd_do` implementation yields every kind-13 field at zero
   unequal bits, including signed zero, for all 100 steps and all five
   categories/layers. Any one-bit difference refutes O14. The scorer must name
   the first `(kt, category, cell, field)` and stop without relaxing the bar.
3. **Production wiring prediction.** The production SI3 column step calls the
   same `ice_thd_do` implementation. Removing that call must fail a direct
   production-call-site test. No second implementation of O14 arithmetic may
   exist in the ice package or scoring harness.
4. **Plant prediction.** A one-ULP change to kind-13 `v_i` must make the scorer
   exit nonzero. A model mutation that bypasses open-water growth must also
   fail against a nonzero recorded O14 event.
5. **JAX prediction.** Real eager execution under `jax.disable_jit()` and JIT
   execution with arrays supplied as runtime arguments are bit-identical.
   Gradients are finite and agree with central differences away from switches;
   vanished-ice and zero-category edge states remain finite.
6. **Caller-control prediction.** Parent versus candidate changes zero bits in
   the D5 C1D one-year output hash
   `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`,
   every other production caller, and default non-SI3 runs. Any changed caller
   outside O14 stops the landing.

No configuration, threshold, cadence, state schema, or scoring choice is made
in this round. The requested O14 identity and bar are applied unchanged.
