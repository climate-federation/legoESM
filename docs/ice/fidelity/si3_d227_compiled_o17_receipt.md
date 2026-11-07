# D227 compiled O17 deformation receipt

## Scope and verdict

- [MEASURED] This change is based on parent `01b095a43e3366dd297e0d1c21d08cca3406de3c` and changes the compiled CPU evaluation of `si3_cgrid_deformation`; the operator-only 100-step and cross-lane runs remain open.
- [MEASURED] On the finished source, the real GIVEN-NEMO-ENTRY step-1 scorer reports zero differing and zero non-finite values in both compiled and eager execution: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d227_scratch/d227_jit_step1_final.json` and `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d227_scratch/d227_eager_step1_final.json`.
- [MEASURED] No runtime-derived zero was added: the parent `jnp.zeros_like` accumulators in `packages/ice/legoesm/ice/si3_o17.py` are unchanged.

## Corrected mechanism

- [MEASURED] The Phase-A assertion that this was not FMA contraction is retracted. `x / 4` and `x * 0.25` are bit-identical; the optimized LLVM IR has plain `fmul` and `fadd`, but x86-64 instruction selection contracts `tension*tension + shear_square` and `divergence*divergence + strain_square*0.25` into `vfmadd231pd`/`vfmadd231sd` in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d227_scratch/hlo_deformation/module_0373.jit__lambda.obj-file.slice_pad_fusion_kernel_module.o`.
- [MEASURED] At cell `[9,24]`, the separately rounded eager value is `0x3d6169d748282408`, while `fma(tension,tension,shear_square)` is `0x3d6169d748282409`; the localized values are in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d227_scratch/localize_deformation.json`.
- [MEASURED] NEMO computes `ecc2 = rn_ecc * rn_ecc` and then `z1_ecc2 = 1 / ecc2` at `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/ICE/icedyn_rhg_evp.F90:232-233`, before its deformation expression at `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/ICE/icedyn_rhg_evp.F90:827`.

## Guard sites and measured necessity

- [MEASURED] `packages/ice/legoesm/ice/dynamics.py:1209-1212` retains the mask `optimization_barrier`, matching the mask multiply at `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/ICE/icedyn_rhg_evp.F90:822-824`; it removes 216 step-1 and 20,762 100-step `divu` signed-zero differences, measured in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d227_scratch/a1_jit_step1_no_mask_barrier.json` and `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_si3/review_d227a_argon.log`.
- [MEASURED] `packages/ice/legoesm/ice/dynamics.py:1215-1217` rounds the divergence and tension products before either can feed an add, matching `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/ICE/icedyn_rhg_evp.F90:827`; at eccentricity 2 the reviewed two-square guard changes 37,769 compiled-versus-eager `delta` differences over 100 real record steps to zero, measured in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_si3/review_d227a_argon.log`.
- [MEASURED] `packages/ice/legoesm/ice/dynamics.py:1218-1221` rounds NEMO's eccentricity square, reciprocal input, and scaled-strain product, matching `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/ICE/icedyn_rhg_evp.F90:232-233,827`; the incomplete two-square guard leaves 563 real step-1 differences at eccentricity 1.5 in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_si3/review_d227a_argon.log`, while the completed guard leaves zero in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d227_scratch/eccentricity_1p5_real_step1_after.log`.
- [MEASURED] The regression prepared for commit at `tests/ice/unit/test_si3_o5_strength_aevp.py:543-565` failed on the parent with 8 of 192 deformation elements different at eccentricity 2 and 18 of 192 different at 1.5, each by one ULP; `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d227_scratch/new_test_before.log` records `2 failed`, and `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d227_scratch/new_test_after_final.log` records `2 passed`.

## Audit and validation

- [MEASURED] The real O17 step-1 full-field scorer is zero-difference after these guards, so its scored O17 kernels expose no additional compiled-versus-eager multiply-feeding-add difference on that record.
- [MEASURED] No speculative guard was added elsewhere. The existing synthetic full-step strict comparison remains xfailed with a signed-zero difference, recorded in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d227_scratch/o17_strict_xfail_final.log`; its marker was therefore retained without weakening the assertion.
- [MEASURED] The focused rung-33, rung-34, rheology, O17 boundary/AD/invariant tests record `43 passed` in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d227_scratch/focused_tests_final.log`.
- [MEASURED] The independent rung-34 card tests record `5 passed` in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d227_scratch/rung34_card_final.log`.

## Open operator coverage

- [MEASURED] `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d227_scratch/operator_run_b.sh` was written but not run. It requires at least 250 GB available memory before each scorer/gate and writes only below `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d227_scratch/operator_b`.
- [PLAUSIBLE] The compiled 100-carried-step O17 free run will determine whether the step-1 closure persists and will expose any remaining signed-zero accumulator difference by field, step, and count.
- [PLAUSIBLE] The compiled O5, lane-3 rung-33/rung-34, and O6 commands will determine whether this shared deformation change preserves their admitted results; no result from those unrun commands is claimed here.
