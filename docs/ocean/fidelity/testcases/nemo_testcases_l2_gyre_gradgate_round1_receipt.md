# Certified GYRE reverse-mode gradient gate — round 1 receipt

Date: 2026-10-11

Status: **STOPPED_FOR_DECISION**. The one-step production-card row reproduced
the preregistered reverse-mode NaN while forward mode and central finite
differences remained finite and agreed to `5.763743549886549e-07` relative.
The six-step reverse and every resource row are **UNMEASURED**; no memory value
or 60-day estimate is inferred from a failed run. No physics, configuration,
stabilizer, clipping, guard, or `stop_gradient` change was made.

## Provenance and reused construction

The frozen preregistration is
`nemo_testcases_l2_gyre_gradgate_round1_prereg.md` at commit `bb949848f` and is
copied outside git as
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round1/preregistration.md`
(SHA-256 `b8e034be346fd36586c01645b4fff5f33e2a6517c8e0342d87e8c06d6b945d15`).
The final numerical row was measured from clean commit
`dd55acbab4df13508b89a3f309cdbbb74e9dcdc0`; its JSON is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round1/cpu_agreement_1step_current.json`
(SHA-256 `2168dbcc7b8fb67f82e552305d6b0e3fccd85e6551b72075f76272923e4e700d`).
The earlier `cpu_agreement_1step.json` belongs to superseded instrument commit
`9aa72b7a2` and is retained but not cited for the final verdict.

The instrument builds `build_nemo_testcase_card("GYRE-zco")`, uses the
campaign forcing helper, and enters every modeled step through the production
`LatLonCGridOceanModel._step_jitted` implementation at
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:13210`.
It reuses the four-field live-volume-normalized loss and per-step Python
composition from `origin/feat/gyre-gradient-twin-2` commit `901cfe0d8`. It also
reuses that lane's Decision 49 construction: each `_step_jitted` call has its
own JIT/rematerialization boundary. A direct production rollout is retained as
truth, and the gate refuses the row unless the rematerialized primal is bitwise
identical. The current one-step run cleared that identity check.

The new `gyre_nemo_card` matrix variant calls the same instrument for one and
six steps and accepts only two `CONFIRMED` rows. The direction is evaluated as
the scalar line `L(base + alpha*direction)`; a direct unit test establishes
that reverse differentiation in `alpha` equals the full input VJP dotted with
the direction. This avoids materializing a 22 x 32 x 30 cotangent without
changing the requested derivative.

## Registered agreement rows

All values below are CPU fp64/libm. Relative bars remain reverse/forward
`<=1e-9` and reverse/FD `<=0.05`; the central-difference line step is `1e-3`.

| steps | model time | reverse | forward | central FD | rev/fwd | rev/FD | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 14,400 s | NaN | `-3.884979632703889e-06` | `-3.884977393501259e-06` | undefined | undefined | **REFUTED** |
| 6 | 86,400 s | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | **UNMEASURED** |

The one-step prediction of a reverse NaN is **CONFIRMED**. The predicted
six-step NaN remains **UNMEASURED**, not confirmed: the rematerialized run did
not write a commit-stamped JSON. The preregistered agreement claim therefore
fails closed.

The plant is non-vacuous. The unplanted nonlinear synthetic control passes the
same classifier; inserting the primal-identity custom VJP doubles only the
reverse derivative, changes the verdict to `REFUTED`, and the direct test exits
nonzero if that flip stops occurring. The real red row is not used as plant
evidence.

## Localization and DECISION_NEEDED

The result reproduces the earlier gradient twin's reverse-only failure: forward
mode and FD are finite and mutually agree, while the pullback is non-finite.
The live NEMO source computes `zsqen = SQRT(en)` and
`dissl = zsqen / zmxld` at
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/MY_SRC/zdftke.F90:832-837`.
The certified lane translates that statement without a reverse guard as
`jnp.sqrt(tke_curr) / l_eps_final` at
`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:3512-3516`.
The earlier branch localized the zero-TKE transpose poison there and landed the
user-authorized primal-bit-identical double-`where` repair in commit
`e7f3797de`; that commit is not an ancestor of this lane.

**DECISION_NEEDED:** Should the previously validated primal-bit-identical
double-`where` guard be landed in the certified lane and this unchanged gate be
rerun? **Pick: yes.** This measurement lane does not land it.

## Reverse resource table

The registered shared schedules are implemented in
`packages/ml/legoesm/training/checkpoint_schedule.py:120-155`. No successful
blocked warm reverse pass was obtained before the CPU campaign cap. GPU 1 is
not exposed in this sandbox: `nvidia-smi` could not communicate with the driver
and JAX reported `CUDA_ERROR_NO_DEVICE`. The exact environment record is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round1/unmeasured_environment.md`.

| backend | schedule | 1-step wall | 1-step peak | 6-step wall | 6-step peak | 360-step estimate | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| CPU | none | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| CPU | uniform | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| CPU | binomial | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| GPU 1 | none | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | GPU unavailable |
| GPU 1 | uniform | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | GPU unavailable |
| GPU 1 | binomial | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | GPU unavailable |

Because no six-step peak `P6` exists, the preregistered 360-step formulas are
not evaluated. In particular, the table does not substitute process RSS from a
compile failure or the earlier twin's unrelated 60-step run.

## Tests and controls

- `tests/ocean/unit/test_nemo_testcase_l2_gyre_gradient_gate.py`: **7 passed**.
- `tests/ocean/unit/test_nemo_recipe.py`: **25 passed in 333.66 s**.
- `nemo_testcase_receipt_citation_gate.py`: **PASS**, 274 citations, no
  failures; all seven planted violations fired and both unplanted baselines
  passed.
- The expensive matrix row itself is red by construction because its one-step
  result is `REFUTED`; it was not redundantly rerun after the stamped CLI row.
- Attempts to launch independent Codex and Claude reviewers were blocked by
  the environment (read-only app-server initialization and untrusted-workspace
  prompt, respectively). This receipt is therefore **UNREVIEWED**.

## OPEN

1. After the guard decision, rerun both CPU agreement rows and require the
   one-step and six-step reverse/forward/FD bars to pass.
2. Measure GPU-1 agreement, peak device memory, and wall time for all schedules
   when GPU 1 is exposed.
3. Measure CPU one-step/six-step host memory and wall time for `none`,
   `uniform`, and `binomial`; only then evaluate the frozen 60-day formulas.
4. Add the SWG certified card's own gradient gate when that card exists.
5. Obtain two independent adversarial reviews before citing numerical results
   or shipping any later physics repair.
