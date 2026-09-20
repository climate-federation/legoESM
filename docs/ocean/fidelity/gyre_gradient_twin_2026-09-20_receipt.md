# GYRE identical-twin gradient receipt — 2026-09-20

Status: **STOPPED — P1 REFUTED and P5 REFUTED.** No optimizer update, retune,
landscape scan, recovery claim, or NEMO run was made.

The frozen preregistration is
`docs/ocean/fidelity/PREREG_gyre_gradient_twin_2026-09-20.md` at commit
`b82ad5889c3644ac109b46ed6739b0864e9013f3`. The incoming lane tip was
`4cac617cd928007506f2de7ccb098f87e04204d1`; the clean measurement tree for the
final P5 run was `4367678b90d5c5a9d24945fdf9f986cf493214e8` on
`feat/gyre-gradient-twin`. Execution used CPU only, JAX x64, fp64/libm, and
`/home/dbalwada/legoESM/.venv/bin/python` with the requested `PYTHONPATH`.

## Delivered construction

`scripts/experiment/gyre_gradient_twin.py` builds the certified `GYRE-zco`
card and its model directly from `nemo_testcase_recipe.py`. The campaign forcing
assembly was moved, without changing its formula, into the shared
`gyre_surface_forcings` helper; the year/phase-3 harness now delegates to that
same helper. No model or config default changed.

The runner provides the requested `--param {c_k,A_h,both}`, `--days`,
`--obs-days`, `--steps`, and `--root` interface. It uses one checkpointed
`jax.lax.scan`, nested `apply_param_overrides` calls inside the differentiated
loss, live T/U/V control volumes, and the new `legoesm.ml.loss` volume adapter
around the existing `area_weighted_mse` reducer. The only extra model plumbing
is:

- static operator-presence dispatch so traced positive `A_h` can reach the
  existing lateral-viscosity arithmetic;
- a fixed-treedef AB3/AM4 cold carry for the scan, with the exact cold history
  and ramp coefficients selected on the first step;
- one shared public GYRE forcing constructor used by both experiment and
  campaign harness.

## Mandatory direct gate

The direct battery was run as:

```text
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src \
/home/dbalwada/legoESM/.venv/bin/python -m pytest -q -s \
tests/ocean/unit/test_gyre_gradient_twin.py
```

The shared volume reducer and independent planted no-inert/FD controls passed.
The real two-step card established that both nested overrides are reachable and
that the start loss is finite and positive. It then failed the production
no-inert gate exactly as required for a broken gradient path:

```text
no-inert-parameters gate: non-finite gradient: ['A_h', 'c_k']
```

The registered comparisons were still evaluated before the assertion:

| parameter | log step | reverse AD | central FD | AD/FD | verdict |
|---|---:|---:|---:|---:|---|
| `A_h` | `1e-3` | NaN | 0.002872696489271756 | undefined | REFUTED |
| `A_h` | `1e-4` | NaN | 0.0028726939601056434 | undefined | REFUTED |
| `c_k` | `1e-3` | NaN | 0.0445953232661965 | undefined | REFUTED |
| `c_k` | `1e-4` | NaN | 0.04459534694446765 | undefined | REFUTED |

The finite differences are finite at both registered resolutions; the failure
is the reverse numerator, not a zero-denominator or forward-mode limitation.
The reverse call took 3689.195171 s including its first compile, and peak
process RSS was 72,342.980 MiB. The full battery ended `1 failed, 2 passed` in
3958.87 s. The failure is intentionally not marked xfail: repairing the path
must make the same direct test turn green.

## Reverse-mode localization

Post-hoc localization identifies
`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:3434`:

```python
dissl_new = jnp.sqrt(tke_curr) / l_eps_final
```

The primal is finite at masked zero TKE, but the square-root transpose evaluates
a zero output cotangent there as `0 / sqrt(0)`, yielding NaN. The committed
direct regression probe reproduces that VJP and shows that the standard
double-`where` guard is primal-identical, returns zero at the masked entry, and
leaves the positive-entry VJP unchanged. The production operation was not
changed because the preregistered instruction was to localize and stop before
optimization. This is reverse mode; the known forward-mode block at the Thomas
solver's `custom_vjp` is not the owner.

## P5 campaign identity

P5 was run from a clean tracked tree. Both paths used legoESM only: the truth
path was the required 60-step checkpointed scan, and the comparison path was
the existing year-from-rest campaign loop. P5 **REFUTED**:

- unequal day-10 T values: 17,895 / 21,120;
- scan hash: `3eb61de0c5b53abedc2c4a87b83dd327c7c1dcb9d9ead6efd06c70941eb8bda1`;
- campaign hash: `37e978c18339ab871b34e73a89a15cc91dee0623264122edbb8f2f2f6bc44165`;
- truth-scan wall time: 150.120924 s;
- campaign-harness wall time: 193.223138 s;
- peak RSS: 6425.191 MiB.

Post-hoc discrimination found bit identity for a standalone first step and for
an ordinary checkpointed warm step with dynamic forcing/config inputs. The
last-bit discrepancy appears only when otherwise identical warm steps are
composed inside `lax.scan`; pre-materializing forcing did not remove it, and a
separate cold/warm `lax.cond` region did not restore bit identity. These
post-hoc checks localize the open P5 debt to the eager-step versus scan-compiled
program boundary; they do not relax or replace the bitwise P5 verdict.

The stamped campaign snapshot remains outside git at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradient_twin/campaign_reference/lego_seed0/day010.npz`.

## P1–P5 verdicts and recovery curves

| prediction | verdict | result |
|---|---|---|
| P1 reverse-mode FD | **REFUTED** | mandatory two-step gate returned NaN AD for both leaves; all four ratios undefined; the 60-step arms were not entered |
| P2 `A_h` recovery | **UNMEASURED** | zero optimizer updates after P1 stop |
| P3 `c_k` recovery | **UNMEASURED** | zero optimizer updates after P1 stop |
| P4 JIT/eager loss | **UNMEASURED** | the mandatory direct gate stopped the 60-step loss experiment |
| P5 campaign identity | **REFUTED** | 17,895 / 21,120 day-10 T values differ bitwise |

There are no recovery curves to report: `A_h`, `c_k`, and joint arms each have
an empty curve and zero updates. `legoesm.ml.training.create_optimizer` is wired
with the preregistered Adam/warmup/cosine/clip settings but was never invoked.
No identifiability landscape or optimizer retune was run because those actions
are allowed only after a passing adjoint check.

The machine-readable counterpart is
`docs/ocean/fidelity/gyre_gradient_twin_2026-09-20_summary.json`; no state
snapshot or other multi-megabyte artifact is committed.

## OPEN

1. Repair the localized `dissl_new` reverse singularity with a primal-identical
   guard, add a production regression, and rerun the direct AD/FD gate before
   any 60-step gradient or optimizer call.
2. Decide how to satisfy the unchanged bitwise P5 across two different XLA
   composition boundaries. Splitting or encapsulating step compilation would
   change the fixed rollout design and therefore needs an explicit decision;
   relaxing P5 is not proposed.
3. After both gates pass, measure P4, then run `A_h`, `c_k`, and joint arms in
   that order. Until then P2/P3/P4 remain UNMEASURED.
4. GitHub issue #1455 could not be read or updated from this sandbox because
   GitHub network access was unavailable. `codex exec` review was also not
   available in-sandbox; this unit is **UNREVIEWED**, and the operator's stated
   Claude review remains required before citation or physics merge.
