# Are the model's tunable parameters reachable by a gradient?

Status of the audit as of 2026-08-16. Every number here is measured by
`scripts/validate/audit_param_gradients.py`; nothing is inferred from the specs.

## Why this is not a bookkeeping question

A parameter the gradient cannot see cannot be trained, however well it is
declared. Two existing gates pass for such a parameter — `test_param_specs.py`
checks it is DECLARED and `test_params_reachability_audit.py` checks a config
file can SET it — and neither touches whether any derivative reaches it.

The motivating measurement: in the 2026-08-16 SCM-RCE campaign `dca` improved
its temperature-and-humidity score by **65 %** (6.06 → 2.13) by tuning its CAPE
threshold alone. That parameter had an **exactly zero** gradient, was declared
`tunable_tier 0` for that reason, and was found only because the search was
derivative-free.

## Coverage

| component | spec modules | tier 1-3 params | audited |
|---|---:|---:|---|
| atmosphere | 14 | 551 | convection, microphysics, gravity-wave drag |
| land | 17 | 142 | **none** |
| ocean | 14 | 124 | **none** |
| ice | 1 | 23 | **none** |
| coupler | 3 | 16 | **none** |
| **total** | **49** | **856** | **305 (36 %)** |

## Results so far

| category | live | dead | blocked | nondiff |
|---|---:|---:|---:|---:|
| convection | **74** | 43 | 5 | 0 |
| microphysics | 85 | 66 | 4 | 0 |
| gravity-wave drag | 14 | 18 | 4 | 0 |
| radiation | — | — | — | — |
| turbulence | running | | | |

Convection was re-audited AFTER the CAPE promotion: 66 live -> 74, the eight
added rows being exactly the promoted thresholds.

**No `nondiff` parameters anywhere so far** — nothing produces a non-finite
gradient, which is the outcome that would break a trainer outright.

### radiation exposes NO trainable parameters at all

`GrayRadiationConfig` declares 9, every one `tunable_tier 0`; `RRTMGPConfig` —
the scheme production runs — declares none. So the radiation scheme contributes
zero trainable parameters to the model. That is a declaration gap, not an audit
failure: the audit's own guard refused to report a vacuous pass.

### `blocked` is one cause, and mostly it is not a defect

All 13 blocked parameters are `min`/`max` branch saturation: the parameter is
read, a finite difference moves the output, and the gradient is exactly 0
because the OTHER branch is selected. They split by what the clamp represents:

* **physical clamp — zero gradient is CORRECT.** `MorrisonConfig.melt_rate`
  under the mass-donor limit (`morrison.py:706`): if all the ice melts anyway,
  the rate genuinely does not change the answer. Same for `M_b_max` mass-flux
  caps (tiedtke, zhang_mcfarlane) and for `P3Config.N_i0` / `cooper_a` under the
  oracle's `N_i_nuc_max` ice-number cap.
* **numerical guard — zero gradient would be a DEFECT.** None confirmed.
  `_COOPER_EXP_CAP` looked like one and is not: it binds only below
  ``T_freeze - T > 263 K``, i.e. T < 10 K, so it never binds in an atmosphere.
  The severing clamp there is the physical one — proven by `N_i0` also being
  blocked while it multiplies OUTSIDE the exponential.

A straight-through estimator belongs on a gate whose saturation is a smoothing
artefact, NOT on a physical cap.

### `dead` (84) is weaker evidence than it looks

`dead` means no path to the output **on the audit's five idealized columns at
cold start**. It is the condition a tuner would also fail to exploit, but it is
not proof the field is unused in a full run — an ice process is inert in a warm
column whatever the code does.

## Fixed here

**The CAPE trigger of all eight schemes that gate on CAPE.** `sigmoid` saturates
to exactly 1.0 above an argument of ~36.7, so its derivative is exactly 0, and a
deep-tropical column sits far inside that dead zone. `cape_trigger` now uses a
straight-through estimator — `soft + stop_gradient(hard - soft)` is the hard gate
bit for bit, while the derivative comes from a sigmoid widened 1e3 — so the
forward physics is unchanged and the gradient is finite, non-zero and correctly
signed across the whole physical CAPE range. The thresholds moved from tier 0 to
tier 2. Gated by `tests/unit/test_cape_trigger_gradient.py`, which includes a
control proving the UNFIXED trigger really was dead so the suite cannot pass
vacuously.

### CONFIRMED after the fix

The convection category was re-audited at the post-promotion SHA. All eight
CAPE thresholds now APPEAR (they produced zero rows before) and every one reads
`live`:

| scheme | verdict | grad |
|---|---|---|
| sbm | live | 4.85e-06 |
| zhang_mcfarlane | live | 2.82e-06 |
| dca | live | 1.72e-06 |
| tiedtke | live | 1.55e-06 |
| emanuel | live | 5.20e-07 |
| mass_flux | live | 6.17e-08 |
| edmf | live | 2.20e-08 |
| bechtold | live | 1.49e-08 |

### A structural finding: the audit could not see what it was judging

The audit enumerates parameters through the collector's TIER selection, so a
parameter excluded for being non-differentiable is never re-tested for
differentiability. The CAPE thresholds produced **zero rows**. The exclusion was
self-confirming. Any future tier-0-for-AD-reasons parameter has the same
problem, and the only escape is a targeted test like the one added here.

## What remains

1. **Turbulence and radiation** — turbulence is re-running after a walltime
   kill; radiation needs parameters to exist before it can be audited.
2. **The four unaudited components (305 params).** The audit's core — the fixed
   random projection, the reverse pass, the finite-difference fallback, the
   four-way verdict — is component-agnostic. What is atmosphere-specific is
   state construction (`build_states`) and scheme activation
   (`_single_scheme_config`, `_CATEGORY_WRAPPER`). Extending it means one
   idealized state + activation map per component, not a new audit.
3. **`packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_bvp.py` has an
   INVALID `__param_spec__`** (missing `params` and `scheme_key`), so its
   parameters cannot be collected at all — neither tunable nor trainable.
   Pre-existing and unrelated to this branch, but the same family of defect.
