# DINO mixed-layer-depth climate audit

**Status: POST-REVIEW PASS.** Two independent adversarial reviews initially
held the lane on time-level claim scope and missing controls. Probe commit
`931e02088ddb0ee24a3f73a1df712af3529a485a` resolves every finding; both
reviewers subsequently returned PASS.

**Pre-registration:**
`scripts/validate/ocean_fidelity/dino_1226/PREREG_mld_climate_audit.md` at
`21c523cde0a`, committed before any MLD field or legoESM-versus-NEMO MLD
statistic was computed.

**Instrument:**
`scripts/validate/ocean_fidelity/dino_1226/mld_climate_audit.py` at clean
producer `931e02088dd`. The run used JAX fp64 on CPU; no model was stepped and
no GPU was used.

**Artifact:** `docs/ocean/fidelity/dino_mld_climate_audit_artifact.json`,
SHA-256 `0f00f2c5bad331a871fbb2237aaddc75e789619ff0230c76a9fdd67b16657624`.
The complete bias/source/base-index maps remain in
`/tmp/dino_mld_audit_codex/mld_maps.npz` (SHA-256
`9fb7344d1e6f92232d211f6a52ff8636022f0d9b0b05acea2b0bae6e6afd9bf0`),
with the day-90 rendering in `mld_bias_day90.png` (SHA-256
`c351ddec6f16ce95faef6b0aa407a5c008a7060a9ac465524a018d64f47f07c2`).

## Verdict

> All four supplied 90-day arms are **DIFF** under the registered MLD bar.
> The deciding failure is the southern-basin RMS difference, **22.4795 m** in
> every arm, above the registered DIFF boundary of **20.2775 m**. The signed
> basin mean bias is only about **-1.8991 m**, so the field-level disagreement
> is substantially larger than its regional mean.

The TKE-floor question also has a registered answer:

> The TKE surface-floor fix moves equatorial day-90 MLD **AWAY_FROM_NEMO**.
> Equatorial RMS rises from **0.000002914 m** in the prefix arm to
> **2.957170038 m** in the fix arm, giving registered
> `delta_RMS = +2.957167123 m` against the +1 m AWAY boundary.

This is a metric-specific result. It does not retract the separate result that
the same fix improves equatorial shear, and it does not add MLD to the
campaign's five-metric acceptance gate.

## Day-90 regional scores

Bias is legoESM minus NEMO. All means and RMS values use the common NEMO
`e1t*e2t` area and the intersected surface wet mask.

| arm | region | mean bias [m] | RMS difference [m] | arm verdict |
|---|---|---:|---:|---|
| basin legacy | channel | -2.742063 | 8.721263 | **DIFF** |
|  | basin | -1.899103 | **22.479491** |  |
|  | equator | -0.000002 | 0.000003 |  |
| stagger corrected | channel | -2.751032 | 8.731916 | **DIFF** |
|  | basin | -1.899102 | **22.479491** |  |
|  | equator | -0.000002 | 0.000003 |  |
| TKE prefix | channel | -2.742063 | 8.721263 | **DIFF** |
|  | basin | -1.899103 | **22.479491** |  |
|  | equator | -0.000002 | 0.000003 |  |
| TKE floor fix | channel | -2.799272 | 9.094337 | **DIFF** |
|  | basin | -1.899179 | **22.479496** |  |
|  | equator | +0.836414 | 2.957170 |  |

**Post-hoc descriptive reading:** the saved bias maps show discrete
level-crossing contours and sparse deep negative excursions rather than a
uniform -22 m offset. That is consistent with the small signed mean and large
RMS. This visual description is not a new classifier; the map arrays are the
result.

The RMS evolution also localizes when the discrepancy appears:

| arm | day | channel [m] | basin [m] | equator [m] |
|---|---:|---:|---:|---:|
| legacy / TKE prefix | 0 | 0.000000002 | 0.000000016 | 0.000000000 |
|  | 30 | 5.999885 | 25.254174 | 0.000002980 |
|  | 60 | 8.607859 | 17.397260 | 0.000002954 |
|  | 90 | 8.721263 | 22.479491 | 0.000002914 |
| stagger corrected | 90 | 8.731916 | 22.479491 | 0.000002915 |
| TKE floor fix | 30 | 5.999885 | 25.254175 | 2.561104 |
|  | 60 | 8.607859 | 17.397261 | 1.532600 |
|  | 90 | 9.094337 | 22.479496 | 2.957170 |

The stagger correction is inert for this MLD statistic to the displayed
precision. The TKE-floor arm departs specifically at the equator by day 30;
its basin RMS remains the same as the prefix arm.

## What NEMO computes

The audit reads the executed DINO `WORK` sources, and the probe refuses to run
unless each quoted source is byte-identical to the relevant source or
configuration override.

* Density-criterion `hmlp`: DINO initializes `nmln=nlb10`, integrates
  `MAX(rn2b,0)*e3w`, advances the level only while the cumulative value remains
  below `grav*rho_c/rho0`, and returns live `gdepw(nmln)`
  (`cfgs/DINO/WORK/zdfmxl.F90:90-104`). `rho_c=0.01 kg m-3` is at line 34.
* Reference level: `zrefdep=10-0.1*min(e3w_1d)`, `nlb10` is the first W level
  deeper than it, and `nla10=nlb10-1`
  (`cfgs/DINO/WORK/domzgr.F90:367-371`). DINO resolves
  `zrefdep=8.9932836129 m`, `nlb10=2` one-based, first resolved W depth
  `10.1387511254 m`, and referenced T depth `5.0335819353 m`.
* DINO resolves `g=9.80665 m s-2` and `rho0=1026 kg m-3`
  (`RUN_VERDICT360_M0/ocean.output:183,233`), giving threshold
  `9.558138401559454e-05 m s-2`.
* Turbocline `hmld`: DINO scans upward for composed `avt` below
  `avt_c=5e-4 m2 s-1` and returns live `gdepw`
  (`cfgs/DINO/WORK/zdfmxl.F90:123-152`; threshold at line 35). TKE and EVD are
  active; DDM, surface-wave mixing, and internal-wave mixing are off
  (`RUN_VERDICT360_M0/ocean.output:766-782`). The supplied snapshots contain
  no `avt`, so `hmld` is **UNMEASURED_INPUT_HAS_NO_AVT**. No density diagnostic
  is substituted for it.

The scored criterion code is not a new transcription. The probe calls
production `_nemo_mld_from_n2_integral`, already certified by the committed
`zdf_mxl_nmln_compare.py` operator lane, with the campaign bridge's S-EOS,
fp64 ladder, live free-surface geometry, and 3-D active mask.

## Time-level scope and in-tool retraction

The registered comparison deliberately applies the exact NEMO criterion
symmetrically to matching NOW T/S/ssh states. Review identified that calling
this the native online `hmlp` overclaimed: executed MLF DINO constructs
`rn2b` from BEFORE T/S, then calls vertical physics with NOW geometry
(`cfgs/DINO/WORK/stpmlf.F90:204-210`). The legoESM NPZs do not save BEFORE
T/S at the scored horizons, so a symmetric native-time-level comparison is
not available.

The probe now prints the retraction and calls the score what it is: an offline
NOW-state application of NEMO's exact criterion, not the native online field.
It also emits this **post-review, unscored** NEMO-only sensitivity:

| day | channel BEFORE-vs-NOW RMS [m] | basin [m] | equator [m] |
|---:|---:|---:|---:|
| 0 | 2.212796 | 10.487622 | 0.000000 |
| 30 | 2.247085 | 10.825181 | 0.000000 |
| 60 | 0.000000 | 17.397263 | 0.000000 |
| 90 | 0.424725 | 0.000000 | 0.000000 |

This sensitivity does not alter any day-90 registered verdict, but it forbids
promoting the result to a claim about native online `hmlp` until legoESM
BEFORE-level snapshots exist.

## Controls and provenance

All controls passed before any verdict printed:

1. The production criterion returns hand-computed base index 1 / MLD 20 m for
   a synthetic column whose cumulative contribution crosses after the second
   level.
2. Moving the first contribution from 0.99 to 1.01 of the threshold changes
   the returned base to index 0 / 10 m; a deliberately false expected level
   raises.
3. Day-0 wet T/S differences are within fp32 storage: max T error
   `9.5366e-7`, max S error `1.9073e-6`, versus fp32 spacing
   `3.8147e-6` at the maximum T/S magnitude.
4. A dry-cell poison changes every registered reduction by exactly zero; a
   wet equatorial poison changes its reduction.
5. Whole-domain wet MLD, bias, base-index, dtype, area, and finiteness gates
   pass for NEMO and every arm.
6. Every arm's live ladder hash equals the freshly built bridge:
   `9536f62732ab8823b2899d26ae0c4582cca09e4b2f49d9e17ff1234f88584ff9`.

The artifact records SHA-256 for 73 inputs: probe and preregistration, all four
NPZs and producer logs, mesh, every restart tile, production criterion and
ladder-hash code, NEMO source/WORK files, namelist, and `ocean.output`. All four
producer trees were stamped clean. The NEMO source checkout records two dirty
tracked files, both unrelated GYRE configuration-list files; none of the
quoted or executed DINO sources differs from its checked source pair.
The exact executable that produced `RUN_VERDICT360_M0` is not identifiable
from the retained run provenance, so binary identity is recorded as unavailable
rather than guessed from a current build artifact.

The live GitHub issue #1455 evidence trail could not be read or updated from
this worktree because network access was unavailable. This document is the
branch-local ledger entry pending coordinator posting.
