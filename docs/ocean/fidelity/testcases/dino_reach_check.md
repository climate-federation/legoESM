# DINO reach check for the NEMO test-case L1 lane

Branch `fidelity/nemo-testcases-l1-codex`. Asks one question: did this lane,
which exists to certify NEMO's LOCK_EXCHANGE and OVERFLOW test cases, change
anything the certified DINO twin depends on? The lane never ran DINO's gate, so
until now the answer was unknown rather than no.

## Summary

| | lane base `f81436227b1f` | lane HEAD `ef949da85` |
|---|---|---|
| DINO cards that construct | 4 of 4 | 1 of 4 at `d1e4270ca`, 4 of 4 after the fix below |
| NEMO GYRE card constructs | yes | **no** — open finding |
| 5-day twin, day-0 state vs NEMO restart | max abs dT/deta/du/dv all `0.000e+00` | identical, all `0.000e+00` |
| 5-day twin, before-level bridge vs NEMO tb/sb/ub/vb | all `0.000e+00` | identical, all `0.000e+00` |
| 5-day twin, worst array difference base vs HEAD | — | `0.000000e+00` |
| resolved run config | sha256 `b38f00db3fba7b612c68e11fdda5a2a74bdf13fd505785b8794e6a4aa4c11562` | byte-identical |

The lane does not move the DINO twin's numbers. It did break the twin's ability
to run at all.

## Finding 1 — the geometric EOS depth arm (FIXED)

Commit `36d4a2f72` restricted `eos_depth="geometric"` to `eos="nemo_teos10"`.
That rejected all three NEMO-faithful DINO cards — `nemo_paper`,
`nemo_dino_kamm`, `nemo_dino_kamm_mlf` — which select the same geometric depth
ladder with `eos="nemo_seos"`. The guard's intent (phase-3 receipt row 9:
confine the depth arm to cited EOS configurations) is right; its allow-list was
written from this lane's two cases alone.

NEMO's `eos_insitu` passes the live geometric `gdept` in **both** of its
pressure-dependent arms:

```
np_teos10/np_eos80   eosbn2.F90:260   zh = gdept(ji,jj,jk,Knn) * r1_Z0
np_seos              eosbn2.F90:297   zh = gdept(ji,jj,jk,Knn)
                     eosbn2.F90:300   zn = -rn_a0*(1 + 0.5*rn_lambda1*zt + rn_mu1*zh)*zt + ...
```

so `rn_mu1*zh` — the S-EOS thermobaric term — is exactly the consumer. Both
pairs already carry a receipt in this tree:

* `nemo_teos10` — `nemo_testcases_l1_phase3_receipt.md:139` is the arm's
  provenance row; `:73-79` of the same file carries the number, where pinning
  geometric took the LOCK stage-1 u RHS error from `5.6854e-9` to `3.6863e-17`.
* `nemo_seos` — `../dino_tendency_certificate.md:15-16`, "required for the
  S-EOS thermobaric depth term"; certified S-EOS density max abs drho' `1.5e-5`.

Fixed by adding `nemo_seos` to the allow-list with both oracle lines quoted in
the error text. Every other EOS is still rejected; a non-vacuity arm in the new
test pins that `wright` + geometric raises.

## Finding 2 — the trapezoid PGF quadrature arm (OPEN, not fixed here)

The same commit added `pgf_quadrature="nemo_trapezoid"` requires
`pgf_scheme="nemo_sco"`. That rejects the NEMO GYRE card, which pairs the
trapezoid with `pgf_scheme="adcroft"`. Same class as finding 1: NEMO's
e3w-weighted trapezoid quadrature belongs to `hpg_zco`
(`dynhpg.F90:235-302`) as well as `hpg_sco` (`:305-393`), so the premise that
the trapezoid *is* the hpg_sco recurrence is too narrow. The GYRE card built at
the lane base and does not build at HEAD. Left open because widening a second
allow-list needs its own citation and receipt row; carried as
`xfail(strict=True)` in the constructibility test so the fix flips it red.

## Finding 3 — the kt walk cannot run, at base or at HEAD (PRE-EXISTING)

`d180_step_walk.py --phase 1` (the kt=5761..5764 whole-state replay against
NEMO's own restarts) refuses on both sides with
`tke_htau_evaluation='nemo_literal' requires native T-point degree latitudes
carried by the NEMO state bridge`. `multistep_replay.build_replay_ic` does not
pass `carry_native_lat_deg=True` to `bridge_nemo_to_legoesm_topo`, while the
card selects NEMO's literal latitude-dependent etau profile. Identical failure
at `f81436227b1f`, so it is not a lane regression — it is a standing harness
gap that makes the campaign's finest state gate unrunnable. Not fixed here.

## What was run instead, and what it cannot see

`kamm_twin_90d.py nemo_dino_kamm_mlf --days 5 --bridge-before`, the receipts'
own twin runner, on CPU fp64 with `LEGOESM_NEMO_E3T=both` (NEMO's true 3-D
ladder — the gate refuses to default it). 160 leapfrog steps from NEMO's day-180
restart; 153 s wall per arm. Byte-identical invocation on both sides, one
variable: the commit.

```
sha256  610a65c2dd329721b4db19ec279071599192c3d52a9be8e1691ba6623c36c847  twin_BASE_d5.npz
sha256  566560fc8600549461bd222c25c08c70952abdb0885556ac7ac761c2967db42d  twin_HEAD_d5.npz
sha256  ba23e1bb7155e41c992427fcee64f3762231750326cf40f1e7602c5e7f0d5d34  twin_BASE_d5.log
sha256  bd5c00b44c187083937833f7cf55d07dd960ad298f2f5ec39610ba795ab57376  twin_HEAD_d5.log
initial_state_sha256    01ec6db577943529f72a6fb225b8bca4b0af6e1e818f8dfc50b222df619344a8  (both)
vertical_ladder_sha256  9536f62732ab8823b2899d26ae0c4582cca09e4b2f49d9e17ff1234f88584ff9  (both)
```

The two archives differ in exactly one key, `producer_git_sha`; every numeric
array differs by `0.000000e+00`, which is why the file hashes differ and the
content does not.

BLIND SPOT, stated because a clean number invites over-reading: the twin archive
stores `eta`, `sst`, `u`, `v` as float32 SURFACE slices at 5 daily samples. This
comparison therefore resolves a base-vs-HEAD difference only to about 1e-7
relative, and only at the surface — it cannot see a sub-float32 change in the
3-D interior. What carries the "no reach" claim is the pairing of that with the
byte-identical resolved run config, not the surface fields alone.

SKIPPED: the 90-day acceptance gate (ACC at days 30/60/90, the receipts'
headline metric). It needs 2880 steps per arm, roughly 45 minutes of CPU each,
and the 5-day window cannot substitute for it — a 5-day trajectory is far inside
the ACC metric's own 0.091 Sv noise floor, so this reach check makes no claim
about ACC. It claims only that the lane changed no DINO number over 160 steps.

## Finding 4 — a DINO unit test has been red across the whole window (PRE-EXISTING)

`tests/ocean/unit/test_dino_experiment.py::TestSurfaceTendencyPlacement::
test_retention_synthetic_violation_both_directions` fails at all three points,
each time on a different fail-closed requirement, because its synthetic config
carries none of the NEMO bridge arrays the `nemo_dino_kamm` card demands:

| commit | what it trips on |
|---|---|
| base `f81436227b1f` | `redi_flux_face_thickness_evaluation='nemo_qco_live' requires raw NEMO e3t_0` |
| lane `d1e4270ca` | `eos_depth="geometric" is certified only with eos="nemo_teos10"` |
| lane HEAD `ef949da85` | `pgf_scheme="nemo_sco" requires an explicit t_depth_ref` |

Not caused by this change — the fix simply moved it past the EOS guard onto the
next unmet requirement. The signal worth recording is that nobody has been
running it: each new fail-closed requirement silently became the reason it was
already failing.

`tests/test_validate_strict_coverage.py::test_known_unvalidated_is_shrink_only_and_real`
is likewise red at `d1e4270ca`, before this change (stale `grid_type` entry).
Both left for their owners.

## Review

One adversarial reviewer (Claude-authored code, so codex + GLM was the standing
requirement; the codex CLI is unavailable on this account, so this landed with
ONE reviewer, not the required two — stated here rather than implied).

Verdict APPROVE, no Critical or Important findings. The reviewer independently
read `eosbn2.F90` and confirmed the core claim: both the `np_teos10/np_eos80`
arm (`:260`) and the `np_seos` arm (`:297`) consume live `gdept`, and the
geometric-depth mechanism in `compute_ocean_rho` /
`iterate_eos_and_pressure_anomaly` special-cases no EOS. Two citation-precision
suggestions, both checked rather than applied on trust:

* UPHELD — the `5.6854e-9 -> 3.6863e-17` numbers live at `:73-79` of the
  phase-3 receipt, not at `:139`; `:139` is the arm's provenance row. The
  citation is now split accordingly, here and in the code comment.
* REFUTED — the reviewer read the guard's intent row as row 6. It is row 9:
  phase-3 receipt `:220` reads `| 9 | eos_depth="geometric" + a non-NEMO EOS |
  the new depth arm has no cited non-NEMO reference | (a) restrict the arm to
  cited EOS configurations |`. "Row 9" stands, unchanged.
