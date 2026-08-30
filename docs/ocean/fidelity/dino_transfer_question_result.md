# DINO transfer question — T3 close and T2 admission finding

Date: 2026-08-30. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

T3 is **TRANSFER_CONFIRMED**. T2 is not yet science-scored: its first two
forward-Euler climate arms exposed and stopped on an illegal MLF time-level
association in the shipped sibling card. This result binds the T3 artifacts,
the T2 failure, and the fail-closed disposition. No GPU, NEMO, or MPI process
was launched while diagnosing or binding these results.

## T3 — catalog construction is certified

The public `nemo_dino_kamm_mlf_v1` recipe and the oracle
`nemo_dino_kamm_mlf` card resolve identically. The config-identity artifact is
`/tmp/dino-transfer-01a053d4/t3/config_identity.json`, SHA-256
`cae74e17dd2e3c60f592c203b398a9d1d15790532be1387090ad9dc35fdd27fe`.
It reports `verdict=PASS`, zero identity-difference rows, a fired one-ULP
`config.asselin_gamma` plant, a fired `outer_integrator` ownership-collision
plant, and a nonempty `nemo_dino_v1` negative control.

The five-day behavioral artifact is
`/tmp/dino-transfer-01a053d4/t3/behavior_identity.json`, SHA-256
`f52990b76afe44cc5a045fd3d2d8e7c10c311a79838bc83296fd6b873cf28169`.
It reports `verdict=PASS`, `bit_differences=[]`, no missing or extra keys,
equal normalized run configs, all required construction receipts present, and
49 compared keys. The oracle artifact SHA-256 is
`d94fd2370413c151723a9986df046d9a5d1051202802b34de2f21b7f9e725f52`;
the catalog artifact SHA-256 is
`fa778f3ed225955cf0f2ae9ac523d48b07b5763840d05fb7d7387726b65c8983`.

Therefore the catalog-built model is bit-identical to the oracle-card model
under the registered bridge and five-day behavioral check. This transfers
construction-path identity for the certified MLF card. It does not extend the
round-94 bridge claim to a standalone start or another card.

## T2 — exact crash and source diagnosis

Both initial `nemo_dino_kamm` climate arms stopped before writing an admitted
artifact. Their logs have SHA-256
`fed80529d1156e424ed5156f445bdfff9f010a7c96d78cf85e3d63831d467e0e`
and `32dc5ee810a8b199ed77b0f6314ec2e9e712caaac3f357041fd348c1f04ed783`.
Equinox's wrapper masked the full stack, but retained the checked message:
`raw-mesh e3w_int must contain only finite values > 0`.

A one-day CPU-only rerun with `JAX_DISABLE_JIT=1`, fp64, and the same bridge
arguments produced `/tmp/t2-fe-cpu-repro.log`, SHA-256
`7c400095929a00cc2b3f53569f75e6742e2dc01dc12fb67863477bac6fb9722e`.
The unwrapped stack reaches `compute_buoyancy_frequency_nemo_bn2` from
`compute_nemo_native_slopes` after the split-explicit update. The raise is a
downstream safety check: live `e3w_int` is already non-positive there.

The checked e3w error is downstream. A barotropic-return probe records finite
SSH after steps 1--3 but explosive growth:
`|eta|max = 0.828586, 2.865547, 95.591409 m`; step 4 is non-finite. That probe
is `/tmp/t2-fe-baro-values.log`, SHA-256
`ff1c2f06f69acf54aebc741e2c5d815f2e4814a5549170f8e2161a273e33bafe`.
The next GM/Redi live-QCO geometry calculation correctly turns the non-finite
SSH into non-finite e3w and fires its positivity check.

Three controls isolate the owner. Removing the BEFORE bridge reproduces the
same sequence (`/tmp/t2-fe-no-before.log`, SHA-256
`27e5973274571d7569c22063a4383ba052e3d2e43689b3bf528ed76e5a5ddefc`).
Reverting all five literal barotropic arithmetic selectors to `generic` also
reproduces it (`/tmp/t2-fe-all-generic-baro.log`, SHA-256
`1adf6feb17c884d0da56d6f675c7208f33361baa4a0dc455ae8be0b2e0d7eb4b`).
Changing exactly `barotropic_diffusion_alpha` from `0.0` to the FE card's
historical `0.01` keeps all five checked steps finite and bounded at
`|eta|max <= 0.827309 m` (`/tmp/t2-fe-alpha-control.log`, SHA-256
`51e2cbd2352e452ca0d47f37e2df2462fa2618013bb8a354ce04dae3917e1c98`).
The owner is therefore zero barotropic damping on the permanent-FE frame, not
the bridge or a literal arithmetic association.

## Disposition and T2 scope

There is no permanent-FE DINO oracle branch from which to derive an equivalent
zero-damping composition. NEMO's MLF `dynspg_ts` contains no legoESM spatial
SSH-diffusion term, but the FE approximation requires its historical `0.01`
2-dx stability crutch. Zero `barotropic_diffusion_alpha` is therefore
MLF-only. The audit also hardens the independently true Nbb/Kaa scope of the
literal `zad_qco_evaluation` and `wzv_call2_evaluation` pair:

- `nemo_dino_kamm_mlf` retains both `nemo_literal` selectors and alpha `0.0`,
  so the certified MLF and catalog configurations are unchanged;
- `nemo_dino_kamm` resolves both selectors to `generic` and restores alpha
  `0.01`;
- model construction rejects a planted literal pair unless
  `outer_integrator` is `leapfrog` or `nemo_mlf`;
- model construction rejects the exact Kamm-FE/centred-boxcar/alpha-zero
  combination before tracing;
- the re-battery admission table is card-aware for exactly these three fields
  and hard-checks the artifact's stamped recipe. Its statistics, reducers,
  floors, and science bars are unchanged.

This is a real negative transfer finding: the full MLF faithful-default bundle
as previously shipped is not legal on the FE sibling. A corrected FE battery
can characterize the named sibling card as a bundle, but it cannot claim that
the MLF-only time-level selectors transferred. T2 remains **NOT RUN / NOT
SCIENCE-SCORED** until the corrected duplicate arms in the handoff complete.

## Block-0 admission correction

The first rerun stopped in
`TestSurfaceTendencyPlacement::test_retention_synthetic_violation_both_directions`
before either registered retention assertion. The production surface-placement
path had not regressed, and none of the T2 card fences caused the exception.
The synthetic ten-column fixture inherits the faithful MLF card but has no raw
bridged NEMO `e3t_0`; commit `d1cf21b425` later promoted
`gm_redi_flux_face_thickness_evaluation="nemo_qco_live"` without adding the
matching generic fixture override. The resulting exception was
`redi_flux_face_thickness_evaluation='nemo_qco_live' requires raw NEMO e3t_0`.

The test setup now pins
`gm_redi_flux_face_thickness_evaluation="tpoint_jacobian"`, paired with its
existing synthetic-grid slope and QCO overrides. Production code is unchanged.
The isolated test passes and still executes both original plants:
`applied_now` must retain below `0.6`, while `leapfrog_rhs` must retain above
`0.6`. The complete Block-0 admission suite passes `178/178` after this
fixture repair.
