# NEMO testcase lane 1 — phase-1 receipt

Date: 2026-08-31  
Session: `01a0591f-a335-7060-9257-6c47bf2149ee`  
Oracle: NEMO 5.0.2 commit `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`

## Verdict

**HOLD before phase 2.**  The requested no-fallback rule exposed real oracle
findings rather than a fully certifiable three-run set:

* OVERFLOW-zps and LOCK_EXCHANGE-zco complete their exact 17 h runs with
  `key_qco + key_RK3`, pass exhaustive geometry/coverage, produce all three
  registered fp64 `Nbb` entry dumps, and show the documented gravity-current
  propagation.
* OVERFLOW-sco in the same pinned mode stops at step 4772/6120 under NEMO's own
  `stp_ctl` when bottom velocity reaches the 10 m/s safety limit.  There is no
  final restart or final entry record.  No fallback was attempted.
* **ERRATUM:** the earlier 64-epsilon tracer classification was
  mis-constructed and its DEBT labels are retracted.  The corrected bar is
  relative, tracer-scaled, and uses a `sqrt(N_steps) * eps(fp64)` floor.
  LOCK salinity is AT-BAR; OVERFLOW-zps salinity is UNMEASURED.  Temperature's
  640--796-epsilon relative excess remains UNMEASURED pending attribution,
  including a possible `key_qco` thickness-weighting effect.  Both tracers are
  nevertheless protected by a non-negotiable `1e-6` relative gross-excursion
  hard failure; UNMEASURED cannot hide a physically large limiter violation.
  Top-level `status` summarizes only measured trajectory rows; the independent
  `unmeasured` list carries prose-only coverage gaps without forcing status.
* TEOS-10, `eos_rab`/BN2, FCT2 implicit, adaptive vertical advection, and BBL
  are source/namelist-arm verified.  `rab`, density, BN2, and BBL transport were
  not included in this first entry-dump format, so their numerical-array
  receipts remain **UNMEASURED**.  This is another reason phase 2 is held.

## Build receipt

The first `makenemo` attempt found the system Perl missing `Text::Balanced`.
Adding the already-installed DINO environment's
`lib/perl5/core_perl` to `PERL5LIB` resolved the toolchain issue without
changing model keys.  Both case builds then completed.  The committed
instrument source is identical in both case-specific `MY_SRC` directories.

| artifact | SHA256 |
|---|---|
| conda arch | `64cf1b90f611936bbb92a7514800f8a8e9c8365be7b3cf9a9963f1615c5c836a` |
| OVERFLOW executable | `eb4acf9651b887a3da8834281112d472692caa0bbadcb0d69779e91dee92e6cb` |
| LOCK_EXCHANGE executable | `33c72eb5351c17f0373c6dbf3a59706bbe04b076f3eac6064d1f412d8c9aaf5d` |
| OVERFLOW cpp history | `aab4c8b8f73f57c8abc38ce617386f6fa17085139d91d4ed245511e982fc7c68` |
| LOCK_EXCHANGE cpp history | `72b4ed509ccf2334d14d4b3907c302dfd0167a3a1f24121d983739402312ce95` |
| instrumented `stprk3.F90` | `fc34801ae6855e0c559472be4fd9d1b16858befbc5241c07dd9996a99b0affa6` |

The resolved keys are exactly:

```text
OVERFLOW:      key_qco key_vco_3d key_RK3
LOCK_EXCHANGE: key_qco key_vco_1d key_RK3
```

`key_xios` is absent.  The 64-bit runtime assertion passes in every emitted
record.  The certified DINO executable and `MY_SRC` aggregate reconcile to
their entry hashes, respectively
`00bc1bf78167b6c46c470533955ac39f6e285acb7730dc54a0ef76c770c5373a`
and `8b78ad0f12726f689e95e46f7241af000c4eca5f2f8a5f112a2c570255bc2930`.

## Run and phenomenology receipt

| run | completion | documented-behaviour sanity | strict tracer bar |
|---|---|---|---|
| OVERFLOW-zps | CONFIRM, 6120/6120, final restart | cold-water thickness-weighted centre moves 10.000 to 96.1658 km; final `max(|u|,|v|)=5.29005 m/s` | UNMEASURED: final T excess 796 eps-relative; S excess 91.43 eps-relative exceeds the `sqrt(6120)=78.23` floor |
| OVERFLOW-sco | REFUTE, stop 4772/6120 | NEMO reports `max |U|=10.00 m/s` at `(i,j,k)=(200,2,100)`; abort state retained | UNMEASURED at final; step 1 and 3060 exist |
| LOCK_EXCHANGE-zco | CONFIRM, 61200/61200, final restart | cold-water thickness-weighted centre moves 16.000 to 27.6778 km; final `max(|u|,|v|)=0.932999 m/s` | temperature UNMEASURED at 640 eps-relative; salinity AT-BAR at 152.69 eps-relative below the `sqrt(61200)=247.39` floor |

The trajectory arrays are serial-local `206 x 7 x 101` for OVERFLOW and
`134 x 7 x 21` for LOCK_EXCHANGE because they retain NEMO's two-cell halos.
The corresponding written global meshes are `202 x 3 x 101` and
`130 x 3 x 21`.  Every record is explicitly `Nbb/before`; step 1 has `Nbb=1`
and the midpoint/final records have `Nbb=3` after RK3 swaps.

## Geometry and coverage gate

The gate passes before trajectory interpretation for both completed runs:

| run | mesh | restart | resolved namelist | geometry |
|---|---:|---:|---:|---|
| OVERFLOW-zps | 45/45 disposed | 17/17 disposed | 539/539 disposed | VERIFIED |
| LOCK_EXCHANGE-zco | 35/35 disposed | 23/23 disposed | 538/538 disposed | VERIFIED |

The planted unaccounted mesh name fails with `mesh coverage mismatch`, and a
planted +1 m perturbation of one `e1t` cell fails with `e1t metric`.  The zps
gate verifies the active monotone-topography U/V/F face-min specialization.
It explicitly reports the 10%-minimum `lk_vco_1d3d` arm as
`WAIVED_DEAD_ARM_key_vco_3d`; this corrects the preregistration's initial
dead-arm association rather than certifying code that did not run.

OVERFLOW-sco has a valid mesh and resolved namelist but no final restart, so
the exhaustive run-level gate correctly hard-fails at the missing-restart
condition instead of granting a partial certificate.

## Round-1 review reconciliation and BBL control

The review claim that the BBL parameters match no OMIP configuration is
**REFUTED**.  The actual target
`/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_cfg:284-291`
(SHA256 `7cfe2d47d78a00553cb28fe72c7e2be8655f96f0ea22920f0b8f17f5b2a47a0b`)
contains exactly `ln_trabbl=.true.`, `nn_bbl_ldf=0`, `nn_bbl_adv=2`,
`rn_ahtbbl=1000.`, and `rn_gambbl=20.`.  The original defect was the missing
citation, now present in the dossier and OVERFLOW configurations.

The preregistered sco control changed only `nambbl.ln_trabbl` from true to
false, held the executable and all other resolved settings fixed, and still
failed the same way: NEMO stopped at step 4773 with `|U|max=10.01 m/s` at
`(200,2,100)`, versus step 4772 with BBL enabled.  BBL is therefore
**REFUTED AS SOLE OWNER**.  In accordance with the preregistration, TEOS-10
sigma-coordinate pressure-gradient truncation is the next suspect; no second
arm was run.  The complete machine-readable receipt is
`nemo_testcases_l1_bbl_control_receipt.json`.

The mechanism concern remains valid even though sole ownership is refuted:
`trabbl.F90:415-434` makes option 2 a density-gradient-proportional downslope
transport, and the sco geometry makes its downslope sign mask nonzero across
the interior columns.  Direct BBL transport and mask arrays remain
UNMEASURED in the phase-1 dump.

## Artifact hashes

| artifact | SHA256 |
|---|---|
| zps committed / run / resolved namelist | `cb53829b8b19098a50eeaec49a0d1338bcde8f65fcb991d1f1ce4b228d28b5a3` / `ec1eac4a45fb8c07a0facce5e4eefb6510d8e3f1e364f5c5597e60ae83ccc53e` / `0656e18075595ecf70e2524445574a100b910e05f047b3d52a5a44f32b49c32e` |
| zps stdout / mesh / restart | `bb40d3c52cbe7a86c86703db8c80d10f68b387e3d15a25ae2f3a5f0f19b24949` / `4692b893eddee3eea5cee2e6f04e1d7fc55e6685100e349369051a6914cbd280` / `dab392f2f058b44e8c10c600a41c9be73ba37656e2af478193a3f6f27bd67160` |
| sco committed / run / resolved namelist | `ff220a7befd1899b0fe31066767ac76153e7282fa40e68ffe1c63288d95a922a` / `e14142fdb0ed187b5766d398dd9bd9f2681929e4d965b7a402250e71d01632a3` / `6617a3d769a51c0781971e1b72b7e079b705b20f00b41d004f790e361f953859` |
| sco stdout / mesh / abort | `9b6e392c91eccd53aaf377ce68152bc2b9a14195bb6db61fb6e172e17c6245cd` / `3a5225c64bd0a79c690123e02b6e0e55dd62c9bd37497c2324fa1292d80b2da9` / `9804cc1d151032c94b4003667a1faf8f9ea7cd2bd2120c2faf15d385993d6dbc` |
| LOCK committed = run / resolved namelist | `ae34648ecdf44893e8543f0516511d239ce59161fa5b2de9924f0169d8185dd4` / `a2151e39c484c0065d04b1513ed16246e16dc19f1fcae04e38b7c34d8140f636` |
| LOCK stdout / mesh / restart | `6fad276368f7b4a91d80b38152eb865dcc93bff6d7a011316ef0d674978b5a46` / `ec3200f559cb44ee76d00498aac168dd6452cc29bd0abdf955dfc4ab061ed935` / `15a7883e5e27df2fec36716c29375ec947892a842ee901e5526e10525db5c7a6` |
| sco no-BBL committed / run / resolved namelist | `5bf45231d14a08354d8cc03693680f60375d4972eeab0f4979cf3b5969727e19` / `05250200fb357842049870b12e79e3269e848970a75383d545dc06d842106271` / `fd7e485ed7a84d4856dc4a717749b1afd1450a58747625f1cece766bd1c06ffc` |

The zps/sco committed files intentionally add the post-run ORCA1-OMIP source
citation as comments; the no-BBL committed file differs from the executed copy
only by a stripped final blank line.  The table therefore pins both byte
representations.  Parsed namelist semantics are unchanged.

Trajectory hashes are emitted record-by-record by
`nemo_testcase_oracle_gate.py` and are also covered by the final
run-root checksum manifest.

## Reproduction surface

The committed reproduction surface is intentionally limited to the requested
paths: the dossier and this receipt under `docs/ocean/fidelity/testcases/`; the
exact three namelists, exact case-local `MY_SRC/stprk3.F90`, exhaustive
manifests, and gate under `scripts/validate/ocean_fidelity/testcases/`.  Raw
oracle runs remain under `/data/abyssal/dbalwada/nemo-testcases-l1/`.

## Explicit remaining process debt

These review findings are **DEBT**, not implied certifications:

* Finding 6: boilerplate WAIVED reasons enforce inventory stability only; they
  are not evidence that a human reviewed each array's scientific role.
* Finding 10: the failed sco arm still needs an `--allow-incomplete-run` gate
  mode and a gate-emitted committed partial-run artifact.
* Finding 11: the vendored `stprk3.F90` should be reconsidered as a minimal
  `.patch`; the five full `usrdef_*` hashes per case must be pinned in a
  committed artifact rather than abbreviated dossier prose.
* Finding 12: `cold_center_x_km` is labelled thickness-weighted while the
  implementation uses reference `e3t_0`; its wet-domain semantics should be
  audited against `tmask` explicitly before certification.

## Review status

Independent Claude adversarial review round 1 returned **HOLD**.  The OMIP
parameter objection is refuted above; minimum-to-ship findings 3, 4, 5, 7, 8,
and 9 are implemented and directly tested, while findings 6, 10, 11, and 12
remain explicitly registered debt.  This receipt remains HOLD before phase 2.
