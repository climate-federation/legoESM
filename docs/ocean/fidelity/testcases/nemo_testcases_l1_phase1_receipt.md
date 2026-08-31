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
* The completed runs exceed the preregistered 64-epsilon closed-tracer bar by
  order `10^-12`.  Those values are scientifically tiny but are **DEBT**, not
  relabelled matches.
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
| OVERFLOW-zps | CONFIRM, 6120/6120, final restart | cold-water thickness-weighted centre moves 10.000 to 96.1658 km; final `max(|u|,|v|)=5.29005 m/s` | DEBT: final T `[13.620424680472283, 20.000000000003535]`; S `[34.999999999999496, 35.00000000000071]` |
| OVERFLOW-sco | REFUTE, stop 4772/6120 | NEMO reports `max |U|=10.00 m/s` at `(i,j,k)=(200,2,100)`; abort state retained | UNMEASURED at final; step 1 and 3060 exist |
| LOCK_EXCHANGE-zco | CONFIRM, 61200/61200, final restart | cold-water thickness-weighted centre moves 16.000 to 27.6778 km; final `max(|u|,|v|)=0.932999 m/s` | DEBT: final T `[4.999999999999828, 30.000000000004263]`; S `[34.99999999999881, 35.00000000000102]` |

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

## Artifact hashes

| artifact | SHA256 |
|---|---|
| zps namelist / resolved namelist | `ec1eac4a45fb8c07a0facce5e4eefb6510d8e3f1e364f5c5597e60ae83ccc53e` / `0656e18075595ecf70e2524445574a100b910e05f047b3d52a5a44f32b49c32e` |
| zps stdout / mesh / restart | `bb40d3c52cbe7a86c86703db8c80d10f68b387e3d15a25ae2f3a5f0f19b24949` / `4692b893eddee3eea5cee2e6f04e1d7fc55e6685100e349369051a6914cbd280` / `dab392f2f058b44e8c10c600a41c9be73ba37656e2af478193a3f6f27bd67160` |
| sco namelist / resolved namelist | `e14142fdb0ed187b5766d398dd9bd9f2681929e4d965b7a402250e71d01632a3` / `6617a3d769a51c0781971e1b72b7e079b705b20f00b41d004f790e361f953859` |
| sco stdout / mesh / abort | `9b6e392c91eccd53aaf377ce68152bc2b9a14195bb6db61fb6e172e17c6245cd` / `3a5225c64bd0a79c690123e02b6e0e55dd62c9bd37497c2324fa1292d80b2da9` / `9804cc1d151032c94b4003667a1faf8f9ea7cd2bd2120c2faf15d385993d6dbc` |
| LOCK namelist / resolved namelist | `ae34648ecdf44893e8543f0516511d239ce59161fa5b2de9924f0169d8185dd4` / `a2151e39c484c0065d04b1513ed16246e16dc19f1fcae04e38b7c34d8140f636` |
| LOCK stdout / mesh / restart | `6fad276368f7b4a91d80b38152eb865dcc93bff6d7a011316ef0d674978b5a46` / `ec3200f559cb44ee76d00498aac168dd6452cc29bd0abdf955dfc4ab061ed935` / `15a7883e5e27df2fec36716c29375ec947892a842ee901e5526e10525db5c7a6` |

Trajectory hashes are emitted record-by-record by
`nemo_testcase_oracle_gate.py --report-only` and are also covered by the final
run-root checksum manifest.

## Reproduction surface

The committed reproduction surface is intentionally limited to the requested
paths: the dossier and this receipt under `docs/ocean/fidelity/testcases/`; the
exact three namelists, exact case-local `MY_SRC/stprk3.F90`, exhaustive
manifests, and gate under `scripts/validate/ocean_fidelity/testcases/`.  Raw
oracle runs remain under `/data/abyssal/dbalwada/nemo-testcases-l1/`.

## Review status

Two independent read-only adversarial-review sessions were launched as
required by the campaign rules.  Both failed before reading the changes because
the reviewer API endpoint was unreachable from the sandbox (`ENOTFOUND` /
network denied).  There is therefore **no independent scientific review
receipt** on this commit.  The HOLD verdict already forbids using these numbers
as a phase-2 oracle certificate; review must be rerun when connectivity is
available.
