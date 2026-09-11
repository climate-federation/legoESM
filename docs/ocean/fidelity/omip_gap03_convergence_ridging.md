# Gap 3: selectable convergence-only tripole ridging

Author: Codex. Uncommitted implementation requested by the user; review belongs
to Claude and GLM. This is a bounded capability addition, **not closure of gap 3
and not a claim of SI3 equivalence**. No simulation or production card was run
or changed. No climate measurement is reported.

## Inventory established before implementation

Searches: `rg -n 'EVP|evp|ridg|landfast|free_drift|tripol' packages/ice`,
`rg -n 'landfast|land_fast|basal.*stress' packages src`, and searches for
`strain_rate_cgrid`, `stress_divergence_cgrid`, and the OMIP ice call sites.

- `ice/rheology.py`: VP constitutive law, EVP/mEVP stress relaxation, strain
  operators for `CubedSphereGrid`, regular `LatLonGrid` (A-grid), and
  `VoronoiMesh` (MPAS). These are not tripole C-grid ice operators.
- `ice/dynamics.py`: EVP/mEVP momentum solvers, stress divergence for those
  three grid types, optional sea-surface tilt at the solver API, and diagnostic
  free drift. The tripole geometry is explicitly unsupported.
- `ice/ridging.py`: grid-independent Lipscomb-style category redistribution,
  including ice-volume and salt conservation and snow/pond freshwater returns.
  Reused unchanged. It is not NEMO's full ridging/rafting implementation.
- `ice/sea_ice.py`: full closing rate from convergence plus shear, then the
  existing redistribution; previously restricted to dynamics-capable grids.
- `ice/transport.py`: geographic cell velocities, rotation onto C-grid faces,
  shared fold-pair flux, donor-cell transport, and conservative divergence.
  Already works on `LatLonCGridGeometry`; reused unchanged.
- Ocean `latlon_cgrid_operators.py` already has strain/stress operators.
  A faithful ice solver must establish compatible T/F/U/V placement, tensor
  transformations and boundary conditions, and add the ice momentum/state
  integration. Merely passing the tripole object into the existing ice solver
  would not implement that chain.
- No landfast implementation was found under `packages` or `src`.
- CORE-II OMIP already exposes categories and ridging, but refused tripole
  ridging and falls back from requested EVP/mEVP to free drift there.

## Oracle source inspection and limits

The house-rule NEMO 5.0.2 location
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2` does not exist on this host.
The accessible source is
`/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1`.
This version difference is explicit; no 5.0.2 reproduction is asserted.

The local ORCA1 configuration
`/burg-archive/glab/users/pg2328/nemo_orca1/ORCA1/EXPREF/namelist_ice_cfg`
selects full dynamics at line 44, Lemieux landfast at line 46, and EVP at
line 70. In the accessible source:

- `src/ICE/icedyn_rdgrft.F90:239`: net closing is
  `Cs/2 * (Delta - abs(div)) - min(div, 0)` for EVP/VP. The implemented subset
  retains only `-min(div, 0)`. NEMO also treats open-water participation and
  rafting; those were not ported.
- `src/ICE/icedyn_rhg_evp.F90:257`: landfast changes tensile strength;
  line 335 additionally enables bathymetry/thickness-dependent bottom stress,
  with masks and stresses at the momentum faces. It is not a velocity clamp
  that can faithfully be attached to diagnostic free drift.

These are source/config observations, not verification of a running binary's
preprocessor state. GitHub issue #1455 could not be read (network connection
failed); no issue comment was posted.

## Implemented scope

`RidgingConfig.closing_scheme="convergence"` selects convergence-only closing.
On tripole, the tendency of a unit scalar from existing ice transport is
`-div(u_ice)`: positive means closing. Clip it to `[0, closing_rate_max]` and
pass it to the existing conservative category redistribution. On the previously
supported grids use the trace of their existing strain tensor, without shear.
Unknown selections raise at both the public step and closing helper.

CORE-II selection, in addition to the caller's other run options:

```text
--prognostic-sea-ice --prognostic-ice-dynamics free_drift
--ice-categories 5 --ice-ridging --ice-ridging-closing-scheme convergence
```

This is a usage example, not a selected experiment. Categories can be any
supported count >= 2; 5 is illustrative. The CLI validates the new selection,
refuses it without ridging/prognostic ice, checks grid support, and forwards
it into `RidgingConfig` and the actual closing-rate call. Direct API use of
convergence ridging also refuses a single-category no-op. FESOM already rejects
this selector through its existing unwired-option allowlist; a dedicated test
isolates this selector so another unsupported flag cannot hide a missing guard.

## Deliberately left out

- Tripole EVP/mEVP and full shear ridging: need the complete staggered tensor
  and momentum chain; not a wiring-only task. Their ice-API guards remain.
- Landfast tensile strength and basal stress: require the omitted momentum
  solve, bathymetry, and face masks. No clamp or surrogate was added.
- Coastal impermeability: inherited ice transport currently has no ocean land
  mask. This subset is a local redistribution capability on that transport
  geometry; it does not certify coastal or Arctic-basin dynamics. Fold handling
  is inherited from the shared transport operator, not duplicated.
- NEMO open-water participation, rafting, and redistribution details remain
  different from the existing Lipscomb-style kernel.
- No distributed integration or real ORCA climate validation was run. Tests
  exercise an active synthetic fold with nonzero seam width; they do not
  establish real-mesh climate fidelity.
- Existing EVP iteration-count config fields are pre-existing debt. This
  subset introduces no iterations, counts, or relaxation parameters and does
  not modify EVP/mEVP APIs. Their alpha/beta/T_evp tuning exclusions remain.

## Defaults and verification

Every existing config/CLI default and the production card remain unchanged.
The new selection defaults to `strain`, preserving existing shear-plus-
convergence behavior and the existing tripole rejection unless the user
explicitly selects the subset. Prognostic ice, category count, dynamics,
transport, ridging enablement, thermodynamics, coefficients, iteration counts,
and tuning exclusions are not changed.

Tests cover analytic closing/opening signs and clipping, rotated velocities,
nonzero paired seam fluxes, finite/nonzero gradients versus finite differences
in float32/float64, JIT parity, real-step volume/salt conservation, unsupported
mechanics rejection, and all forwarding hops. Driver forwarding tests execute
the actual `main()` AST statements, not copies of the config construction.
Mutation checks replace production implementations in memory, leaving the
working diff untouched; logs are under `/tmp/gap03_mutations/` for this session.

Review is **UNREVIEWED** until Claude and GLM return actual verdicts. The initial
Claude invocation timed out without a review; the diff-review attempt in minimal
mode reported "Not logged in". GLM diff review failed DNS resolution. These are
blocked review attempts, not approvals.


## Validation receipts (CPU)

All pytest invocations used `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1` and
`/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python` (this checkout
has no `.venv/bin/python`). The temporary runner `/tmp/run_gap03_tests.py`
prepends this checkout's `src`, root and package directories to `sys.path`,
then calls pytest. No installed-source substitution was used.

- New convergence-ridging tests: **16 passed**, including full-step JIT parity.
  The later added active-fold JIT assertion was also rerun: **1 passed**
  (the same test, not an additional unique test).
- CORE-II category/flag tests: **29 passed**.
- Dispatch ratchet: **24 passed**.
- Existing C-grid transport and ridging-kernel tests: **30 passed**
  (9 transport, 21 ridging).
- Existing extended ice physics: **68 passed**.
- Existing dynamics regression rerun: **121 passed, 1 deselected** (the
  100-step mEVP integration). This brings the unique relevant test total to
  **288 passed**; repeated focused runs are not added to this total.
- Source hygiene/parameter/contract audit: **4,995 passed, 20 failed, 2 skipped**.
  All 20 failing cases were rerun against an untouched `git archive HEAD`
  snapshot: **20 failed**, confirming existing debt. Failures are in unrelated
  FV3/ocean/constants/private-import/contract sites. The baseline log is
  `/tmp/gap03_baseline_audit.log`.
- **21/21 production mutations detected**, including all CLI -> resolver/config
  -> closing-helper forwarding hops, each closing branch, signs/cap, input
  guards, and the FESOM allowlist. The mutation runner updates package
  re-exports too, so tests cannot accidentally exercise cached original code.
- Baseline comparison: **74 existing nested ice config defaults and 204
  existing CLI defaults unchanged**. Only the new closing selection is added,
  defaulting to `strain`. `/tmp/gap03_check_defaults.py` compares the actual
  values against the untouched HEAD snapshot.

The combined broader dynamics run was interrupted during the unrelated
100-step mEVP integration; that test is unverified, not counted as passed.

The dynamics rerun completed with `-k 'not TestMEVPLongRun'`; its log is
`/tmp/gap03_dynamics_tests.log`. No test tolerance, skip marker, production
coefficient, or default was changed to obtain these results.

Choices: ASKED — add an explicitly selectable, correct subset and preserve
all defaults/production cards. UNASKED scientific selections: none. No commit.

UNVERIFIED: Claude/GLM approval, the interrupted 100-step mEVP test, real
ORCA-mesh/distributed integration, and NEMO 5.0.2 statistical equivalence.
