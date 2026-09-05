# NEMO testcase lane 2: GYRE phase-1 closure receipt

**Verdict: VERIFIED for the registered phase-1 claim.**  The physics-only,
single-CPU NEMO 5.0.2 GYRE vehicle completed 4,320 WS-RK3 steps (two 360-day
years), emitted all three fp64 lane-1-format step-entry dumps, and passed the
fail-closed geometry, resolved-selector, inventory, initial-condition, and
seasonal-spinup gates.  This phase makes no legoESM matching claim.

## Provenance and base

- Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`.
- Branch: `fidelity/nemo-testcases-l2-gyre-codex`, based on lane 1 commit
  `3d609df4c413d236325d9319f6d93b97dc16fe2a`, because lane 1 was not in
  `origin/main` when this lane began.
- NEMO source: `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2`, commit
  `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`.  Its pre-existing dirty files
  were not used as configuration inputs; source/config inputs were copied into
  the run's `provenance/` directory and hashed.
- Run root: `/data/abyssal/dbalwada/nemo-testcases-l2/gyre`; CPU serial,
  `OMP_NUM_THREADS=1`.  The instrumented executable SHA256 is
  `3e7b30256344bdca1f9d99da9101adf94e24062e959416ae25f0852291a0b36f`.
- The exact executed configuration is
  `scripts/validate/ocean_fidelity/testcases/configs/gyre_omip_l2_namelist_cfg`
  (SHA256 `9259268a28f6cc5d6e5c8aeaa445567156250b62e31e2f1bc02941f476e9d4dc`).
  Recursive hashes for the run and every copied provenance input are in
  `nemo_testcases_l2_gyre_run_sha256.txt`.

Live issue/PR state was unavailable from this environment, so issue #1455,
campaign issue #1699, and PR #1700 comments are **UNVERIFIED** here.  No remote
claim was posted.

## Case dossier: source-resolved GYRE

The running `usrdef_*` source is the OCE GYRE implementation:

- `src/OCE/USR/usrdef_nam.F90:64-100` reads `namusr_def`, resolves
  `30*nn_GYRE+2` by `20*nn_GYRE+2`, takes `jpkglo`, and closes both horizontal
  directions.  With `nn_GYRE=1`, `jpkglo=31`, this is 32 x 22 x 31.
- `src/OCE/USR/usrdef_hgr.F90:75-170` constructs the regular 106 km grid,
  rotates it 45 degrees, and supplies the northern beta plane.
- `src/OCE/USR/usrdef_zgr.F90:60-207` selects the flat z-coordinate box and
  the 31-level MI96 depth/thickness ladder; 30 levels are wet.
- `src/OCE/USR/usrdef_istate.F90:41-101` defines the double-gyre initial state:
  horizontally uniform temperature/salinity profiles, rest, and zero SSH.
- `src/OCE/USR/usrdef_sbc.F90:40-53,77-145,148-220` identifies and implements
  the analytic seasonal momentum, heat, and freshwater forcing (Hazeleger and
  Drijfhout 2000).

The resolved `nn_itend=4320`, `rn_Dt=14400`, `nn_GYRE=1`, and `jpkglo=31`
are shipped GYRE defaults; 4,320 four-hour steps are 720 days, exactly two
360-day years.

Physics-only compilation used OCE with `key_qco key_vco_1d3d key_RK3`.  These
are the ocean keys in `/data/abyssal/dbalwada/ORCA1-omip/cpp_ORCA1.fcm`; ice,
ISF, TOP, and XIOS were omitted.  The shipped PISCES selector explicitly has
`ln_p4z=.false.` at `cfgs/GYRE_PISCES/EXPREF/namelist_pisces_cfg:5`, and no TOP
component was built or executed.  For precision: the shipped file also has
`ln_p2z=.true.` at line 4, so `ln_p4z=.false.` disables standard P4Z but would
select reduced P2Z if TOP existed; omission of `key_top` is what makes this
vehicle biology-free.

**Deliberate OMIP-style deviation:** the shipped GYRE namelist actually selects
EOS-80 (`cfgs/GYRE_PISCES/EXPREF/namelist_cfg:124-127`), not S-EOS; this vehicle
records `ln_teos10=.true., ln_eos80=.false., ln_seos=.false.` for OMIP coverage.

The qco nonlinear free surface also requires the lane-1 canonical terrain-
following pressure-gradient arm: `ln_hpg_zco=.false., ln_hpg_sco=.true.`.  The
shipped zco arm was tried once and NEMO stopped before step 1 with
`dyn_hpg_init : non-linear free surface incompatible with hpg_zco`; that failed
initialization is preserved under `failed_init_hpg_zco/` and was disclosed in
the amended preregistration.  All other resolved GYRE choices remain shipped
choices: vector momentum advection, ENE vorticity, FCT2 tracers, Demange TKE,
and the three-pass barotropic filter (`rn_bt_alpha=0.07`).

## Build, run, and dump receipt

The build used the lane-1 external `MY_SRC/stprk3.F90` byte-for-byte (SHA256
`fc34801ae6855e0c559472be4fd9d1b16858befbc5241c07dd9996a99b0affa6`), including
magic `NEMO_L1_ENTRY_1`, header layout, Fortran-order payload, and exact
`Nbb`/before-level semantics.  An initial completed executable did not include
that instrument because an incremental build retained the stock object; it is
preserved under `uninstrumented_completion/` but excluded from certification.
After invalidating the exact `stprk3` build products, the relinked executable
changed hash and printed all dump markers.

The certified integration ran from 2026-09-01 07:42:57 to 07:43:56 EDT
(58.755 s) and ended normally after writing
`GYRE_OMIP_L2_00004320_restart.nc`.  Dumps are present at steps 1, 2160, and
4320; each is 936,048 bytes, fp64, and carries `(jpi,jpj,jpk,jpts)=(36,26,31,2)`.
Their SHA256s are respectively `9def5f4986853f497a5e0507bea185fe1ec4348715e2aeca0b14507df8e24fa0`,
`e1e8340b95cf98ecfb2ea80b53067b0ed142c59980d0d5ec4b3d44c07666923e`,
and `d07f0bd32cdde9cd2ca98982e1095eb8d4f25c9d1dc89866f09539a9ecf3b587`.

## Gate and controls

`nemo_testcase_oracle_gate.py` was extended, rather than replaced.  Its
committed manifest accounts exactly once for 39 mesh arrays, 28 restart arrays,
and 580 resolved namelist selectors.  It verifies the exact build keys,
lane-1 instrument hash, clean normal-completion log, final step/date/restart
dimensions, constant 106 km metrics, every staggered closed mask, the rotated
affine coordinates, beta-plane affinity, the source-recomputed MI96 ladder and
its 1-D/3-D identities, TEOS-10 and RK3-compatible selectors, fp64 dump
structure, the analytic IC, and gross spinup.

The preregistered pointwise depth/thickness identities are enforced at
`1e-15` (and the 3-D/1-D thickness copies are exact).  Separately, reevaluating
the MI96 transcendental formula in NumPy differs from NEMO's optimized Fortran
libm by at most `4.166623822316079e-14` pointwise; that value is reported as a
libm diagnostic with a `512*fp64_eps` portability guard, not substituted for
or used to relax the preregistered identity bar.

At step 1, wet velocities and SSH are exactly zero and each wet tracer level is
horizontally uniform.  By step 4320, max wet speed is
`0.3529080118695336 m s-1` and SSH spans
`[-0.3934087791258727, 0.4516915706194917] m`; the wet mean is `+0.1551752 m`
south of 30 N and `-0.3004181 m` north of 40 N.  This non-vacuous spatial
split is consistent with the documented double-gyre spinup under analytic
seasonal forcing.  It does not claim to measure the forcing's seasonal phase
response.  Full machine output is in
`nemo_testcases_l2_gyre_phase1_receipt.json`.

Both preregistered controls fired:

- planted `+1 m` in `e1t`: exit 1, `DEBT: e1t metric`;
- planted file-side unregistered mesh array: exit 1,
  `DEBT: mesh coverage mismatch: missing=['PLANTED_UNACCOUNTED_FILE_ARRAY'], extra=[]`.

Two Codex-internal adversarial passes initially returned HOLD on provenance,
schema/topology, MI96-bar, and phenomenology non-vacuity gaps.  Every finding
was dispositioned in the gate or receipt; both internal passes then reran the
gate and returned APPROVE.  They are not the repository's required dual
review.  Claude reviewed the resulting Phase-1 package afterward and returned
**SHIP** in the present review round, conditional only on the corrections now
recorded here.  The independent GLM review remains outstanding.

The remaining items are explicitly outside the phase-1 claim and stay in the
machine receipt's `unmeasured` ledger: TEOS-10 density values, `rab`, `bn2`,
adaptive vertical-advection partition, BBL transport/mask geometry, and global
tracer-inventory closure.  Geometry -> IC -> `kt=1` -> first-divergence
matching against legoESM begins only in the next phase.
