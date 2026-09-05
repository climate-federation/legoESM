# NEMO testcase lane 2 — GYRE phase-1 preregistration

Date: 2026-09-01

Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`

Status: **PREREGISTERED BEFORE THE LANE-2 BUILD OR RUN.**  This phase certifies
only the NEMO 5.0.2 oracle configuration, execution, step-entry dumps, geometry,
and coverage inventory.  It makes no legoESM matching claim.

## Base and immutable provenance

Lane 1 is not merged in `origin/main`.  Lane 2 is based on the lane-1 tip
`3d609df4c413d236325d9319f6d93b97dc16fe2a` and reuses its binary entry-dump
format, fail-closed manifest, parser, gate structure, and planted controls.
The managed sandbox makes the shared worktree Git metadata read-only, so the
branch ref is maintained in writable local Git metadata at
`/tmp/codex-gyre-local-git`; the requested bundle is emitted from that ref.

| item | pinned value |
|---|---|
| NEMO tree | `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2` |
| NEMO commit | `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796` |
| architecture | `arch-conda.fcm`, SHA256 `64cf1b90f611936bbb92a7514800f8a8e9c8365be7b3cf9a9963f1615c5c836a` |
| arithmetic | `REAL(wp)` with the lane-1 runtime `STORAGE_SIZE(1._wp)=64` assertion |
| shipped GYRE namelist | SHA256 `1f5c4257b739efc9f890bf9358932c10647f9e9b4037cd40d9d2cdeea6fc1ed9` |
| ORCA1 key card | `/data/abyssal/dbalwada/ORCA1-omip/cpp_ORCA1.fcm`, SHA256 `1f475adc3868d77277c20b460dec83ed558ce55c1fe5baca321739629389d108` |
| execution | one CPU process, no GPU and no `mpirun`; root `/data/abyssal/dbalwada/nemo-testcases-l2/` |

The oracle checkout was already dirty before this lane.  Existing changes are
not treated as inputs: a new configuration name beginning `GYRE_OMIP_L2` is
created from the tracked `GYRE_PISCES` reference, and all executed inputs are
copied and hashed under the lane-2 run root.

## Source dossier

`usrdef_nam.F90:64,70,79-100` reads `nn_GYRE`, `ln_bench`, and `jpkglo`, sets a
closed `32 x 22 x 31` domain at `nn_GYRE=1`, and leaves 30 wet levels plus the
dummy bottom record.  `usrdef_hgr.F90:75-91,119-170` constructs a 45-degree
rotated, constant-106-km beta-plane grid.  `usrdef_zgr.F90:64-88,93-175,178-207`
selects a flat full-step z-coordinate, constructs the 31-level Madec-Imbard
ladder, and wets levels 1--30 inside the four closed walls.

`usrdef_istate.F90:55-77,82-101` starts from rest and zero SSH with horizontally
uniform analytic temperature and salinity profiles.  `usrdef_sbc.F90:40-53,
77-145,148-220` applies analytic seasonal heat, freshwater, and double-gyre
wind forcing on the 360-day calendar.  The configured 4320 steps at 14400 s
span exactly 720 days, hence two complete forcing years.

Physics-only means no executed TOP/PISCES step.  The shipped
`GYRE_PISCES/EXPREF/namelist_pisces_cfg:4-5` sets `ln_p2z=.true.` and
`ln_p4z=.false.`; therefore `ln_p4z` alone disables standard P4Z, not all
biology.  The OMIP-style build omits `key_top`, so neither P2Z nor P4Z is in
the executed call path.
`trcnam_pisces.F90:46,58-85` is the source proving `ln_p4z` is the standard
PISCES selection toggle.

**Deliberate OMIP-style EOS deviation (one line): shipped NEMO 5.0.2 GYRE
EOS-80 (`namelist_cfg:124-127`) -> `ln_teos10=.true.`, with EOS-80 and S-EOS
false.**

## Exact build and resolved configuration

The tracked ORCA1 card contains `key_si3 key_xios key_qco key_vco_1d3d key_RK3
key_isf`.  Removing sea ice and ice shelf as requested, removing `key_top` for
physics-only execution, and applying lane 1's toolchain-only XIOS omission
leaves exactly:

```text
key_qco key_vco_1d3d key_RK3
```

The run otherwise retains the shipped GYRE dynamics and forcing: free-slip
walls, FCT2 active tracers, vector momentum with C2 kinetic energy and ENE
vorticity, the lane-1 canonical `hpg_sco` pressure gradient required by qco,
split-explicit Demange filter 3,
level Laplacian momentum diffusion, isoneutral tracer diffusion, TKE plus EVD,
nonlinear bottom drag, and the analytic surface boundary condition.  The
resolved `output.namelist.dyn`, `cpp.history`, executable, copied inputs,
stdout, mesh, final restart, and dump files must all be SHA256-pinned.

## Step-entry contract

The lane-1 `stprk3.F90` instrumentation is reused byte-for-byte, including its
`NEMO_L1_ENTRY_1` magic and lane-1 marker label.  It writes
`ts(:,:,:,:,Nbb)`, `uu(:,:,:,Nbb)`,
`vv(:,:,:,Nbb)`, and `ssh(:,:,Nbb)` before calendar/forcing, `stp_2D`, or any
RK stage.  Records are preregistered at steps `1`, `2160`, and `4320`; each
must carry the magic/version, step, `Nbb`, local dimensions, `jpts=2`, and
64-bit storage.  The instrument is write-only.

## Fail-closed gate and registered outcomes

The lane-1 registry rule is unchanged: every variable discovered in the mesh,
final restart, and resolved namelist occurs exactly once as `VERIFIED`,
`WAIVED`, or `UNMEASURED`, with a nonempty reason.  Missing or extra entries,
hash drift, non-finite verified data, wrong dtype/dimensions, wrong selectors,
or absent final restart hard-fail.

Preregistered classifications:

1. **CONFIRM build/run** iff `cpp.history` resolves the compiled cpp file and
   its `fppkeys` line is exactly the three keys above, the executable exits
   zero at step 4320, a final restart exists, all three fp64
   entry records exist, and stdout has no `ctl_stop`, NaN, infinity, or floating
   exception.  Otherwise **REFUTE** with no fallback.
2. **CONFIRM geometry** iff dimensions are `32 x 22 x 31`; all eight horizontal
   metrics are 106000 m; the rotated coordinate increments are affine; Coriolis
   is a nonconstant affine beta-plane function; walls are closed; the interior
   has exactly 30 wet levels; the bottom is flat; and the 1-D/3-D depth and
   thickness identities clear the fp64 pointwise `1e-15` normalized bar.
3. **CONFIRM IC/entry** iff the step-1 dump is finite fp64, its wet T/S fields
   are horizontally uniform, u/v/SSH are exactly zero, and its halo-stripped
   dimensions match the written mesh.  Otherwise **REFUTE**.
4. **CONFIRM seasonal gyre spin-up sanity** iff the two-year run completes,
   the final wet state is finite, final velocity and SSH are nonzero, and final
   SSH contains both positive and negative interior anomalies.  Values are
   reported; this is a gross phenomenology sanity, not a trajectory bar.
5. **CONFIRM coverage controls** iff a planted file-side unaccounted mesh array
   fails with a coverage mismatch and a planted +1 m nonzero `e1t` perturbation
   fails the metric check.  A zero perturbation is forbidden.

Geometry/coverage sees the oracle artifact inventory and analytic mesh
identities; it is blind to whether legoESM reproduces them.  The phase therefore
leaves legoESM geometry, IC, kt=1 parity, RK stages, tendencies, and trajectory
**UNMEASURED**.

## Pre-step source-resolution correction

The first executable invocation stopped during initialization, before step 1
and before any entry dump, with `dyn_hpg_init : non-linear free surface
incompatible with hpg_zco`.  `dynhpg.F90:188-197` enforces one pressure-gradient
arm and `:117-123` dispatches `ln_hpg_sco` to `hpg_sco`; lane 1 already certifies
that canonical qco arm.  The successful run is therefore preregistered with
the shipped `ln_hpg_zco` changed true -> false and `ln_hpg_sco` false -> true.
This is the only additional scientific selector change.  The stopped
initialization produced no model-step measurement and is retained as a failed
bring-up artifact.
