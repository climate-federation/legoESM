# GYRE high-resolution round 17 receipt

Date: 2026-10-19.  Status: **HELD FOR ACQUISITION**.  This round builds the
smallest common-state test of the round-16 vorticity/Coriolis carrier.  It does
not name a physics owner and changes no production physics.

## Finding, disposition, and evidence

Round 16 is **CONFIRMED** as a term-carrier measurement.  In the days 30--180
western-boundary/native-surface-V signed mean, legoESM minus vector-math NEMO
was `1.6614925847170393e-6 m s-1`, while scalar-math minus vector-math NEMO was
`3.567844130008744e-8 m s-1`: `46.56853057964074x` the arithmetic twin.  The
source artifact is
`gyrehr_rounds/round16/budget_attribution_with_scalar_twin.json`, SHA-256
`403297d6ebebfe8d0d567100d883542eb9a010c35d8eb208d2b448f76aaf0645`.
That measurement identifies `total_vorticity_coriolis` as the carrier; its own
`physics_owner` field is `UNMEASURED`.

The next owner remains **UNMEASURED**.  Neither registered round-17 NEMO root
exists, so the preregistered common-state q substitution cannot be scored.
There is no evidential basis to change the ENE q formation, metric transports,
or completed stencil.  No physics or configuration change follows.

The committed proving test does the following:

1. Starts vector-NEMO, scalar-NEMO, and legoESM from the same admitted
   vector-math day-180 restart at step 4,320 (SHA-256
   `a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12`).
2. Runs one `r4_smag` step, single rank, and records stage 3 immediately before
   and after `dyn_vor`, plus each stage's formed q and metric transports.
3. Substitutes only scalar-NEMO's stage-3 formed q into legoESM.  Live
   velocities, thicknesses, metric transports, masks, and stencil stay live.
4. Confirms q formation only if the substituted western-boundary surface-V
   signed mean enters the scalar/vector NEMO envelope and the stage-3 HPG and
   advection arrays remain bitwise unchanged.  Otherwise it refutes q as the
   owner boundary and leaves the next discriminator to the transports or
   completed term.

The preregistration is commit `bb7c3bccc` and has SHA-256
`ff503c09dee3dfb03e1e9454b63dc4b6da17f8d2bf5a47378c8ad5cdf86e3ebc`.
It was committed before the writer, scorer, or any round-17 acquisition.

## Implementation and controls

The legoESM change is a private, default-`None` test seam.  It replaces the
already formed q only in the ENE-total operator and raises on an ambiguous
total-vorticity-plus-q override, a wrong shape, or a non-ENE-total dispatch.
The ordinary path is array-identical with the seam absent.  Unit controls show
that an identical q is bitwise inert; a planted q word moves the vorticity
component while the two live transport arrays remain bitwise unchanged.

The NEMO writer is write-only and guarded by `GYRE_R17_WRITE`.  Control and
instrumented runs use the same executable and must produce byte-identical
terminal restarts.  It is single-domain only and refuses `RANKS != 1`.
Admission hashes every artifact, source patch, resolved deck, arithmetic
flavour, producer commit, and common restart.  The reader ignores only the
outer work-array words outside NEMO's executed loop bounds and refuses every
non-finite consumed word.  A planted active q word must make the final verdict
refuse.

The stopped round-17 draft was treated as notes.  Five defects were found and
fixed before this receipt: it tried to put developed step 4,321 into a card
whose registered run control starts at one; verified the ten-step deck after
overwriting its run controls; included `manifest.sha256` in itself; required
uninitialised outer work-array halos to be finite; and exposed an accumulated
boundary subtraction instead of the direct existing vorticity component.
The corrected card retains `nn_it000=1`, `nn_itend=26280`, `nn_GYRE=4`,
`rn_Dt=3600 s`, and mode-32 Smagorinsky; 4,321 is used only by the forcing
clock and NEMO one-step run.

## Acquisition needed

Both no-run preflights pass from the clean committed tool:

```bash
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-646520212 && scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_round17_vorticity/run.sh --preflight-vector
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-646520212 && scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_round17_vorticity/run.sh --preflight-scalar
```

The operator should acquire the two one-step records exactly as follows:

```bash
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-646520212 && EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round17/oracle_gyre_r4_smag_stage3_vorticity_vector RANKS=1 scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_round17_vorticity/run.sh --run-vector
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-646520212 && EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round17/oracle_gyre_r4_smag_stage3_vorticity_scalar RANKS=1 scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_round17_vorticity/run.sh --run-scalar
```

After both print `ACQUISITION_DONE`, score the registered row and its active-q
plant:

```bash
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-646520212 && PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_round17_vorticity.py --vector-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round17/oracle_gyre_r4_smag_stage3_vorticity_vector --scalar-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round17/oracle_gyre_r4_smag_stage3_vorticity_scalar --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round17/stage3_vorticity_discriminator.json
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-646520212 && PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_round17_vorticity.py --vector-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round17/oracle_gyre_r4_smag_stage3_vorticity_vector --scalar-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round17/oracle_gyre_r4_smag_stage3_vorticity_scalar --plant-q --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round17/stage3_vorticity_discriminator_plant.json
```

## Tests and review

Focused seam/reader/writer controls: `21 passed`.  The high-resolution split:
`139 passed, 3 deselected`; the shared GYRE/card/operator split: `149 passed,
2 deselected`.  These include the shipped-card ceiling, both high-resolution
ladders, Smagorinsky, phase-3 stage operators, full-state driver, climate,
movie, divergence, budget, passivity, active-word, manifest, and dispatch
controls.  Both arithmetic-flavour preflights end in `PREFLIGHT_OK`.
The receipt citation gate passes all four compiled-source citations with zero
unmapped, failing, or stale map entries; its shifted-line plant refuses.

Review status is **UNREVIEWED**.  The required read-only Claude reviewer was
invoked but returned no verdict before its stalled session was terminated.
No GLM-5.2 endpoint or binary is available in this environment.  Those are not
waivers.  This round lands an instrument and an acquisition request, not a
scientific owner verdict or physics edit.

## Inherited pins / OPEN

Inherited unchanged: shipped one-degree digest `99b2dfce324b0f77`, the
one-degree no-worsening ceiling, 31 levels, WS-RK3, split-explicit free
surface, TKE/EVD, Redi tracers, the `r4_smag` deck, 3,600-s step, literal
360-day forcing clock, scalar-libm certified CPU ladder policy, vector-math
certified climate reference, the round-16 western-boundary wet-span reduction,
and the four-times twin screen (context only; not the causal verdict bar).

OPEN, in order:

1. Operator acquires both common-state, one-step NEMO arithmetic flavours.
2. Run the registered q substitution and its active-word plant.
3. If q is refuted, substitute the recorded metric transports or completed
   term one boundary at a time.  If q is confirmed, trace its formation
   against the executed statements before considering a shared-path edit.
4. Run both required independent reviews when their services are available.
5. Post the finding to issue #1455 when network access is available.

Choice audit: the user asked for the smallest common-developed-state proving
test, one-variable q substitution, owner localization before any physics
change, affected-card gates, and an exact acquisition request when records are
missing.  The records are missing, so no owner is named and no physics is
changed.  No unasked physical option or long run was added.

## Compiled-source citations

At stage 3 NEMO uses the half-step `Kmm` thickness/velocity state, including
the half-sum `r3f`, at
`GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/stprk3_stg.f90:231-247`.
The executed call order is HPG, vorticity/Coriolis, then advection at
`GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/stprk3_stg.f90:399-410`.
The configured total-vorticity ENE dispatch calls `vor_ene(...,ntot,...)` at
`GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/dynvor.f90:235`.
That routine forms total vorticity, divides by live F thickness, forms metric
transports, and applies the signed two-point V stencil at
`GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/dynvor.f90:536-573`.
