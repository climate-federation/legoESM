# NEMO testcase L2 GYRE round 151 — amplification threshold

Date: 2026-09-22

Status: **HELD**.  No production physics or configuration changed.  The
preregistered threshold-amplification prediction is **REFUTED**: at day 240,
the `1e-4 K` initial-temperature arm differs from the unperturbed run by only
`2.369745168564984e-5 K` T3D RMS, `0.0048028732860537935` of the
`4.9340155932201335e-3 K` threshold.  The exact registered verdict is
`INCONCLUSIVE_UNDER_REGISTERED_DISCRIMINATOR`, because this is above the
fallback's `1e-6 K` ceiling while remaining 208.21 times below the campaign
threshold.

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round151.md` at
`e89d774a1`.  Member evidence was produced clean at `f65614f06`; the final
score was emitted clean at `5a4de7330`.  All evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round151/`.

## Fixed perturbation and admission

The compiled Round-129 source guards on nonzero seed, adds
`1e-10 * sin(NINT(pdept)*73 + NINT(gphit*1000)*179 + seed*997) * ptmask`
to temperature, and prints the executed seed at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_istate.f90:101-105`.
Round 151 reused seed 1, the same Fortran-`NINT` transcription, depths,
latitudes, wet mask, temperature field, card, forcing, timestep, precision,
and certified year stepper.  The user-ordered absolute amplitude was the only
changed input.

Each arm changed all 18,000 wet initial-temperature cells and zero S, u, v, or
SSH cells.  The applied fp64 peaks after addition to the base temperature were
`1.000000082740371e-8`, `9.999999805998527e-7`, and
`9.999999808485427e-5 K`.  All three manifests name clean commit
`f65614f063a2e69105ca6b9c346f2d8beeeed876`, fp64/libm, CPU, 2,160 steps,
six-step daily snapshots, and the same certified stepping-gate SHA-256
`e57fe1c475a1d386f30856f1841a2efb65162df6968b74b1a200f8850bdd9112`.
Each member has exactly 360 finite daily snapshots.

The `1e-8`, `1e-6`, and `1e-4 K` runs took respectively `2009.16`, `1926.38`,
and `2000.55 s` wall time while sharing the CPU.  The unperturbed control is
the Round-149 after arm preserved bit-for-bit by Round 150.  Its independently
read day-240 gap to the admitted 360-day NEMO seed-0 record reproduces
`1.644671864406711e-2 K` exactly as binary64.

## Registered eight-day table

Every value is unweighted fp64 T3D RMS over NEMO's 18,000-cell wet `tmask`,
candidate minus legoESM member 0.  No NEMO difference enters these rows.

| day | `1e-8 K` arm | `1e-6 K` arm | `1e-4 K` arm |
|---:|---:|---:|---:|
| 30 | `2.3114479615982225e-9` | `1.469692733697682e-7` | `1.3893697001452605e-5` |
| 60 | `2.1672222953849213e-9` | `1.2833663026770813e-7` | `3.432647130972633e-5` |
| 90 | `2.1216241467098796e-9` | `1.252539792487413e-7` | `4.84648449996782e-5` |
| 120 | `2.3641907672914486e-9` | `1.5288947409904106e-7` | `2.6040031795491694e-4` |
| 180 | `2.117351591146339e-9` | `1.2842214168313392e-7` | `4.234941412903075e-5` |
| 240 | `2.0855080249137747e-9` | `1.2602164982834932e-7` | `2.369745168564984e-5` |
| 300 | `2.0415969098458075e-9` | `1.2424872275709545e-7` | `1.8615770743499382e-5` |
| 360 | `2.2507716091146918e-9` | `1.2417890247790187e-7` | `2.3899491270730377e-5` |

The largest observed response is the `1e-4 K` arm's day-120 spike,
`2.6040031795491694e-4 K`.  It is 2.604 times the imposed amplitude but still
18.95 times below the campaign threshold, and it collapses by day 180 rather
than growing toward the day-240 gap.  At day 240, response divided by imposed
amplitude is `0.20855`, `0.12602`, and `0.23697`; there is no measured
transition to gap-scale amplification in the tested range.

The `1e-8 K` day-240 row is `9.98266576755432` times Round 129's
`2.0891293703252062e-10 K` four-member sensitivity floor.  Thus the frozen
statement that it would remain near the `1e-10 K` class is directionally
reasonable to one order of magnitude but not bit-level equality; it is not
used to rescue the failed large-arm prediction.

## Prediction ledger and retractions

1. **REFUTED:** `1e-4 K` did not cross `4.9340155932201335e-3 K` at day 240;
   it reached `2.369745168564984e-5 K`.
2. **CONFIRMED:** `1e-8` and `1e-6 K` remained below the amplification bar.
3. **REFUTED:** the registered threshold did not lie between `1e-6` and
   `1e-4 K`.
4. **INCONCLUSIVE by the frozen alternative:** not all three day-240 rows are
   below `1e-6 K`, so this round does not mechanically promote
   `SYSTEMATIC_TRACER_STEP_OWNER`, even though no tested arm approaches the
   campaign gap.
5. **CONFIRMED:** no production physics, card, state schema, configuration,
   stabilizer, forcing, timestep, or year harness changed.

The first scoring attempt is explicitly retracted as an instrument result.  It
pointed the control reproduction at the 30-day-only `year_owners` directory,
refused the absent day-240 restart, and emitted no scientific table.  The
corrected gate uses the admitted 360-day `year_fromrest` ensemble and refuses
scoring from a dirty worktree.  The pre-correction JSON is preserved as
`amplification_pre_exact_initial.json`; its trajectory rows equal the final
ones, but it is not authoritative because its displayed initial delta was
formed by post-hoc rescaling.  The authoritative score replays the exact
amplitude multiplication used by member mode.

## Controls, scope, and review

The exact binary64 threshold self-check passes equality, rejects the adjacent
value below, and accepts the adjacent value above.  The amplitude plant moves
the admitted `1e-8 K` manifest value by one binary64 step; it prints
`STATUS PLANT-FIRED: member-amplitude` and exits 1.  The day-registry plant
removes day 240; it prints `STATUS PLANT-FIRED: day-registry` and exits 1.

No production statement changed, so GYRE, generic NEMO-GYRE, DINO,
LOCK_EXCHANGE, OVERFLOW, and ORCA2 have no trajectory blast radius this round.
ORCA2 is **UNMEASURED-WITH-SPEC**: run its ocean-only member 0 plus the same
three seed-1 amplitudes and score its native wet T3D population before
transferring this result.

The required read-only Codex review was attempted after measurement.  Its
verbatim terminal result was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
codex_exit=1
```

Independent review was unavailable in-sandbox.  It emitted neither `SHIP` nor
`DO NOT SHIP`; no approval is fabricated.

The authoritative `amplification.json` SHA-256 is
`52c14162e61c484885659c11ccbb9137e3219fe5d2c6d353cecc5d5202b2a628`.
The three member-manifest SHA-256 values, in increasing-amplitude order, are
`9d1dceb02f9f9f4f23374261aaffbbbb14a72e810dec11dc109809ae8976ef57`,
`9269dbfaa2aa6553ff4ccfc5724b4ae7c6179726ccb4aed0855677808a5e040e`, and
`65ec36d5f68b3120d20ad1f6fa71a012cb70d7f66d53db7fe1df2cc589c3ca21`.

## Verification

The receipt and preregistration citation gates each report PASS with one
compiled-source citation, zero unmapped citations, zero failures, and an empty
map audit.  Shifting the compiled perturbation citation by two lines reports
FAIL and exits 1.

The focused four-file suite reports exactly `1 failed, 58 passed in 22.92s`.
The sole failure is the known pre-existing worktree-stamp ratchet, naming only
the Round-146 RHS-family gate and Round-50 LDF-association gate.  The new
Round-151 emitter is absent from its offender set; all 58 remaining focused
tests pass.  No full ocean tree was run because this round changed no
production path and its three completed 360-day integrations are the requested
trajectory test.

## OPEN — round 152

Do not redefine the year target from this measurement: the largest day-240
response is 208.21 times too small, while the preregistered systematic fallback
also did not fire.  Return to the operator's developed-state fallback without
claiming this intermediate result proves its owner: from NEMO's day-180 entry,
complete the production-JIT tracer step through advection, tracer LDF, ZDF, and
content assembly, rather than stopping at the first attribution circle.  Rank
the first non-bit active-branch statement by its one-step T tendency magnitude,
then year-score only a source-exact candidate.  No new amplitude, configuration,
or acquisition is authorized by this receipt.
