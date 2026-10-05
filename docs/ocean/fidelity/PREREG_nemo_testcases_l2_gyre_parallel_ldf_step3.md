# Preregistration: re-proof of held WS-RK3 stage-3 LDF content routing

Date: 2026-09-16. Frozen at
`3e7a15c1e64e036e2e066dcfceaa663f25406f24`, before any current-tip
measurement or candidate edit. This is a re-proof of the Round-66--70 held
candidate, not a new operator hypothesis and not authority to land it.
Decision 41 keeps all stage-3 work downstream of the autopilot's current kt=1
stage-1 walk.

## Candidate and compiled statement

The candidate removes the private
`route_gm_redi_stage3_source` selector and unconditionally, only for the
already selected `rk3_ws` branch when GM/Redi is configured, adds the existing
signed `dT_gm/dS_gm` arrays to the stage-3 source tuple passed to the single
WS tracer helper. It changes no card, coefficient, timestep, carry, public
API, stabilizer, or other integrator.

The compiled GYRE program zeros tracer `Krhs` and then accumulates advection
and SBC at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:824-868`.
In stage 3 it adds QSR, calls `tra_ldf`, and then calls `tra_zdf` at the same
file's `:917-965`. The selected isoneutral LDF reads Kbb tracer gradients at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:175-204`, forms
Kmm-metric fluxes at `:227-246`, and adds their divergence with a positive
sign to `Krhs` at `:257-305`. The selected ZDF recurrence forms the
Kbb-content plus Kmm-thickness-times-`Krhs` right-hand side at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565`.

At the frozen legoESM tip the existing one-variable arm is defined at
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1359-1362`
and routes the already computed arrays into the stage-3 tuple at
`:7721-7731`. The production candidate is exactly that arm with the private
boolean replaced by the executing `gm_redi is not None` condition.

## Lane 1: current-tip local exactness

Use the existing
`nemo_testcase_l2_gyre_round67_ldf_order.py --round69-native` gate; do not
write a replacement. Its oracle inputs remain
`round64/oracle_krhs_split/oracle_krhs_split_kt00000002.bin` and the
Round-46/kt2-stage record tree. Run the candidate from a clean scratch commit
and require the tool's production-route and model-order expectations to be
`after`.

Prediction: the gate prints `STATUS CONFIRMED`; both production content metric
dictionaries and both post-solve kt3 metric dictionaries are exactly equal to
the frozen Round-69 JIT-native arm; the private selector is absent. In
particular the routed content maxima remain
`5.954039670541533e-5` for T and `7.651457053725608e-6` for S, and the kt3
maxima remain `8.916073106490785e-7` for T and
`7.235656340753849e-8` for S. Any unequal dictionary member, retained private
selector, wrong record stamp, or nonzero gate exit refutes the current-tip
local proof.

## Lane 2: existing Decision-41 stage twin

Run the existing
`nemo_testcase_l2_gyre_round46_kt2_stage_gate.py --mode stage-twin` on a clean
tip baseline and on the clean candidate scratch commit, using the named
Round-40 stage-1 record and all of the gate's current Round-46/75/81/94 record
inputs. Compare every literal output row by name; no second stage harness is
permitted.

Frozen structural prediction: with NEMO's recorded entry, only the kt=1 and
kt=2 stage-3 T and S output rows may move. No stage-entry row, earlier-stage
row, stage-3 U/V/SSH/thickness/W/TKE/transport row, or external row may move.
The route alone is predicted to make **no** stage output row BIT: all four
given-entry stage-3 tracer rows remain non-bit. In the chained table, kt=1
stage-3 T/S are the first rows allowed to move; all kt=1 rows preceding those
must be identical. Downstream kt=2 rows may move and will be enumerated rather
than summarized. A moved forbidden row, a missing row, or a newly BIT row
outside the four named given-entry tracer rows refutes the structural claim.

## Lane 3: full kt=1..10 Rule-12 ladder

Run `nemo_testcase_l2_gyre_phase3_gate.py` on the clean preregistered tip and
again on the clean candidate scratch commit. Compare all 954 registered rows
with the gate's own `--compare-to`/`--comparison-output` path and preserve the
literal moved-row table. A scientific DEBT exit is expected and is not a
harness failure.

Prediction: kt=1 and kt=2 endpoint rows remain bit-identical to the tip; the
first moved rows are kt3-before T and S; at least one kt3--10 row moves; no
AT-BAR row becomes DEBT and first-over-bar remains kt2 U/V. The comparison is
predicted to FAIL Rule 12 because at least one moved cell worsens by more than
two row-scale binary64 ULPs. Any earlier move, AT-BAR loss, different
first-over-bar boundary, or incomplete row census refutes this ladder
prediction. The exact moved-row count and every before/after row value are
measurements, not preregistered constants, because Round 85 landed upstream
momentum changes after the original 53-row Round-69 result.

## Disposition and scope

Even if all three lanes confirm, preserve the candidate only as a held
manifest patch. Do not edit production in the delivered branch and do not
claim a landing. Day 30 is not rerun: the requested experiment is the full
kt=1..10 ladder, and the immutable current-tip day-30 T RMS reference remains
`1.2397011295506804e-2 K` without a candidate measurement. DINO, ORCA2,
LOCK_EXCHANGE, and OVERFLOW receive no new numerical claim. NEMO source,
records, executables, and the concurrent autopilot lane are read-only.

## Addendum after the frozen Round-69 target was refuted

The first clean candidate run at `6e0386627c29b1eab2e284fcee673a5523712df4`
completed after the preregistration above. Artifact
`parallel/ldfstep3/candidate/local_proof.json`, SHA-256
`c51ad1e7a3aed0821997193e3d6a6076ce0747283883f9efbe5cde849846a60f`,
is **REFUTED** against the old Round-69 dictionaries. Its current T/S content
maxima are `5.743498263655056e-5` and `7.387909136014059e-6`; its current T/S
kt3 maxima are `8.600420500215478e-7` and `6.979443156751586e-8`. Thus the
old absolute metrics are not a valid current-tip exactness target after the
Round-85 upstream momentum landing. This failed prediction remains part of
the evidence.

Before running another causal arm, freeze the discriminator that answers the
narrow implementation question without reusing the stale absolute target.
Use the existing Round-70 mode of the same Round-67 LDF-order gate on the
clean tip, with its private `ldf_only` arm and the same Round-64/Round-46
records. Compare that arm to the already measured clean candidate artifact.
For both T and S, require exact equality of the complete
`content_vs_oracle` and `kt3_vs_oracle` metric dictionaries, not rounded
maxima. Also require zero unequal cells in the candidate's live source
injection row. Equality confirms that production is bit-exact to the
current-tip private route given the recorded operands; any dictionary member
different refutes that local implementation proof. The stale Round-69
absolute-target refutation is not reclassified if this same-tip comparison
confirms.
