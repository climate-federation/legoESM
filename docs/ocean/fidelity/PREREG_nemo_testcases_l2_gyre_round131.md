# Preregistration — NEMO testcase L2 GYRE round 131

Date: 2026-09-20

Incoming lane tip: `404e9fc626199f61e16e1150ccc08e9501240c1b`

This document is frozen before the committed Round-131 record-admission gate
is implemented or run. Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round131/`.

Round 129 showed that the year gap is systematic rather than member spread,
and Round 130 showed that the held locally exact patches do not carry its
day-240 magnitude. The operator therefore directed four 360-day sensitivity
arms in which one state family is reset to NEMO at every daily boundary. A
daily reset is an oracle-input experiment, not a production configuration or
a landing candidate.

A read-only orientation inspection before this preregistration found an
apparent conflict with the operator's record premise: the path named as the
daily source appears to stop after day 30. That inspection is not promoted to
evidence. The committed, planted admission gate below decides whether any
reset arm may run.

## P0 — fail-closed daily-record admission

The only admitted daily oracle root is
`phase3/year_owners/nemo_seed0`. For a 360-day, six-steps-per-day member it
must contain exactly the 360 completed-step restart boundaries
`kt=6,12,...,2160`, with no missing or duplicate step. Every boundary must
contain the state consumed by the registered families:

* tracers: `tn`, `sn`;
* vectors: `un`, `vn`, `ub_e`, `vb_e`, `ubb_e`, `vbb_e`;
* free surface: `sshn`, `ssha`, `sshb_e`, `sshbb_e`; and
* TKE: `en`, `avm_k`, `avt_k`, `dissl`.

The gate also requires the expected `namelist_cfg` cadence
(`nn_stock=6`, `nn_itend>=2160`), validates the variable schema at every
boundary, and compares every 30-day overlap against the immutable monthly
NEMO year root `phase3/year_fromrest/nemo_seed0`. Each overlapping science
array must be bit-identical. File-name existence alone is not admission.

Frozen prediction: the operator's stated premise is **REFUTED**. The named
root contains 30 daily boundaries (`kt=6..180`, days 1–30), is missing 330
boundaries (`kt=186..2160`, days 31–360), and its namelist terminates at
`nn_itend=180`. The formal gate must stop before a model step. The falsifier
is a complete admitted set of all 360 boundaries with the required schema
and overlap identity.

A synthetic missing-boundary plant and a required-variable plant must each
print `STATUS PLANT-FIRED` and exit nonzero. The real incomplete record must
print a named `STATUS STOPPED-FOR-RECORD` and exit nonzero while still writing
its audit JSON. A control or plant that exits zero is a gate failure.

No monthly state may be silently reused for intervening days, interpolated,
or substituted for a daily boundary. The monthly year record is a comparison
control only.

## P1 — conditional four-family experiment

Only if P0 passes, run four independent 360-day member-0 arms from the clean
incoming tip with the certified GYRE-zco card, fp64/libm policy, 2,160 steps,
six-step snapshots, and tag `year`. Immediately after each completed daily
step and after recording the pre-reset score state, replace exactly one
family for the next step:

1. `tn/sn`;
2. `un/vn` plus `ub_e/vb_e/ubb_e/vbb_e`;
3. `sshn/ssha` plus `sshb_e/sshbb_e`; or
4. `en/avm_k/avt_k/dissl`.

The free control is the landed Round-130 arm at
`phase3/round130/arms/r109_handoff/lego_seed0_year`. Score every model field
at days `30,60,90,120,180,240,300,360` against that free arm and against the
matched NEMO boundary. The owner family is the arm removing the largest
amount of day-240 T3D RMS growth; zero removal exonerates a family. Report the
western-third, upper-100-m birthplace using the Round-122 region/level
partition.

Frozen directional prediction, conditional on admission: resetting vector
state and its carried barotropic histories removes the most day-240 T3D
growth because the amplified mode is the western jet. This is falsified if a
different family has the largest registered removal. No result is inferred
when P0 fails.

The requested three-step cadence is admissible only if the oracle record
contains every `kt=3,6,...,2160` boundary with the same schema and controls.
Daily-only data cannot be interpolated into this arm.

## P2 — outcome and scope

If P0 fails, no reset run starts, no owner is named, production is unchanged,
and the round ends `STOPPED_FOR_RECORD`. The receipt must distinguish absent
daily states from the available 30-day checkpoints and leave an explicit OPEN
item for a complete certified record.

If P0 passes, all four arms run before ranking. This round lands no physics,
changes no card or carried state, and requests no configuration decision. A
separate read-only Codex review must try to refute record completeness,
family isolation, boundary timing, and any ownership claim. Compiled-source
citations and the shifted-citation plant remain mandatory.
