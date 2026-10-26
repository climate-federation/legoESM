# Preregistration: held-forcing downstream split-explicit chain, round 6

Date: 2026-08-29. Frozen before the round-6 CPU/offline measurement.

## Question and inherited ownership

Rounds 4--5 disposed both components of row 1.1 with complete oracle-substitution
matrices.  U closes with the three-axis lateral/vorticity/pre-loop-Coriolis arm;
V closes only after adding pressure gradient.  This round asks whether the
downstream split-explicit recurrence closes when its already-owned slow forcing
operand is held to the existing NEMO dump.  It changes no production code and
does not authorize a composite physics change.

The arm injects existing `spg_dump_zu_frc.bin` and
`spg_dump_zv_frc.bin` at the production `barotropic_substeps_latlon_cgrid`
entry.  DINO's periodic U face is reconstructed as west face = the last NEMO
column and remaining faces = NEMO columns; V uses a zero south wall followed by
the NEMO north faces.  These are the committed bridge conventions in
`nemo_state_bridge.py:_u_east_to_face_periodic` and `_v_north_to_face`.

## Ordered measurements and bars

The live oracle initializes the centred loop from BEFORE at
`cfgs/DINO/MY_SRC/dynspg_ts.F90:561-580`, initializes accumulators at
`:600-605`, and enters the substep loop at `:614-620`.  The inherited existing
dumps and registered populations remain unchanged.

| Subrow | Literal BEFORE measurement | Held counterfactual | Bar and disposition |
|---:|---|---|---|
| 1.1 | production slow forcing, retained from rounds 4--5 | exact NEMO `zu_frc/zv_frc` face reconstruction | held receipt must be bit-exact on 9,758 U / 9,868 V wet faces |
| 1.2 | `sshn_e/un_e/vn_e` loop seed | additionally hold exact NEMO seed only for the targeting continuation | POINTWISE `1e-15`; literal score remains authoritative; held arm cannot promote it |
| 1.3 | first-substep `ssh/ub/vb` under held row 1.1 | row 1.1 + exact row 1.2 seed | ACCUMULATING `1e-12` plus aggregate campaign axes |
| 1.4 | final `puu_b/pvv_b/pssh/un_adv/vn_adv` under held row 1.1 | row 1.1 + exact row 1.2 seed | ACCUMULATING `1e-12` plus aggregate campaign axes |

The literal row-1.2 score is preregistered to remain an ordered stop if any
component is not `AT BAR`; the exact-seed arm is explicitly
`TARGETING-ONLY-BEHIND-ROW-1.2`.  For rows 1.3--1.4, the primary causal
counterfactual is the exact-forcing-only arm; exact seed is a secondary
targeting discriminator.  `MATCHED` requires every registered component to
pass `fidelity_bar_gate.classify`.

## Downstream rows 2--6

If literal row 1.2 and held-forcing rows 1.3--1.4 all pass, rows 2--6 may be
promoted in execution order using their existing dumps.  Otherwise they remain
`ORDERED-BLOCKED`; already committed probes may be run offline and reported
only as `TARGETING-ONLY`, never as chain verification across the stop.

| Row | Existing evidence | Source |
|---:|---|---|
| 2 | `seq_dump_hdiv_nnn_kt00005761.bin` | `stpmlf.F90:349-376`; `src/OCE/DYN/divhor.F90:151-197` |
| 3 | `seq_dump_r3{t,u,v}_aaa_kt00005761.bin`, `seq_dump_r3f_kt00005761.bin` | `stpmlf.F90:378-394`; `src/OCE/DOM/domqco.F90:140-186` |
| 4 | `stp_dump_08_dynzdf_kt00005761_{u,v}.bin` and the completed ZDF receipt | `stpmlf.F90:396-409`; `dynzdf.F90:137-141,166-178` |
| 5 | `wzv_dump_ww_call2.bin`, with call 1 as ordering control | `stpmlf.F90:411-412`; `sshwzv.F90:168-348` |
| 6 | `baro_dump_{u,v}_{before,after}_kt00005761.bin` | `stpmlf.F90:578,709-765` |

## Admission and controls

The wrapper must run from a clean committed worktree on CPU/fp64 with lane
`d180` and `LEGOESM_NEMO_E3T=both`; it must prove that the production entry was
intercepted and restored, record original and held forcing hashes, obtain an
exact wet-face held receipt, preserve exact populations and finite values, and
exercise the inherited planted scorer and zero-shift controls.  It hashes the
wrapper, inherited probe, production modules, oracle sources, dumps, restart,
mesh, namelists, and executable.  A dirty tree, missing call, wrong shape/time
level, failed control, or nonzero inherited exit is `INVALID`.  This uses no
new NEMO writer, so no SLOT block is created.

## Instrument amendment after invalid attempt 1

Attempt 1 exited during state construction, before a production step or any
score, because the inherited probe directly accessed two optional DINO-card
attributes absent from the clean branch commit.  The probe now uses fail-safe
`getattr(..., None)` checks for those optional literal-evaluation selectors.
This changes neither a model setting nor any registered arm, array, population,
bar, or stop rule; attempt 1 is `INVALID` and supplies no evidence.

Attempt 2 then reached the same state-construction call and exited before a
step because the clean commit's bridge predates the optional
`carry_native_lat_deg` keyword.  The probe now passes that keyword only when
the installed bridge signature exposes it.  On this clean commit neither
literal selector exists, so omission is the exact legacy behavior.  Attempt 2
is also `INVALID`; all arms, scores, and gates remain frozen.

## Provenance retraction after adversarial review

The first nominally accepted round-6 receipt is **WITHDRAWN**.  Its detached
measurement checkout was clean, but the editable Python install resolved
production modules from another worktree; the stamped production hash did not
match the claimed commit.  The instrument now fails unless every imported
production module resolves beneath the measured checkout, stamps those paths,
and asserts restoration of every monkeypatch.  The rerun must put the measured
checkout's package roots first on `PYTHONPATH`.  No prior round-6 score is
admissible.
