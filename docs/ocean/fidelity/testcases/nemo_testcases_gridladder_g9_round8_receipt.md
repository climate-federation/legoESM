# Grid ladder rung G9 — round 8 receipt (gridladder_r52 continuation, Claude)

Status: **LANDED; G9 CERTIFIED.** One cited statement: the stage-2/3 tracer
and momentum transport now reads NEMO's stored stage face ratios r3u/r3v(Kmm).
With it, every G9 row is bit-exact in both labels, kt 1..10.

- codex review: operator (this continuation is Claude-authored)
- independent review: operator (Claude)
- evidence: `gridladder_rounds/round52/resume2/`
- NEMO: not built or run in this session

## 1. Codex's unfinished diff

Codex left three uncommitted files when it hit its capacity limit.  The diff
held a stage-2 boundary hook that only writes diagnostics, a reformat of the
two squeezed lines, and a test that printed values and asserted nothing.  It
contained no NEMO-cited fix.  It was stashed unreviewed as
`stash@{0}` ("codex r52 unfinished stage-2 diagnostic (not a cited fix)").

## 2. Round-52 review items

(b) Commit `6522b68d8` had squeezed two call sites into 306- and
189-character lines to keep receipt line numbers fixed.  Commit `ceb4c591a`
restores normal formatting with no semantic change.  The existing diff-based
re-anchor tool then moved 78 citation lines.

(a) At full depth, kt 1 had one T cell one ULP off NEMO's `trazdf` solution:
y 59, x 138, level 12.  A passive debug-callback probe left the whole T field
bit-identical.  It localised the error upstream of the content recurrence.
Kbb T and both thicknesses were exact, but the stage-3 FCT advection rate was
7.240387917899515e-08 against NEMO's 7.240387917899567e-08.

That rate differed in 98 T / 95 S cells, all in 11 columns.  The cause was
six stage-3 faces (4 U, 2 V) where the card's stage-3 transport differed from
NEMO's `oracle_tracer_transport_kt00000001_s3` record.  The card's stage-1/2/3
velocities were exact, so the transport formation owned the difference.

## 3. Owner and fix (also the kt-4 owner)

At every stage, NEMO forms the transport at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:282-283`.
The face thickness uses `r3u(Kmm)`.  The barotropic correction is formed at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:276-277`
and divides by `1+r3u(Kmm)`.

At stage 2, `r3u/r3v(Kmm)` is the stored blend from
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:177-179`.
At stage 3, it is the stored blend from
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:221-223`.
Here `n_baro_upd = np_HYB`.  The card rebuilt the ratio from the interpolated
sea level, which is real-equal but differs in the last place on a few faces.

**First owning statement of the kt-4 over-bar row:** the stage-2 transport
at `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:282-283`,
read with the stored `r3u/r3v(Kmm)` blend of lines 177-179.
This was measured against NEMO's admitted kt-2 stage-2 record, using its
recorded entry and external handoff:

| kt-2 stage-2 transport | stored ratios OFF | stored ratios ON |
|---|---:|---:|
| zFu cells unequal | 88 (7 faces) | 0 |
| zFv interior cells unequal | 35 (3 faces) | 0 |

There are 1748 further zFv differences in the cyclic seam column (x = 0).
They are identical with the fix on or off, so they come from how that column
is stored for the comparison.  This replay's stage-update frame is NOT a
validated instrument: it differs in about 5270 cells both with and without
the fix, while the full-chain rescore below is exact.  No claim rests on it.

The fix adds a new default-off field, `rk3_stage_transport_stored_r3uv`,
selected only by the G9 card.  It hands the existing stored-ratio pair helper
(already used for G1+ momentum faces) to the stage-2/3 transport faces.
The library default and every other card are unchanged.  A caller grep shows
the field is set only in the G9 card builder.

## 4. Tests, graphs, rescore

The kt-1 value test is jitted, with state and forcing traced, in x64.  It is
now **full depth**: G9's T and S are bit-exact on every wet level against
NEMO's `trazdf` solution.  Removing only this round's wiring restores exactly
the one T cell (y 59, x 138, level 12) and no S cell.  Removing only round 7's
wiring restores exactly 105 T / 96 S level-1 cells.  Test log: 1 passed.

A base (HEAD before the fix) versus current address-masked step-jaxpr census
gave 26 of 26 non-G9 graphs byte-identical, with G9 alone different.
`jaxpr_summary.log` SHA-256:
`755926ccfcf42327a1b4d95b635dab0d68cd70da39cd6a29c36f33043ca5ebe5`.

The rescore used round 7's tool, roots and bar, at commit `bf3fae38c`.  It
covers ssh, `uu_b`, `vv_b`, u, v, T and S, plus all-level rows at restarts:

| label | kt | exact / AT-BAR / DEBT |
|---|---:|---:|
| independent | 1..10 | 7 / 0 / 0 each (all levels at restarts: 0 unequal) |
| given entry | 1, 3, 5, 7, 9 | 7 / 0 / 0 each |
| given-entry two-step | 2, 4, 6, 8, 10 | 7 / 0 / 0 each |

Before this fix, round 7 had DEBT at kt 4 (ssh, 1.21e-15).  That run used the
same tool and roots, so this round changes one variable.  The rescore hashes
(SHA-256) are:

- `g9_independent_kt1_4.json`: `365881d896b98652be943f80c3939c50932a0920013977e1c5fe6f8e91f36bc0`
- `g9_independent_kt5_10.json`: `bf6d2b71a013c232c536024ebab6a4d72580fc34e0ed2b8fc846a1b536cad3ea`
- `g9_given_entry_kt1_10.json`: `5377a1a9914ba47cda2d06131f34d57c6ec7d5fc52df52bd276f9e10ef51052a`

## 5. Certification and the ORCA2 fold import

**G9 CERTIFIED**: every row in both labels is exact for kt 1..10.  Under
Decision 122, this certifies G9's selected statements for import onto the
ORCA2 card.  That covers:

- the stage-transport raw V mask and fold exchanges;
- the FCT fold exchange;
- the end-of-stage pivot fold;
- the stage clock pair;
- NEMO's stage-3 Krhs order;
- the stored Kmm r3t in the tracer continuity solve;
- this round's stored r3u/r3v in the stage transport.

The same compiled lines run in the ORCA2 build:
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:177-179`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221-223`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:276-277`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:282-283` and
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:284-290`.

The ORCA2 card itself is unchanged here (its graph is byte-identical).
Landing the import there remains the ORCA2 lane's step.  That lane must still
show no exact row lost and an unchanged or later first over-bar row.  This
also meets the operator's ladder-gap note: the stage-1 3-D V transport on the
T-pivot row is now exercised and certified on a rung (G9).

## 6. Gates and disposition

The receipt citation gate on this receipt found 9 citations, with 0 failures,
0 unmapped citations and 0 failing map entries; its unit suite passed 17 / 17.
The default gate and the private-import ratchet ran file by file in CPU x64
with the prescribed PYTHONPATH.  The counts were 17 + 6 + 68 + 25 + 16 + 2 + 8
= 142 passed, 0 failed.  The changed G9 and G2 census tests passed 3 / 3.
Combined gate-log SHA-256: `688e1af8ccd0f59d6cdd80f3b3b032fb3e6354268ebbe6736a8b335d371b3899`.

ASKED: review and dispose of Codex's diff; extend the kt-1 test to full depth
and explain or fix the cell; restore normal formatting and re-anchor; name
and fix the kt-4 owner G9-only; rescore both labels kt 1..10; state the
ORCA2 import; run the gates; commit.

UNASKED, offered for revert: none.  The new field and its G9 selection are
the asked fix.

UNVERIFIED: the kt-2 stage-2 stage-update replay instrument, which is
unvalidated as described in section 3 and is not cited for any claim.
