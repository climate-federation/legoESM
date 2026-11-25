# ORCA2 round 314 — full namelist audit of rung 10 (the shipped ORCA2 card)

Date: 2026-10-18 (Claude round; codex review: operator).

Status: **STOPPED_FOR_DECISION** — read-only audit; no card, setting or
default changed.  Six rows need the user (DECISION_NEEDED below).

## Headline

- NEMO's ORCA2_OMIP_L4 run resolves 1,108 namelist keys (829 ocean, 279
  sea ice).  Basis = the record run the card replays.  Against it: 161
  MATCH, 8 DIFFERS, 32 UNCOMPARED, 690 NOT-MODELLED, 217 INERT.  Against
  EXP00 (second column): 160 MATCH, 9 DIFFERS, same 32/690/217.  Every
  MATCH and DIFFERS row reads both the NEMO value and a card value.
- The 8 DIFFERS vs the record: 2 are earlier decisions (the ice-dependent
  TKE mixing length, Decision 135; ice-ocean top drag, Decision 133).  6
  are unexplained: the isoneutral tracer diffusion (3 rows), the
  eddy-induced velocity, the mixed-layer eddies, and the declared run
  length.  EXP00 adds the ninth, icebergs (on in EXP00, off in the record
  and the card; Decision 7).
- No DIFFERS row is a ladder-certified statement waiting for import.
- Four pieces of interior physics the deck runs are absent from the card
  and from its "unmeasured" list: interior T/S damping, geothermal heating,
  river-mouth mixing, mixed-layer eddies.  Only the eddy-induced velocity is
  declared.

## Review fixes (hold of 2026-10-18; no card changed)

- The 56 initial-T/S and wave-mixing file-setting rows were counted MATCH
  by comparing a fixed label with itself.  Now: interval, interpolation and
  climatology (24 rows) compare NEMO's value against what the card reads
  (the builder's two record indices and weight and the file's 12 records;
  the wave-mixing reader's single record of a one-record file, no time
  interpolation).  All 24 MATCH.  File type, weights, rotation and
  land-fill names (32 rows) are UNCOMPARED: the audit reads no card value.
- Rows were compared against EXP00.  Now the record is the basis and EXP00
  is a labelled second column.  Before fix 1 the record basis gave 221
  MATCH / 8 DIFFERS (the review's number); after it, 189 / 8 / 32.
- Pinned test on the old rule: 2 failed (the plant "did not raise"; counts
  220/9), `fix_old_rule.log`.  New rule: 7 passed, `fix_new_rule.log`.
  Old script restored and byte-compared.
- Fix commit `707c012d7`; table regenerated there on a clean tree
  (`audit_fix.json`, `audit_fix.log`).  Re-read by eye: run length (record
  10, EXP00 5840), the T/S interval and climatology flags, the wave-mixing
  interval.  All agree with the raw namelists.

## Second review fixes (hold of 2026-10-18; no card changed)

- 38 MATCH rows never read the card.  Eight now do: single-column switch
  (the card's grid shape, 148 x 180), the four alternative
  vertical-coordinate switches (the builder builds z-star), restart start and initial T/S
  (the files the builder opens: no restart, both climatology files), the
  runoff depth file (none on the card), the eddy-energy equation (none on
  the card), mixed vorticity scheme (the card runs energy-enstrophy).  The
  other 28 (switches NEMO has off with no card value to read) are
  NOT-MODELLED; NEMO's value is still asserted off.  MATCH 189 -> 161
  (record) and 188 -> 160 (EXP00),
  NOT-MODELLED 662 -> 690 on both.
- TKE minimum energy now expects 1e-10 only when NEMO's resolved
  wave-mixing switch is on, else the namelist value
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:835-836`).  The record has it
  on, so the row still MATCHes (card 1e-10, namelist 1e-6).
- Pinned test on the old rule: 3 failed (counts, the off-switch plant, the
  wave-mixing plant), `fix2_old_rule.log`.  New rule: 9 passed,
  `fix2_new_rule.log`.  Old script restored and byte-compared.
- Fix commits `15c8c1a94` (rules, test) and `909f1759b` (run-length text);
  table regenerated at `69e2027fa` on a clean tree (`audit_fix2.json`,
  `audit_fix2.log`).

## Provenance

Base `78d61ed80` (round 313).  Script `4eb36a759` (+ header fix
`b69e76896`); table generated at `b69e76896` on a clean tree.  Script:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round314_namelist_audit.py`;
table: `docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_round314_namelist_audit_table.md`
(one row per key: NEMO value, record value, ocean.output check, card value,
status, class, reason).  Evidence `orca2_rounds/round314/` (`audit.json`,
`audit.log`).  No preregistration: an audit makes no prediction to falsify.

- NEMO's value (the basis) = the record run's own resolved dump; the
  record is rung 10's record, round 5's
  `orca1ice_surface_entry_every_step_a_np2` (the record every rung-10
  ladder replays).
- EXP00 (second column) = `cfgs/SHARED/namelist_ref` overlaid by
  `cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg`, and the same for the two SI3
  files.
- Card value = the built card (`build_orca2_zps_card` on that record's
  deck root): its resolved config, its carried mixing seeds, its masks; for
  file names, the files the builder opens; for one TKE constant, the
  scheme's module value.  Never a comment.

## Instrument checks

- Parse vs NEMO's own resolved dump of that run (`output.namelist.dyn`,
  `output.namelist.ice`): 1,078 keys equal; 17 equal once explained (16
  partially-set arrays, 1 string NEMO truncates to its declared length);
  13 not in any namelist file (NEMO's built-in initial value, taken from the
  dump); 0 unexplained.  The first run failed this check (continuation lines
  of long arrays), which is how the parser bug was found and fixed.
- ocean.output cross-check: 422 agree, 221 not printed, 462 not applicable
  (structure fields, arrays), 3 differ.  All 3 are NEMO changing the value
  at run time: the domain decomposition (2 x 1, namelist 0 x 0 = automatic),
  and the barotropic substep count (65, namelist 30, recomputed because the
  automatic option is on).
- Fail-closed: any unclassified key, any INERT row whose switch does not
  resolve as claimed, any DIFFERS row without a class, or any unexplained
  parse disagreement exits 1.  The direct test pins the counts on both
  bases, the 8 DIFFERS rows vs the record and the 9 vs EXP00, and that no
  MATCH row lacks a card value; a planted rule change turns it red;
  flipping one card setting in the built card moves exactly that row.
- Spot-checked by eye against the raw files (namelist_cfg, namelist_ref,
  the record's ocean.output): the isoneutral coefficient shape (20,
  `ORCA2_OMIP_L4/EXP00/namelist_cfg:304`), its velocity (0.018, `:306`), the
  mixed-layer eddies (on, `:312`), the stabilising correction (on, `:302`),
  the eddy-induced velocity (on, `:317`), the run length (EXP00 5840,
  `:21`; record 10 from its dump), the ice-dependent mixing length (2,
  `:415`), the implicit ice-ocean drag (on, from `namelist_ref:818`), the
  ice category count (5 in EXP00, 1 in the record).  All agree with the
  table in both columns.

## The record is not EXP00 in 50 rows

- Run control (3): length 10 steps, initial-state output, restart at 10.
- Icebergs off (1, Decision 7).
- Sea ice (46): the record runs a one-category ice deck (1 category, 3 ice
  and 3 snow layers, salinity option 2, no ponds, other initial ice) where
  EXP00 runs 5 categories, 10/5 layers, salinity option 4, ponds.  This is
  Decision 80's rung 10 and Decision 50's scope.  The card replays this
  record's ice fluxes, so rung 10's oracle is the one-category ice, not
  EXP00's.

## DIFFERS (8 vs record; 9 vs EXP00), by what they change

| what it changes | setting | record | EXP00 | card | class |
|---|---|---|---|---|---|
| isoneutral tracer mixing | stabilising split of the vertical term | on | on | off (fully implicit) | (c) |
| isoneutral tracer mixing | diffusivity shape | half the diffusive velocity times local grid spacing | same | constant | (c) |
| isoneutral tracer mixing | diffusive velocity | 0.018 m/s | 0.018 m/s | none (1000 m2/s everywhere) | (c) |
| eddy parameterisations | eddy-induced velocity | on | on | off (declared unmeasured) | (c) |
| eddy parameterisations | mixed-layer eddies | on | on | off (not declared) | (c) |
| vertical mixing | surface mixing length under ice | scaled by ice thickness | same | not scaled | (b) Decision 135 |
| drag | implicit ice-ocean top drag | on | on | off | (b) Decision 133 |
| run length | declared steps (3 h each, 365-day calendar) | 10 (30 h) | 5840 (two years) | 2920 (one year) | (c) |
| icebergs | icebergs | off (MATCH) | on | off | (b) Decision 7, EXP00 only |

Round 320 update: the three isoneutral rows now MATCH (lane B L2 on rung 10;
record 164 MATCH / 5 DIFFERS, EXP00 163 / 6). Geothermal stays NOT-MODELLED
(lane A L4 imported, refuses with tracer lateral diffusion).

- NEMO's isoneutral diffusivity on ORCA2 is about 2000 m2/s near the
  equator and about 850 m2/s poleward of 60 degrees (arithmetic from the
  domain file's grid spacings, not a model run); the card uses 1000 m2/s.
- No row is (a): the vertical-mixing ladder's certified statements (V1-V5)
  are already on the card; the geometry ladder's pending FCT fold statements
  are not namelist settings.

## NOT-MODELLED (690), by what they would change

- Interior T/S damping, 4 rows (`ORCA2_OMIP_L4/EXP00/namelist_cfg:328`,
  `:48`): NEMO restores T and S toward climatology where the damping map is
  set; the card applies no damping.  Not declared.
- Geothermal heating, 12 rows (`ORCA2_OMIP_L4/EXP00/namelist_cfg:256`):
  NEMO heats the bottom from a map; the card has no bottom heat source.
  Not declared.
- River-mouth mixing, 12 rows (`ORCA2_OMIP_L4/EXP00/namelist_cfg:161`):
  NEMO adds 1e-3 m2/s of vertical diffusion over the top 15 m at river
  mouths; the card does not.  Not declared.
- Eddy-scheme parameters, 11 rows: follow the two eddy DIFFERS rows above.
- Icebergs, 35 rows: follow the iceberg row (Decision 7).
- Surface fluxes, 215 rows (bulk formulae, surface restoring, freshwater
  budget, forcing files, coupling frequency, runoff input): the card is
  driven by the record's recorded surface fluxes and computes none of them.
- Sea ice, 279 rows: the card runs no ice model and replays the record's
  ice-affected fluxes (Decision 50).
- Chlorophyll input, 11 rows: the replay reads the same chlorophyll file
  each step; the card's own shortwave settings all MATCH.
- Barotropic substep limit, 1 row: the card fixes NEMO's resolved 65
  substeps instead of deriving it.
- Calendar, 3 rows: no calendar; the replay supplies each step's forcing.
- Switches NEMO has off with no card value read, 28 rows (closed seas,
  wetting and drying, Shuman filter, lateral-wall vorticity, nine surface-forcing
  switches, iceberg runoff, open boundaries, three extra mixing
  schemes, wave-height mixing length, drag boost, vorticity masking, ice
  shelves, six stochastic switches).
- No effect on model state, 79 rows: output, decomposition, diagnostics,
  run control.

## Configuration choices

None made: no card, setting or default changed (ASKED: none needed;
UNASKED: none).  Two classifications are mine and stated in the table:
a switch NEMO has off counts as MATCH only where the audit reads a card
value showing the feature absent, otherwise NOT-MODELLED; and parameters of a feature the card turns off count as NOT-MODELLED,
following their switch row.  Added by the review fixes: the 32 file-setting rows with no card value
read are UNCOMPARED, not MATCH; and a one-record wave-mixing file held for
the whole run is read as the card's climatology flag (T).

## DECISION_NEEDED (batched; nothing changed)

1. Isoneutral stabilising split: off -> on (NEMO's split of the vertical
   isoneutral term); changes T/S mixing along sloping density surfaces.
   Pick: on, after a new isoneutral ladder rung certifies it.
2. Isoneutral diffusivity shape: constant -> half the diffusive velocity
   times local grid spacing; about doubles it near the equator, lowers it
   poleward.  Pick: on, same rung as 1.
3. Isoneutral diffusive velocity: none (1000 m2/s) -> 0.018 m/s; the value
   choice 2 uses.  Pick: with 2.
4. Eddy-induced velocity: off -> on (growth-rate-scaled, 3000 m2/s
   reference); adds eddy advection of T and S.  Pick: on, after its own
   ladder rung (already declared unmeasured).
5. Mixed-layer eddies: off -> on; restratifies the mixed layer.  Pick: on,
   after a rung; and add it to the card's unmeasured list meanwhile.
6. Declared run length.  Time step 10800 s (3 h): the record runs 10
   steps (30 h), EXP00 runs 5840 steps (two years), the card declares 2920
   (one year).  Changes no measured number (the ladders replay 10 steps).
   No recommendation: the user decides (keep 2920, or 10, or 5840).

## Gates and tests

JAX_ENABLE_X64=1, CPU, PYTHONPATH as in `land.sh`; evidence
`orca2_rounds/round314/`.

- Receipt citation gate on this receipt: 12 citations, failures 0, status
  PASS (`citation_gate_receipt.json`).
- Default gate list plus the new test, run in four foreground pieces (one
  ten-minute tool limit; the first attempt ran in the background and died
  without a summary line, so it does not count): 133 passed
  (`gate_default_a.log`, includes the private-import ratchet, 6 passed),
  1 passed and 1 passed (`gate_default_b1.log`, `gate_default_b2.log`, the
  two slow face-mask tests), 16 passed (`gate_default_b3.log`, includes the
  six round-314 tests).  151 passed, 0 failed.
- New test non-vacuous: rewriting the mixed-layer-eddy rule turns the
  pinned-table test red (restored, byte-compared).
- After the review fixes (record basis, UNCOMPARED rows), commit
  at `c208304ef`, same pieces: citation gate on this receipt 12
  citations, failures 0, PASS (`citation_gate_receipt_fix.json`); 135
  passed (`gate_fix_a.log`, includes the private-import ratchet, also run
  alone: 6 passed, `gate_fix_ratchet.log`); 2 passed (`gate_fix_b1.log`,
  face-mask); 15 passed (`gate_fix_b2.log`, barotropic state + the seven
  round-314 tests).  152 passed, 0 failed.
- After the second review fixes, at `d82c6f46d`, same pieces: citation
  gate on this receipt 13 citations, failures 0, PASS
  (`citation_gate_receipt_fix2.json`); 135 passed (`gate_fix2_a.log`,
  includes the private-import ratchet, also run alone: 6 passed,
  `gate_fix2_ratchet.log`); 2 passed (`gate_fix2_b1.log`, face-mask); 17
  passed (`gate_fix2_b2.log`, barotropic state + the nine round-314
  tests).  154 passed, 0 failed.
- Second resume: codex review: operator.
- Review: codex review: paused (quota).  Independent Claude review
  (operator): CONFIRMED the DIFFERS rows, absences and direct test; HELD on
  the two instrument defects fixed above.

## OPEN

1. The DECISION_NEEDED batch above.
2. Damping, geothermal heating and river-mouth mixing are undeclared
   absences on the card; whether to declare or build them is the same kind
   of question as 4-5.
3. TOP/PISCES namelists not audited: ORCA2_OMIP_L4 does not compile passive
   tracers, so they cannot reach the ocean.
