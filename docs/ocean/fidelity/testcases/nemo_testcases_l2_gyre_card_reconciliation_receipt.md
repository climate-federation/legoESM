# GYRE, THE TWO RECEIPTS THAT CONTRADICT EACH OTHER: they report THE SAME FIVE NUMBERS, at two different step LABELS — and the card really did resolve into two programs

Gate: `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_card_reconciliation_gate.py`
(five modes: `--config-diff`, `--two-path`, `--oracle-floor`, `--score-both-roots`,
`--closure-ablation`; `--self-check` runs five plants, each of which must be refused).
Tests: `tests/ocean/fidelity/test_nemo_testcase_l2_gyre_card_reconciliation.py`.
Artifacts: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/card_reconciliation/`,
SHA-256 beside them in `card_reconciliation_artifacts.sha256`.
CPU, fp64, `PrecisionPolicy.fp64(transcendentals="libm")`, `JAX_ENABLE_X64=1`.

## VERDICT

**There is no contradiction between the two receipts. They report the same
five numbers about the same step of the same configuration, under two
different names for that step**, and the eleven orders between them is the one
step that separates the ladder's `kt2` row from its `kt3` row.

| quantity, max abs vs NEMO | ladder's `kt3` row (this round, `ladder_sweep_before.json`) | year-owners' "step 2 output" (`year_owners/equal_input_kt2.json`) |
|---|---:|---:|
| `T` | `8.741316e-03` K | `8.7413e-03` K |
| `S` | `1.356889e-03` g/kg | `1.3569e-03` g/kg |
| `u` | `7.193412e-04` m/s | `7.1934e-04` m/s |
| `v` | `8.606873e-04` m/s | `8.6069e-04` m/s |
| `ssh` | `7.072559e-07` m | `7.0726e-07` m |

Five fields, every printed digit. The ladder's row named `kt3` scores the
state **entering** step 3, which is exactly step 2's OUTPUT — the quantity the
year-owners receipt measures. The `7.6e-16` the reconciliation task quotes is
the ladder's `kt2` row, which scores the state **entering** step 2, i.e. step
2's INPUT. The two receipts were never measuring the same thing.

**So the year-owners receipt's headline stands unchanged and is now
INDEPENDENTLY REPRODUCED on the ladder's own program: the difference is born
in step 2.** What is RETRACTED is the framing that made it look like a
disagreement — the ladder has reported this same `4e-04` at `kt3` all along,
as its own "known downstream divergence", and the round-8 receipt says so.

**AND, SEPARATELY AND FOR REAL: the card resolved into two programs.** The
kt=1..10 ladder and the from-rest year harness built their models from
DIFFERENT configurations of one card. Exactly two leaves differed. That is a
genuine defect, it is not the cause of the `4e-04`, and it is fixed in this
round by making the card resolve the certified values itself.

## 1. THE SIDE-BY-SIDE TABLE

`card_reconciliation/config_diff_before.json`. Both programs are resolved as
OBJECTS and every leaf of the nested config NamedTuple is compared; the
statement that builds each is read from the harness's own source by AST, so
each row is attributed to a line rather than to a transcription.

| model-config field | ladder (`nemo_testcase_l2_gyre_phase3_gate.py:run`) | year (`nemo_testcase_l2_gyre_year_fromrest.py:run_member`) |
|---|---|---|
| `freshwater_closure` | `'real_freshwater'` | `'virtual_salt_flux'` |
| `fix_eta_drift` | `True` | `False` |

and the statements that make them differ:

| | config argument | `_replace` applied |
|---|---|---|
| ladder | `cfg` | `freshwater_closure='real_freshwater'`, `fix_eta_drift=True` |
| year | `card.recipe.model_config` | none |

Nothing else differs — same grid, same z-coordinate, same initial state, same
forcing function, same stepping call, same precision policy. Both harnesses
already import `_surface_forcings`, `expected_masks` and `lego_fields` from
the certified gate and both call `LatLonCGridOceanModel.step`. **The first
divergent statement is the model CONSTRUCTION, and it is the only one.**

**THE TWO ROWS ARE ONE SELECTION, NOT TWO FREE FIELDS.** The model REFUSES
`real_freshwater` with `fix_eta_drift=False`
(`ocean_model_latlon_cgrid.py:3112-3124`), so only three of the four
combinations are constructible at all. The pair is one choice with a
mechanically enforced companion.

**A PRIOR ROUND ALREADY FOUND THIS AND NAMED THE MEASUREMENT IT DID NOT RUN.**
`nemo_testcases_l2_gyre_tke_runaway_receipt.md` records two instruments
disagreeing at `kt=2`, localises the cause to exactly these two fields, and
says: "The discriminating measurement, named and not yet run: step the year
harness's card with those two fields set the gate's way and re-take the
self-A/B." This round is that measurement, and the gate that carries it is
committed so the next round does not have to find it a third time.

**What the year harness's own docstring claimed.** `_gate_module` says the
year "is stepped by exactly the functions the kt=1..10 ladder steps this card
with ... so a year cannot silently run a different program from the ladder
that certified it." That sentence was FALSE in its conclusion while true in
its premise: the functions were shared, the CONFIGURATION was not. Prose is a
pointer, never a citable fact — and this is the second time in this campaign
that a Rule-10 claim was carried by a docstring instead of by a check.

## 2. WHAT THE CONFIG DIFFERENCE IS WORTH — AND IT IS NOT THE `4e-04`

`card_reconciliation/two_path_before_yearroot.json`. Both programs stepped
from rest on the same commit, bit-compared field by field at every step
boundary, on NEMO's own wet masks.

| | `T` | `S` | `u` | `v` | `ssh` |
|---|---:|---:|---:|---:|---:|
| entering `kt=2`, max abs ladder−year | `0.0` | `0.0` | `0.0` | `0.0` | `4.3368e-19` |
| cells bit-unequal | 0/18000 | 0/18000 | 0/17400 | 0/17100 | 600/600 |
| entering `kt=3`, max abs ladder−year | `7.1054e-15` | `1.4211e-14` | `1.8621e-15` | `1.8943e-15` | `1.0517e-17` |
| cells bit-unequal | 47/18000 | 13/18000 | 12955/17400 | 12138/17100 | 575/600 |

Then step 2 run from NEMO's OWN entry state on BOTH programs and scored
against NEMO's `kt=3` entry — the year-owners' own measurement, taken twice:

| program | `T` rms | `S` rms | `u` rms | `v` rms | `ssh` rms |
|---|---:|---:|---:|---:|---:|
| ladder | `4.154395e-04` | `5.993918e-05` | `1.190945e-05` | `1.731725e-05` | `1.182551e-07` |
| year | `4.154395e-04` | `5.993918e-05` | `1.190945e-05` | `1.731725e-05` | `1.182551e-07` |

**Identical to every printed digit.** The reseed is shown to have written
(`3.306e-12`), so neither arm is the free run wearing another name.

**CONFIRMED: the configuration difference does not own the `4.15e-04` K.** It
is worth `1e-14` over two steps and the difference under test is `4e-04` —
ten orders apart. My own leading hypothesis entering this round was that the
config owned it; **that hypothesis is REFUTED by this measurement** and is
recorded here rather than quietly dropped.

**WHICH OF THE TWO ROWS OWNS EVEN THAT `1e-14`?  MEASURED, not split.** The
standing rule is that a two-row table attributes nothing. Section 7 resolves
it: `freshwater_closure` is BIT-INERT on this card (`real_freshwater` against
`virtual_salt_flux` is `0` cells unequal on all five fields), so the entire
`4.3e-19` m and `1.4e-14` K belongs to `fix_eta_drift` and nothing to the
closure. **The table is now attributable, and it attributes to one field.**

**What this does NOT say.** Two steps is not a year. Section 6 measures the
year.

## 3. THE ORACLE HAS TWO RECORDS OF THIS CARD, AND ONE GATE STILL DEFAULTS TO THE UNCERTIFIED ONE

`card_reconciliation/oracle_floor.json`, `both_roots.json`.

The campaign built a deliberate controlled pair: v1 (`gyre_kt1_10`) compiled
without `-fno-tree-vectorize`, so gfortran vectorises transcendental calls
into glibc's `libmvec`; v2 (`round19_oracle_v2_external`, bit-identical to the
year's `nemo_pristine`) compiled with the flag, so those calls go to scalar
`libm`. Same CPP keys, same physics, different low bits from `kt=2`.

Measured in THE LADDER'S OWN UNITS (`score`'s `normalized_max_abs`, against
its `1e-15` bar):

| kt | `T` | `S` | `u` | `v` | `ssh` | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | bit-identical |
| 2 | `4.54e-16` | `5.79e-16` | `6.25e-17` | `9.71e-17` | `4.34e-18` | all UNDER the bar |
| 3 | `5.75e-15` | `1.54e-15` | `2.89e-13` | `2.89e-13` | `1.12e-14` | all ABOVE the bar |

**RETRACTION.** An earlier draft of this round claimed the oracle's own spread
might sit ABOVE the ladder's bar at `kt=2`, which would have made the ladder's
`kt=2` rows uncertifiable. It does not: `5.79e-16` against `1e-15` is `1.7x`
of headroom. What `kt=2` loses to the oracle's build is bit-EQUALITY, which
`score()` already reports separately from status. Killed by the gate's own
`--oracle-floor` mode and independently by a review, before it was recorded
anywhere but here.

**The ladder's rows are root-independent, measured rather than assumed**
(`--score-both-roots`, the ladder's own program against each record):

| kt | field | vs v1 | vs v2 |
|---:|---|---:|---:|
| 2 | `T` | `6.054358e-16` | `6.054358e-16` |
| 2 | `S` | `5.786374e-16` | `5.786374e-16` |
| 2 | `u` | `2.747841e-12` | `2.747840e-12` |
| 2 | `ssh` | `4.770490e-18` | `4.336809e-19` |
| 3 | `T` | `3.722344e-04` | `3.722344e-04` |

**HANDED BACK TO THE LADDER LANE, not fixed here** (they are that lane's
files): the certified receipts pass `--oracle-root .../round19_oracle_v2_external`
while `nemo_testcase_l2_gyre_phase3_gate.py`'s own `ROOT`, `STAGE2_ROOT` and
`STAGE3_ROOT` still default to the v1 tree, and a review reports the gate's
`require` at `:1023` fails on those defaults. Separately,
`year_fromrest/nemo_pristine/binary.sha256` names a `_SM_YRPERT` binary while
the executable beside it is `_SM_R41ADVSP` — a provenance mis-stamp, reported
not repaired.

## 4. THE CALL-CHAIN DIFF, AND THE OTHER HALF OF IT

The config table is the answer to "do they compute the same way". It is not
the answer to "do they START from the same bytes", and a state constructor
that differs in the last bit is a second program just as surely as a config
field is. Both halves are now measured, not described.

| stage | ladder | year | verdict |
|---|---|---|---|
| card | `build_nemo_testcase_card("GYRE-zco")` | same | same object |
| initial state | `card.recipe.initial_state` | the same, round-tripped through numpy with a perturbation field | **BIT-IDENTICAL, measured**: `T 0/21120`, `S 0/21120`, `u 0/21780`, `v 0/22080`, `eta 0/704` cells unequal |
| precision | `PrecisionPolicy.fp64(transcendentals="libm")`, required | the same, required | same |
| forcing | `_surface_forcings(card, state, kt)` from the certified gate | the same function, same module, SHA-256 stamped | same |
| model construction | `LatLonCGridOceanModel(grid, z_coord, cfg)` | `LatLonCGridOceanModel(grid, z_coord, card.recipe.model_config)` | **THE FIRST AND ONLY DIVERGENT STATEMENT** |
| stepping | `model.step(state, dt=card.dt_s, freshwater=, surface_forcing=)` | identical call | same |

The state row is enforced, not printed: the gate REFUSES if any cell differs,
and a `state-drift` plant at `1e-18` turns it red on 3120 of 21120 cells.

## 5. WHAT THE PRIOR RECEIPTS NOW MEAN

| receipt | claim | status after this round |
|---|---|---|
| round-8 ladder | `kt2` `T` `6.05e-16`, `S` `5.79e-16` AT-BAR; `u`/`v` `2.7e-12`/`3.3e-12` DEBT; first-over-bar `kt=2` | **UNCHANGED and reproduced** on this commit (`ladder_sweep_before.json`), and root-independent |
| round-8 ladder | `kt=3..10` are "the campaign's known downstream divergence" | **UNCHANGED**, and it is now named: `kt3` IS step 2's output, the same quantity the year-owners receipt calls the step-2 gap |
| year-owners | "the difference is BORN in step 2"; equal-input `T` `4.1544e-04` K rms on 17999/18000 cells | **UNCHANGED and independently reproduced on the ladder's own program**, to every printed digit, against both oracle records |
| year-owners | its numbers were taken on the year harness's program | superseded only in PROGRAM, not in value: the ladder's program gives the same five numbers, so no conclusion of that receipt moves |
| tke-runaway | "two instruments disagree at kt=2 ... the discriminating measurement, named and not yet run" | **RUN, and the disagreement DISSOLVES — it is the same off-by-one, not the configuration.** See below |

**The eleven orders were never between two receipts. They are between two
consecutive steps of one trajectory, and both instruments always showed them.**

### 5b. THE TKE ROUND'S "TWO INSTRUMENTS DISAGREE" IS THE SAME OFF-BY-ONE

`nemo_testcases_l2_gyre_tke_runaway_receipt.md` records a self-A/B (`--ladder-dump`,
the year harness) finding `kt=2` MOVING by `4.75e-04` while the against-NEMO
gate finds `kt=2` BIT-IDENTICAL before and after, and concludes "Both cannot be
describing the same trajectory". They can, and they do.

`--ladder-dump` writes its array AFTER the step
(`nemo_testcase_l2_gyre_year_fromrest.py:600-611`: `state = model.step(...)`
then `arrays[f"kt{kt:03d}_{name}"] = values`), so its `kt001` is the state
ENTERING step 2 — which is precisely the gate's `kt2.before` row. Lined up:

| the same state | self-A/B row | gate row | self-A/B | gate |
|---|---|---|---:|---:|
| after 1 step | `kt001` | `kt2.before` | **0 fields differ, `0.0`** | **bit-identical, every field** |
| after 2 steps | `kt002` | `kt3.before` | 8 of 13 fields, `4.75e-04` | never compared — that receipt says no matched BEFORE run past `kt=2` exists |

**The two instruments AGREE on every row where both have a number, and the row
they "disagreed" on was never measured by both.** The configuration difference
cannot have caused it either: it is worth `4.3e-19` m at that state, thirteen
orders below `4.75e-04`. **The tke-runaway receipt's open reconciliation is
CLOSED, and closed as a labelling artifact rather than as physics.**

One off-by-one in step labelling generated both of this branch's standing
contradictions. The cure is not vigilance: a row named `kt2` that scores a
state produced by one step, sitting next to a harness whose `kt002` scores a
state produced by two, will regenerate this every round.

## 6. THE UNIFICATION

**What changed: one hunk.** `nemo_testcase_recipe.py`, inside the
`gyre_vector_ene_c2` branch only, the GYRE card now resolves
`freshwater_closure="real_freshwater"` and `fix_eta_drift=True` itself. Nothing
else moved. The L1 cards are untouched — they carry no surface freshwater, so
the field is inert there and moving it would be a choice nobody asked for.

Fixing the CARD rather than the year harness is what makes this a fix and not a
patch: an independent review enumerated every model built from this card and
found **six** call sites passing the bare config — three in the year harness,
three in the year-owners harness — plus two more in other probes. All of them
are unified by one hunk. Every certified gate that already applied the
`_replace` is bit-unchanged, because the `_replace` is now a no-op.

### The gates

| gate | before | after |
|---|---|---|
| ladder `kt=1..10`, 50 scored rows, every field of every row | `ladder_sweep_before.json` | **ALL 50 ROWS IDENTICAL** — `absolute_max`, `normalized_max_abs`, `reference_max_abs`, `status`, `exact` and `n_unequal`, every one unchanged |
| `--config-diff --require-unified` | 2 differing fields, exit 1 | **empty table, exit 0**, and it now covers THREE harnesses |
| the year's initial state vs the card's | bit-identical | bit-identical |
| `--self-check` | had NEVER passed — died on its first plant | **seven plants, all refused, exit 0** |
| step 2 from NEMO's own entry, unified path | `4.154394627928` K (ladder), same (year) | `4.154394627928` K on all THREE arms including the reconstructed legacy control |
| the year, control member, day 30 | `1.4241e-02` K | `1.4241e-02` K (identical to five figures) |
| the year, control member, day 360 | `4.0714e-01` K | `4.0797e-01` K (`+0.20 %`) |

**The ladder did not move by a single bit**, which is what "the `_replace` is
now a no-op" means when it is measured rather than asserted.

**The year, control member, before and after** (wet rms against NEMO, fp64,
same NEMO restarts on both sides, same 12 scored days — one variable):

| day | `T` before [K] | `T` after [K] | `S` before | `S` after | `ssh` before | `ssh` after |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | `1.4241e-02` | `1.4241e-02` | `2.2345e-03` | `2.2345e-03` | `4.5334e-04` | `4.5334e-04` |
| 90 | `5.7269e-02` | `5.7269e-02` | `1.0875e-02` | `1.0875e-02` | `1.3263e-03` | `1.3263e-03` |
| 180 | `2.4400e-01` | `2.4400e-01` | `3.5646e-02` | `3.5646e-02` | `2.5294e-03` | `2.5294e-03` |
| 210 | `3.1845e-01` | `3.1841e-01` | `4.5947e-02` | `4.5946e-02` | `3.0724e-03` | `3.0724e-03` |
| 300 | `4.5241e-01` | `4.5273e-01` | `6.5394e-02` | `6.5401e-02` | `6.2871e-03` | `6.2910e-03` |
| 360 | `4.0714e-01` | `4.0797e-01` | `6.9290e-02` | `6.9236e-02` | `7.6119e-03` | `7.6120e-03` |

**Identical to five figures through day 180, and `0.2 %` apart at day 360.**
That is what a `4.3e-19` m seed does over 2160 steps on this card: nothing for
half a year, then a fifth of a per cent. **The unification does NOT close the
year's gap and was never going to** — the gap is `0.4` K and the program
difference is worth `8e-04` K of it.

**A provenance check worth stating**: the before-run reproduces the year
receipt's recorded day-30 (`1.4241e-02`) and day-360 (`4.0714e-01`) figures to
every printed digit, at a commit 100+ commits later. So the pre-unification
numbers in that receipt were still current when this round replaced them, and
the before/after pair is a controlled comparison rather than two different
models.

### The gate can fail, and that was proven the hard way

The first version of this gate READ each harness's construction statement by
AST. An adversarial review killed it in one line: rebinding the config through
a local —

```python
cfg = card.recipe.model_config
cfg = cfg._replace(freshwater_closure="virtual_salt_flux", fix_eta_drift=False)
```

— reintroduces the exact defect and the gate printed "the SAME RESOLVED
OBJECT" and exited `0`. Three more spellings were equally invisible. **A gate
that cannot see the bug it exists to catch is worse than no gate**, because the
green light is recorded.

The reader now EXECUTES each harness and CAPTURES the object it hands to
`LatLonCGridOceanModel` (Rule 10: instantiate, never trust a declaration). No
spelling can hide, because every spelling ends at the constructor. The AST
reader survives for DISPLAY only and nothing is asserted from it.

The same review found the plants had **never run**: the drift plant perturbed
`A_h`, which is not a field of the config at all (it lives under
`lateral_viscosity`), so `--self-check` died with `AttributeError` on its first
arm and the rest never executed — and the end-to-end test that claimed to prove
the gate could fail was passing because the subprocess CRASHED. Fixed, and the
subprocess test now asserts the refusal MESSAGE so a crash cannot masquerade as
a refusal. A new `program-drift` plant hands the year harness the
pre-unification program **without changing its construction statement by one
character**, and the gate must report the original two rows — which it does.

## 7. THE KNOB THIS ROUND SETS IS HALF COSMETIC, AND THAT IS A FINDING

`card_reconciliation/closure_ablation.json`. Before landing a card change,
ask whether the field it sets does anything. Both independent reviews arrived
at the same arithmetic from opposite directions, so it was measured:
`fix_eta_drift` held `True`, the card stepped twice under all three closures,
trajectories bit-compared.

| | max abs difference from the `real_freshwater` arm | cells unequal |
|---|---:|---:|
| `virtual_salt_flux`, `T` | `0.0` | **0 / 18000** |
| `virtual_salt_flux`, `S` | `0.0` | **0 / 18000** |
| `virtual_salt_flux`, `u` / `v` / `ssh` | `0.0` | **0 / 17400, 0 / 17100, 0 / 600** |
| `none`, `T` | `3.952154e-03` K | 17999 / 18000 |
| `none`, `S` | `1.875104e-03` g/kg | 17850 / 18000 |
| `none`, `ssh` | `6.239189e-05` m | 600 / 600 |

with the sizes a live channel WOULD produce printed beside them, from GYRE's
own forcing this step (max `|emp| = 3.6872e-05` kg/m2/s, `dt = 14400` s,
top cell `10` m): a virtual salt flux would move `S` by `1.8106e-03` g/kg per
step, and a real freshwater source would move `ssh` by `5.1750e-04` m.

**CONFIRMED: `freshwater_closure` selects NOTHING on this lane beyond on/off.**
Turning the channel OFF is a first-order change; choosing between the two ways
of carrying it is bit-for-bit identical. The freshwater reaches the state
through the barotropic continuity source (`F_slow_eta`, built for any closure
other than `"none"`, `ocean_model_latlon_cgrid.py:5598-5601`) and **the virtual
salt term never reaches salinity at all** — a `1.8e-03` g/kg per-step source
that produces `0` unequal cells.

Three consequences, and none of them is "revert".

1. **For NEMO fidelity this is the RIGHT answer by accident.** NEMO's GYRE sets
   `sfx = 0` (`usrdef_sbc.f90:160`) and carries E-P through volume under
   `key_qco`, so a live virtual salt flux would be a deviation. The card was
   already getting NEMO's physics — through a path that does not go through
   the knob that names it.
2. **It is nonetheless a DEFECT in the model's configuration surface**, and it
   is not GYRE-specific: any card on the NEMO RK3 lat-lon lane that selects
   `virtual_salt_flux` silently gets no salt flux. **The mechanism is
   CONFIRMED at a statement, not left plausible** — see section 7b(ii): the
   `nemo_literal` implicit solve is handed only the tracer CONTENT, captured
   before the virtual-salt block, and perturbing the field it ignores by
   `+1e3` moves salinity by exactly `0.0`. Reported to the lane that owns
   `ocean_model_latlon_cgrid.py`, not fixed here.
3. **Half of this round's card change is therefore cosmetic and half is not.**
   `freshwater_closure="real_freshwater"` is bit-inert; `fix_eta_drift=True` is
   the whole measured effect. Both are still landed, because the point is that
   the card resolves the CERTIFIED program rather than that the two fields are
   individually load-bearing.

## 7b. TWO FINDINGS HANDED BACK, BOTH FROM THE DIFF REVIEWS, BOTH MEASURED

Neither is this round's to fix; both are larger than what this round landed.

**(i) A CERTIFIED INSTRUMENT THAT MEASURES NOTHING.** The phase-3 gate's
`pre_implicit_tracer_override` (`nemo_testcase_l2_gyre_phase3_gate.py:1856-1864`)
plants NEMO's own pre-vertical-diffusion tracer state so the rows named
`oracle_pre_zdf.T` / `oracle_pre_zdf.S` score the implicit solve given NEMO's
inputs. **It is inert.** A review planted `+5.0` K and `+5.0` g/kg into the
override and got **all 18000 wet cells bitwise identical** to the un-planted
run; only the 3120 DRY cells moved. Those rows are scoring legoESM's own
advection, not the injection. Rule-12 discharges resting on them need
re-taking.

**(ii) THE `nemo_literal` IMPLICIT SOLVE DISCARDS EVERYTHING WRITTEN TO T/S
AFTER THE CONTENT IS CAPTURED.** `implicit_solver.py:625-648` hands
`implicit_vertical_diffusion_nemo_tracer_pair` only the `content_rhs_*`
arrays; `field_1`/`field_2` are dropped. Measured: perturbing `field_2` by
`+1e3` moves `max|dS|` by **`0.0`** on that arm, against `1000.0` on the shared
Thomas arm. The content is captured at `ocean_model_latlon_cgrid.py:7805`,
BEFORE the virtual-salt block at `:7898-7963` — which is the mechanism behind
section 7, now CONFIRMED rather than plausible. Everything that mutates T/S
between those two points is discarded: the virtual salt flux, the
`ab2_scope="advective"` tracer increment, and `T_solve_in`'s surface and column
sources. Only the three NEMO testcase cards and the DINO preset select this
arm, and none of them run a virtual salt flux today — but finding (i) is a
consequence of the same statement, and that one IS live.

## 7c. RETRACTIONS THIS ROUND

| retracted | what killed it |
|---|---|
| "the two receipts describe different configurations, so one of them is measuring the wrong thing" — the premise this round was commissioned on | the ladder's `kt3` row and the year-owners' step-2 row are the SAME five numbers to every printed digit |
| "the configuration difference owns the `4.15e-04` K" — my own leading hypothesis entering the round | both programs produce `4.154394627928e-04` K given NEMO's own inputs; the programs differ by `1e-14` |
| "the oracle's own spread may sit ABOVE the ladder's bar at `kt=2`" — my own, drafted and never recorded elsewhere | `--oracle-floor`: `4.54e-16` on `T` against a `1e-15` bar, under it with `1.7x` headroom; it first exceeds the bar at `kt=3` |
| "`freshwater_closure` carries a salinity source that compounds over the year" | `--closure-ablation`: `real_freshwater` against `virtual_salt_flux` is `0` cells unequal on all five fields |
| the first version of this round's own gate, which reported UNIFIED on two different programs | an adversarial review rebound the config through a local and the gate never saw it |

## 7d. THE FIX EXPOSED THREE STALE CITATIONS, AND A COMMITTED GATE CAUGHT THEM

Rule 12, in miniature. Inserting eighteen lines into
`nemo_testcase_recipe.py` moved three anchors the receipt-citation gate pins:
`274 -> 292`, `311 -> 329`, `931 -> 949`. The gate went red exactly as built.
Corrected by the procedure that gate already records in its own map comment
from round 34 — update the citation, record the shift beside it, re-audit —
and `audit_map()` is empty afterwards. **Pure integer corrections; no
citation's SYMBOL changed, so nothing any receipt claims has moved.**

One of the three lives in `round8_receipt.md`, which another agent is editing
concurrently on this branch. It is a single integer on one line, in a section
about the pressure-gradient selector rather than the WS-RK3 stage program, and
it is named here so that agent can take theirs on a conflict without losing
anything.

## 7e. THE SUITE

`5 failed, 1083 passed, 7 skipped, 20 deselected in 2699.29s (0:44:59)` on the
whole of `tests/ocean/fidelity`, and every one of the five is accounted for:

| failure | cause | disposition |
|---|---|---|
| `test_nemo_testcase_round35_stamp_scope.py`, 2 tests | **MY MEASUREMENT, not my code**: I ran the suite with `LEGOESM_GATE_ALLOW_DIRTY=1` in the environment, and these two tests assert that exact latch does NOT leak | pass without the variable — re-run quoted below |
| `test_nemo_testcase_receipt_citation_gate.py`, 3 tests | **MINE, real**: the card edit shifted three pinned citations | fixed, section 7d |

After both: `24 passed in 1.60s` over those two files, and
`70 passed, 2 deselected in 52.90s` over the reconciliation, year, year-owners
and certified phase-3 gate files. The new gate's own file:
`8 passed, 2 deselected`, plus `2 passed` on the slow-marked end-to-end pair.
`--self-check`: seven plants, all refused, exit `0`.

## 8. CHOICES, ASKED AND UNASKED

| # | choice | status |
|---|---|---|
| 1 | The GYRE card now resolves `freshwater_closure="real_freshwater"` and `fix_eta_drift=True` itself, instead of leaving each harness to `_replace` them | **NOT A CHOICE, by the campaign's own rule**: these are the values the certified kt=1..10 ladder has always run, and every kt=2 stage gate on this branch already applies the same pair. The card is made to resolve the certified program so there is ONE card. The ladder's `_replace` becomes a no-op and is left in place (that file belongs to another lane) |
| 2 | Which oracle record this round scores against | **NOT CHOSEN.** The gate takes the root as an explicit argument at every call site, scores against BOTH where it matters, and stamps the identity in every artifact |
| 3 | The L1 cards (`LOCK_EXCHANGE-zco`, `OVERFLOW-zps`) are left untouched | Deliberate and scoped: the edit is inside the `gyre_vector_ene_c2` branch only. Those cards have no surface freshwater, so the field is inert there and changing it would be an unasked choice |
| 4 | Whether `fix_eta_drift` should be on at all | **NOT TAKEN — ASKED, and it is the round's biggest open finding.** See question 1 below. Taking it would put the card back out of step with the ladder, i.e. two programs again, so it is not this round's to take |
| 5 | Whether the ladder gate's stale default oracle root should move to v2 | **NOT TAKEN.** That file belongs to the ladder/stage lane; reported in section 3, not edited |

**UNASKED list: empty.**

## 9. OPEN QUESTIONS

1. **`fix_eta_drift` has no NEMO analogue, the certified card selects it, and
   the guard that FORCES it is vacuous on this card.** This is the round's
   biggest finding and it needs a decision.
   - NEMO adds `emp` LOCALLY (`sshwzv.f90:137`,
     `pssh(Kaa) = (pssh(Kbb) - rDt*(r1_rho0*emp + zhdiv))*ssmask`) and has NO
     global sum anywhere in the free-surface path. legoESM's fixer adds a
     GLOBAL uniform eta shift sized by an area-weighted volume residual
     (`ocean_model_latlon_cgrid.py:6588-6651`). Rule 9: never add a stabilizer
     the oracle lacks.
   - The `raise` that makes `real_freshwater` require it
     (`:3006-3018`) justifies itself by a volume defect of
     `0.4667 * sum(A*F)/rho_0`. **On GYRE that sum is ZERO**: the card's `emp`
     is de-meaned by NEMO itself (`usrdef_sbc.f90:151-158`) and measures
     `5.5e-22` kg/m2/s in area-weighted domain mean at both kt=1 and kt=4320,
     so the implied uniform shift is `7.8e-21` m/step against a LOCAL source of
     `5.2e-04` m/step. The guard's own text names its provenance — "the cosine
     filter's average over n=20" — and GYRE runs `nemo_ab3am4` at n=50.
   - Measured over 12 steps on both arms: global-mean eta stays below
     `1.8e-18` m with the fixer ON and `2.8e-19` m with it OFF. **It has
     nothing to correct.** What it does instead is seed a difference:
     ON minus OFF at step 12 is `4.9e-12` m in eta, `1.9e-10` K in `T`.
   - **THE QUESTION: (a) leave the card as landed, matching the certified
     ladder, and carry a global projection NEMO has no analogue for; or (b)
     scope the guard to configurations where `sum(A*F) != 0` and then move BOTH
     the ladder and the card to `real_freshwater` with `fix_eta_drift=False`.**
     I would take (b) — it is the Rule-9 answer and the guard's justification
     is measurably absent here — but it changes the certified ladder, so it is
     a user decision and a ladder-lane change, not one this round takes.
2. **The ladder gate's own `ROOT`/`STAGE2_ROOT`/`STAGE3_ROOT` default to the
   v1 oracle tree while the certified receipts pass a v2 root.** Handed back.
3. **`year_fromrest/nemo_pristine/binary.sha256` names a different build than
   the executable beside it.** A provenance mis-stamp; reported, not repaired.
4. **The step-label collision will regenerate.** `kt2` on the gate and `kt002`
   on the year harness name states one step apart. A shared naming convention,
   or a gate that refuses a comparison whose two sides carry different step
   counts, is the only thing that stops this recurring.

## 10. THE REVIEWS

Two independent fresh Claude reviews before the code (the claim and the
call-chain diff), two after (the diff). **codex was unavailable — it is working
concurrently on the WS-RK3 stage program in a separate clone — and GLM is
unavailable on this account, so all four reviewers were Claude.** Said here
rather than left implicit.

| review | finding | disposition |
|---|---|---|
| claim A | the premise is a misreading: `7.6e-16` is the ladder's `kt2.before` row, after ONE step, and the two harnesses AGREE at `kt3` | **ADOPTED — it is the verdict.** Converged independently with my own measurement |
| claim A | the config diff is numerically inert here; the proposed fix changes nothing | **ADOPTED**, and measured to `1e-14` over two steps and `0.2 %` over a year |
| claim A | `fix_eta_drift` is not NEMO-faithful; `sshwzv.f90:137` is local | **ADOPTED** — question 1 |
| claim A | `freshwater_closure` may be dead on this lane; `1.8e-03` g/kg per step predicted, `1.4e-14` measured | **MEASURED and CONFIRMED** — section 7, and it is why `--closure-ablation` exists |
| claim B | my "the oracle's spread may be above the bar" is FALSE; `5.79e-16` against `1e-15` | **RETRACTED**, and my own gate had already killed it |
| claim B | the two oracle records are a deliberate vectorised/scalar-math pair, not a defect | **ADOPTED** — section 3 |
| claim B | the ladder gate mixes oracle families: its own defaults are v1, the receipts pass v2 | **HANDED BACK** — question 2 |
| diff A | **KILL**: the source-reading reader misses a config rebound through a local, so `--require-unified` passed on two different programs | **FIXED** — the reader now EXECUTES and captures |
| diff A | **KILL**: the plants had never run (`A_h` is not a field), so `--self-check` was red and an end-to-end test passed on a crash | **FIXED** — seven plants, all refused, and the subprocess test now asserts the refusal message |
| diff A | `OWNERS_HARNESS` declared and never checked | **FIXED** — a third captured program, covered by `--require-unified` |
| diff A | `flatten` and `config_rows` attacks all FAILED; the card change breaks nothing (`104 passed`) | recorded |
| diff B | the guard forcing `fix_eta_drift` is vacuous on GYRE; `sum(A*F) = 5.5e-22` | **ADOPTED** — question 1, with its numbers |
| diff B | the `nemo_literal` implicit solve discards `field_1`/`field_2`; VSF death CONFIRMED at a statement | **ADOPTED** — section 7b(ii) |
| diff B | `pre_implicit_tracer_override` is INERT: a `+5.0` K plant moves 0 of 18000 wet cells | **HANDED BACK** — section 7b(i), the round's most consequential hand-back |
| diff B | the year receipt's headline was produced by a program that no longer exists at HEAD | **ADOPTED** — section 6 measures the before/after and section 5 says which claims survive |
| diff B | recommends landing `real_freshwater` with `fix_eta_drift=False` after fixing the guard | **NOT TAKEN, ASKED** — it would put the card out of step with the certified ladder, i.e. two programs again. Question 1 |
