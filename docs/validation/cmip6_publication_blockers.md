# CMIP6 publication blockers that metadata fixes cannot clear

Companion to `cmip_readiness.md`. That file tracks *component* readiness;
this one records the two things that will get legoESM output **rejected at
ESGF publication even when every file is byte-perfect CF and CMOR**, because
neither is a defect in the writer.

Everything below was verified against the current code and the live
controlled vocabularies on **2026-08-03**. Re-verify before acting: this
repo's prose goes stale, and so does the CV.

---

## 1. `institution_id` and `source_id` are not in the CMIP6 CV

### What is wrong

The writer stamps two identifiers that do not exist in the CMIP6 controlled
vocabulary:

| Global attribute | Value legoESM writes | Registered? |
|---|---|---|
| `institution_id` | `CU` | **No** |
| `source_id` | `legoESM-1-0` | **No** |

Checked directly against `WCRP-CMIP/CMIP6_CVs@main` on 2026-08-03:

* `CMIP6_institution_id.json` holds **49** registered institutions. `CU` is
  absent, and *no* entry mentions Columbia at all.
* `CMIP6_source_id.json` holds **132** registered sources. `legoESM-1-0` is
  absent, and no entry matches `lego` or `LEAP`.

These are not cosmetic:

* Both are in the CV's `required_global_attributes` (verified: 30 entries,
  both present), so both are stamped on **every** file.
* `source_id` is also a **filename** component
  (`rsut_Amon_legoESM-1-0_amip_r1i1p1f1_gn_197901-197901.nc`), and both are
  **DRS directory** components under `drs_tree=True`
  (`CMIP6/CMIP/<institution_id>/<source_id>/<experiment_id>/...`).

So the identifiers are not confined to an attribute that could be patched
after the fact — they are baked into the paths and names of everything we
write. This invalidates a whole campaign's output at the last step regardless
of how good the physics or the metadata is.

(The CMIP6 publication validator, PrePARE, checks CV membership for these.
That was **not** confirmed by running it: PrePARE is no longer shipped with
`cmor` (3.15.2) and is not separately packaged on conda-forge or PyPI, so it
could not be run here. The unregistered status above *was* confirmed
directly against the CV files.)

`CFWriter` already warns about this once per process (see
`check_cv_registration` in `packages/core/legoesm/io/cmor_output.py`). Note
what that function actually does: it compares against the two placeholder
constants `_DEFAULT_UNREGISTERED_INSTITUTION_ID` / `_DEFAULT_UNREGISTERED_SOURCE_ID`.
It is a *reminder that the defaults are still in place*, *not* a validator —
it cannot confirm that some other value **is** registered, and it will fall
silent for any non-default value, registered or not.

### What must be registered, and with whom

Both are registered by pull request to **<https://github.com/WCRP-CMIP/CMIP6_CVs>**
(the WCRP CMIP Panel's CV repository). Two separate files:

**`CMIP6_institution_id.json`** — one line, id to full postal description:

```json
"NASA-GISS": "Goddard Institute for Space Studies, New York, NY 10025, USA"
```

**`CMIP6_source_id.json`** — a structured entry. All 132 published entries
carry these eight keys, so treat them as required:

`activity_participation`, `cohort`, `institution_id` (a list), `label`,
`label_extended`, `model_component`, `release_year`, `source_id`

plus `license_info` (present on most). `model_component` is a per-realm map
(`atmos`, `land`, `ocean`, `seaIce`, `aerosol`, `atmosChem`, `landIce`,
`ocnBgchem`) where each realm gives a `description` and a
`native_nominal_resolution` — i.e. registration requires committing to a
public description of the model configuration, not just a name.

### What the values would likely be

**This is the project owners' call, not a decision to make in code**, and the
repo does not currently settle it. The evidence that exists, and its limits:

* The writer's long-form `institution` default is `"Columbia University"`
  (`_global_attrs`), which is the only claim about affiliation in the writer.
* The git remote is `git@github.com:climate-federation/legoESM.git` — i.e.
  the code lives under a `climate-federation` organisation, **not** under a
  Columbia/LEAP one. So the hardcoded `"Columbia University"` and the actual
  home of the project do not obviously agree, and which entity should be
  registered is genuinely open. Resolve that before filing anything.
* Whatever is chosen, `CU` is a poor token: two letters, collides with
  several universities, and the CV's house style is a recognisable centre
  acronym (`NASA-GISS`, `CNRM-CERFACS`, `CCCR-IITM`).
* `source_id` is the right *shape* already — compare `GISS-E2-1-G`. Only its
  registration is missing, not its form.

Do not invent a value and ship it: an unregistered non-default value is
*worse* than the current one, because `check_cv_registration` compares
against the placeholders and will stop warning entirely.

The `license` global needs no attention here — `CMIP6_LICENSE` names only
legoESM and the standard CC-BY-4.0/PCMDI terms, no institution.

### Does the writer need to change once they are registered?

**Partly. It is not purely an external registration.**

* **No change needed in `CFWriter` itself.** `institution`, `institution_id`
  and `model_id` are all already constructor parameters
  (`packages/core/legoesm/io/cmor_output.py`, `CFWriter.__init__`). Pass the
  registered values and the correct attributes are written.
* **`further_info_url` needs no separate edit.** It is derived as
  `https://furtherinfo.es-doc.org/CMIP6.<institution_id>.<source_id>.<experiment_id>.<sub_experiment_id>.<variant_label>`,
  so it follows automatically. Caveat: the URL only *resolves* once the
  matching ES-DOC model documentation is also published — that is a third,
  separate registration, distinct from the two CV PRs.
* **The production AMIP path does need a change.** The CMOR writer for AMIP
  is constructed in `packages/coupler/legoesm/driver/diagnostics.py`, which
  hardcodes `model_id="legoESM-1-0"` and does not pass `institution_id` at
  all, so it silently takes the `"CU"` default. There is no CLI flag for
  either in `scripts/run/run_amip.py`. Making the registered values usable
  therefore needs the two values wired through plus CLI flags, per this
  repo's rule that a new user-tunable field ships a flag in every affected
  run script. `scripts/run/run_omip2_io_smoke.py` hardcodes `model_id` the
  same way.
* Once registered values become the defaults, the warning silences on its own
  — the placeholder constants are what it compares against.

---

## 2. `Omon/tos` and `Omon/tosga` are written in Kelvin, not degC

### What is wrong

CMIP6 `Omon` declares `tos` (sea surface temperature) and `tosga` (its global
average) in **`degC`**. legoESM produces Kelvin.

The vendored table loader records this rather than hiding it:
`UNITS_DEVIATIONS` in `packages/core/legoesm/io/cmor_table_loader.py` maps
`("Omon", "tos")` and `("Omon", "tosga")` to `"K"`, so the *label matches the
data*. That is the safe choice — stamping `degC` onto Kelvin values would
make every number wrong by 273.15 with nothing raising anywhere — but it is
still non-compliant output, and those two variables are unpublishable as
CMIP6 until it is resolved.

### Is it ours to fix, or a deliberate choice?

**It is ours to fix.** It is documented, but it is not a physics constraint
and it was never a considered decision — it is a defect that was *frozen in
place* by the old hand-typed table.

The producer is `scripts/run/run_omip2_io_smoke.py` (verified in current code,
2026-08-03):

```python
tos = T[..., 0] + constants.T_freeze  # surface T → K for CMIP6 tos
```

The comment asserts CMIP6 wants Kelvin. It does not. That line was written to
match the old legoESM table, which said `K` because it was typed by hand and
was wrong. So the deviation is circular: the producer converts to match a
table that was itself mistaken.

### Scope — smaller than it looks

`tos` is written by **only** that one smoke script. No production driver emits
it: the only `CFWriter` constructions outside tests are the AMIP path in
`driver/diagnostics.py`, this smoke script, and the module docstring example.
(`scripts/run/run_omip_core2.py` mentions `tos`, but *reads* it as forcing
input — not CMOR output.)

It is also entirely outside the DECK **amip** priority-1 target, which is
atmosphere-only: `tos` is an `Omon` ocean variable.

### The fix

Two coordinated edits, which must land together:

1. Drop the `+ constants.T_freeze` in `run_omip2_io_smoke.py` (and fix the
   misleading comment) so the data is degC.
2. Remove the two `UNITS_DEVIATIONS` entries, so the writer stamps the
   table's `degC`.

`tests/unit/test_cmor_compliance.py::test_known_deviations_are_exactly_these`
pins the deviation set by exact membership, so it goes red if one side is
changed without the other — which is the intended guard. Do **not** remove
the `UNITS_DEVIATIONS` entry alone: that relabels Kelvin data as degC and
silently corrupts every value.

An ocean owner should confirm there is no downstream consumer relying on
Kelvin `tos` before this lands.

---

## Status summary

| Blocker | Ours to fix? | Blocks DECK amip? | Action |
|---|---|---|---|
| `institution_id` = `CU` unregistered | External + small wiring | **Yes** | PR to `WCRP-CMIP/CMIP6_CVs`, then wire + add CLI flag |
| `source_id` = `legoESM-1-0` unregistered | External + small wiring | **Yes** | Same PR; also ES-DOC docs for `further_info_url` to resolve |
| `Omon/tos`, `Omon/tosga` in K not degC | **Yes** | No (ocean, not amip) | Two-line coordinated fix; needs an ocean owner's sign-off |
