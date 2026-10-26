# LES → SCM case programme

Source: the GISS case-study slide *"Field campaigns → LES → ModelE in SCM
mode"* (A. Fridlind), which lists sixteen boundary-layer and convective
regimes, each with a field campaign, an LES intercomparison specification and
an aerosol specification.

This page records which of them legoESM can run today, where each forcing came
from, and what the rest are waiting on. It is a plan, not a claim about the
code: the status column is backed by the case registries themselves
(`SAM_SCM_CASES`, `ANALYTIC_SCM_CASES` and `DEPHY_SCM_CASES` under
`packages/atmosphere/legoesm/atmosphere/forcing/scm/`) and by tests that load
and integrate every registered case.

## Three levels of "we have this case"

**Matched pair.** A legoESM LES driver integrates the case from its own forcing
deck, *and* a single-column twin is built from the same deck. Only these can be
used to score a turbulence closure against an LES, which is what the programme
is for.

**Single column only.** The published forcing is registered and the column
integrates, but no legoESM LES runs the case, so there is nothing to score
against except observations.

**Not present.** The forcing is not obtainable, or exists only behind an
account, or the case has no 1-D forcing at all.

Runnable is also not the same as *scored*: producing an LES reference is a
separate GPU run, and no reference frames exist yet for the cases added most
recently.

Where a case forces the column in a way this model cannot represent — the
nudging that several of them use — the registry says so in the case's own note,
and a test checks that every unrepresentable channel the loader reports has been
declared. A case with an undeclared gap is one quietly running different physics
from its specification.

## Status

| # | Regime | Case (reference) | Aerosol in the spec | legoESM status |
|---|---|---|---|---|
| 1 | dry convective boundary layer | idealized (Bretherton & Park 2009) | — | **matched pair, substitute case**: `cbl` (Nieuwstadt CBL_N91) and `wangara` (Day 33) cover the regime from different specifications |
| 2 | dry stable boundary layer | GABLS1 (Cuxart et al. 2006) | — | **matched pair**: `gabls1` |
| 3 | marine stratocumulus | DYCOMS-II RF02 (Ackerman et al. 2009) | observed (2 modes) | **matched pair**: `rf02`, at the intercomparison's fixed 55 cm⁻³ droplet concentration — not aerosol-aware. That concentration is the case's only drizzle lever here and it is a live one: it produces 5.3× the rain tendency of RF01's 140 cm⁻³ on the same column |
| 4 | marine trade cumulus (shallow) | BOMEX (Siebesma et al. 2003) | — | **matched pair**: `bomex` |
| 5 | marine trade cumulus (deep, raining) | RICO (van Zanten et al. 2011) | — | **matched pair**: `rico` |
| 6 | stratocumulus-to-cumulus transition (Lagrangian) | SCT (Sandu & Stevens 2011) | — | **single column**: `sandu_ref`, `sandu_fast`, `sandu_slow`, the three published composites, from DEPHY-SCM. The composites nudge temperature and moisture toward the trajectory analysis and this model cannot, so the free troposphere is unconstrained over the three days; the surface temperature, subsidence and geostrophic wind are applied. `astex` (ASTEX flight 209) remains the matched pair for the same regime |
| 7 | continental cumulus (ensemble) | RACORO (Vogelmann et al. 2015) | observed profile (3 modes) | **not present**: needs the ARM RACORO forcing ensemble. `armcu` (ARM SGP, 21 June 1997) is registered as a regime PROXY and is a different IOP |
| 8 | Arctic mixed-phase stratus | M-PACE (Klein et al. 2009) | observed (2 modes) | **single column**: `mpace`, from DEPHY-SCM. Its thermodynamic forcing runs as specified; its wind does not — the case nudges the wind toward its initial profile, which this model's forcing cannot represent, so the column runs with no wind forcing and no Coriolis |
| 9 | Antarctic mixed-phase stratus (Lagrangian) | AWARE (Silber et al. 2019, 2021, 2022) | estimated (1 mode) | **not present**: no public case forcing found |
| 10 | tropical deep convection | TWP-ICE (Fridlind et al. 2012) | observed profile (3 modes) | **not present**: needs the ARM variational-analysis forcing |
| 11 | mid-latitude synoptic cirrus (Lagrangian) | SPARTICUS (cf. Mühlbauer et al. 2014) | — | **not present**: DEPHY ships an unrelated idealized cirrus case (Borella 2025), not this one |
| 12 | mid-latitude cold-air outbreak (Lagrangian, ensemble) | ACTIVATE (Tornow et al. 2021, 2022) | observed profile (3 modes) | **not present**: no public case forcing found |
| 13 | high-latitude cold-air outbreak (Lagrangian, ensemble) | COMBLE (Tornow et al., in prep.) | observed/estimated (3 modes, 1 INP) | **single column, COMBLE-FORCED**: `comble` runs the intercomparison's thermodynamic and dynamic forcing (V2.4) — skin temperature, geostrophic wind, initial state — but not its three prognostic aerosol modes, sea-spray source or ice-nucleating-particle closure. Those are first-order physics in a cold-air outbreak, so this is not a submission to that intercomparison |
| 14 | marine cumulus and congestus (Lagrangian, ensemble) | CAMP²Ex (Stanford et al., in prep.) | observed profiles (3 modes) | **not present**: no public case forcing found |
| 15 | subtropical marine deep convection (Lagrangian, ensemble) | SEAC⁴RS (Stanford et al., in prep.) | observed profiles (TBD) | **not present**: no public case forcing found |
| 16 | continental sea-breeze convection (Lagrangian, ensemble) | TRACER (Matsui et al., in prep.) | observed profiles (TBD) | **not present**: the TRACER MIP is a real-geometry simulation exercise, so there is no 1-D forcing file to register |

legoESM additionally runs DYCOMS-II **RF01** (`dycoms`), ASTEX flight 209
(`astex`) and a neutral Ekman layer (`ekman`) as matched pairs, plus plane-CRM
drivers for GATE and LBA, none of which appear on the slide.

## Where the forcings come from

The single-column cases are **not vendored**. Each is downloaded from the group
that publishes it, into the same cache the gSAM decks use:

```bash
python scripts/data/fetch_dephy_cases.py --list     # what, and from where
python scripts/data/fetch_dephy_cases.py            # fetch everything
```

* **DEPHY-SCM** (`https://github.com/GdR-DEPHY/DEPHY-SCM`) publishes the
  Sandu & Stevens transition composites, M-PACE and the ARM SGP cumulus case in
  a common single-column format that legoESM already reads.
* **COMBLE-MIP** (`https://github.com/ARM-Development/comble-mip`) publishes the
  cold-air-outbreak forcing. It declares the same format but uses the
  intercomparison's own variable names, so the registry carries a small,
  documented rename; a test checks the renamed values against the raw file
  element by element rather than trusting the table. One of those renames is
  load-bearing rather than cosmetic: the file calls its surface-temperature
  series `ts` where the loader looks for `ts_forc`, and without it the case
  came back with no surface forcing at all — a 20 h cold-air outbreak with the
  sea surface switched off, which still integrated and still produced finite,
  innocuous-looking profiles.

Both sources are pinned to an upstream commit and to the SHA-256 of the bytes
this repo was tested against, so a changed file fails the fetch instead of
quietly becoming a different experiment under the same case name.

## What the missing cases would need

Rows 7 and 10 (RACORO, TWP-ICE) exist as ARM forcing datasets. They are
distributed through the ARM Data Center, which needs an account, so they are one
credential away rather than unobtainable. Rows 9, 12, 14 and 15 (AWARE,
ACTIVATE, CAMP²Ex, SEAC⁴RS) are papers in preparation with no public forcing
found. Row 11's cirrus intercomparison forcing was not located. Row 16 has no
1-D forcing by construction.

Building any of these from a paper's figures is not an option: an
observation-derived large-scale forcing is a time series, not a formula, and a
sounding traced off a figure would be a case that resembles the published one
without being it.

## Open questions

**RF01's droplet concentration is unverified.** Its forcing deck specifies none
— it does not precipitate, so the original intercomparison had little reason to
— and the 140 cm⁻³ this repo uses is inherited from its own driver rather than
read off the deck. PyCLES' version of the same case uses 100 cm⁻³ and a reviewer
put the published value at 55. RF02's 55 cm⁻³ is on its deck and is not in
question. This mattered little while the single column could not condense; now
that it can, the number is live on both sides, and settling it needs the
original case description rather than another run.

**M-PACE could be given a wind.** Its wind forcing is a relaxation toward the
initial profile, and this model has no relaxation channel — but it does have a
geostrophic-wind channel, and feeding the relaxation target through it would
restore both the wind and a true Coriolis parameter at 71.75 °N. That is a
reinterpretation of the case rather than a bug fix, so it has not been done; it
is the obvious next step if the case is to be used for anything wind-dependent.

## The aerosol column is still a gap

Every legoESM case above prescribes a single cloud-droplet number concentration
rather than an aerosol size distribution. Nine of the sixteen rows specify
observed or estimated aerosol modes, and one specifies an ice-nucleating
particle concentration; running those as specified needs an activation path
from aerosol modes to droplet number, which the drivers do not have.

The droplet number is also *prescribed*, not predicted, so it never depletes by
collision–coalescence and there is no droplet–drizzle–liquid-water feedback.
For RF02 that is the difference between reproducing the case's first-hour mean
drizzle, which the fixed concentration does, and reproducing the thinning and
patchiness the drizzle causes later, which it cannot. COMBLE's specification
goes further still and asks for three prognostic aerosol modes with a sea-spray
source and an ice-nucleating-particle closure; the registered case runs its
thermodynamic and dynamic forcing, not that aerosol configuration.

## Condensation in the single-column arm

The single-column arm of the LES-vs-SCM comparison runs **with condensation on
by default**, using the same microphysics scheme as the LES and the case's own
droplet concentration. It used to default to no microphysics at all, which left
the cloud layer as supersaturated vapour: no liquid, no condensation buoyancy,
and — for the stratocumulus cases — no cloud for the shared Stevens (2005)
longwave to cool from, which is the entire energy source of those cases.

**Any stratocumulus closure ranking produced before this change is void, not
merely biased.** Cloud-top radiative cooling is the dominant buoyancy source
and entrainment driver in RF01, RF02 and ASTEX, so a column without liquid was
not a worse simulation of those cases but a different flow: a stably stratified
moist layer with no energy source, accumulating supersaturation instead of
condensing it. Dry-case rankings (GABLS1, the Ekman and convective boundary
layers) are unaffected. Results carry the microphysics scheme in their output
manifest, so old and new runs can be told apart; they must not be blended.
