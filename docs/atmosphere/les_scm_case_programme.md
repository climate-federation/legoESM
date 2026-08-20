# LES → SCM case programme

Source: the GISS case-study slide *"Field campaigns → LES → ModelE in SCM
mode"* (A. Fridlind), which lists sixteen boundary-layer and convective
regimes, each with a field campaign, an LES intercomparison specification and
an aerosol specification.

This page records which of them legoESM can actually run today, and what the
rest are waiting on. It is a plan, not a claim about the code: the status
column is verified by the case registries themselves
(`SAM_SCM_CASES` and `ANALYTIC_SCM_CASES` under
`packages/atmosphere/legoesm/atmosphere/forcing/scm/`) and by the tests that
load every registered case.

## What "runnable" means here

A case is runnable when **both** arms exist and are registered:

1. a legoESM LES driver that integrates the case from its own forcing deck,
   producing a reference under `results/`, and
2. a matched single-column twin built from the *same* deck, so an SCM closure
   can be scored against the LES rather than against a differently forced
   column.

A case with only one arm is not listed as runnable, because the comparison the
programme is for cannot be made.

Runnable is not the same as *scored*. Registering a case makes it selectable and
integrable; producing its LES reference is a separate GPU run, and scoring a
closure against that reference is a third step. No RF02 reference frames exist
yet — the case has been integrated only at smoke resolution.

Two known differences apply to every stratocumulus case here, RF02 included, and
are not specific to it. The single-column arm runs with condensation switched
off by default, so it carries supersaturated vapour (up to 22 % on RF02's
initial column, 12 % on RF01's) where the LES holds liquid, and with no liquid
the shared Stevens (2005) longwave has no cloud to cool from. Both are listed in
the tuner's own `known_scm_les_differences` and are identical across the closures
being compared.

## Status

| # | Regime | Case (reference) | Aerosol in the spec | legoESM status |
|---|---|---|---|---|
| 1 | dry convective boundary layer | idealized (Bretherton & Park 2009) | — | **substitute**: `cbl` (Nieuwstadt CBL_N91) and `wangara` (Day 33) cover the regime from different specifications |
| 2 | dry stable boundary layer | GABLS1 (Cuxart et al. 2006) | — | **runnable**: `gabls1` |
| 3 | marine stratocumulus | DYCOMS-II RF02 (Ackerman et al. 2009) | observed (2 modes) | **runnable**: `rf02`, at the intercomparison's fixed 55 cm⁻³ droplet concentration — not aerosol-aware. That concentration is the case's only drizzle lever here and it is a live one: it produces 5.3× the rain tendency of RF01's 140 cm⁻³ on the same column |
| 4 | marine trade cumulus (shallow) | BOMEX (Siebesma et al. 2003) | — | **runnable**: `bomex` |
| 5 | marine trade cumulus (deep, raining) | RICO (van Zanten et al. 2011) | — | **runnable**: `rico` |
| 6 | stratocumulus-to-cumulus transition (Lagrangian) | SCT (Sandu & Stevens 2011) | — | **substitute**: `astex` (ASTEX flight 209) is the other Lagrangian Sc→Cu deck; the Sandu & Stevens cases themselves are not in the repo |
| 7 | continental cumulus (ensemble) | RACORO (Vogelmann et al. 2015) | observed profile (3 modes) | **needs forcing data** |
| 8 | Arctic mixed-phase stratus | M-PACE (Klein et al. 2009) | observed (2 modes) | **needs forcing data** |
| 9 | Antarctic mixed-phase stratus (Lagrangian) | AWARE (Silber et al. 2019, 2021, 2022) | estimated (1 mode) | **needs forcing data** |
| 10 | tropical deep convection | TWP-ICE (Fridlind et al. 2012) | observed profile (3 modes) | **needs forcing data** |
| 11 | mid-latitude synoptic cirrus (Lagrangian) | SPARTICUS (cf. Mühlbauer et al. 2014) | — | **needs forcing data** |
| 12 | mid-latitude cold-air outbreak (Lagrangian, ensemble) | ACTIVATE (Tornow et al. 2021, 2022) | observed profile (3 modes) | **needs forcing data** |
| 13 | high-latitude cold-air outbreak (Lagrangian, ensemble) | COMBLE (Tornow et al., in prep.) | observed/estimated (3 modes, 1 INP) | **needs forcing data** |
| 14 | marine cumulus and congestus (Lagrangian, ensemble) | CAMP²Ex (Stanford et al., in prep.) | observed profiles (3 modes) | **needs forcing data** |
| 15 | subtropical marine deep convection (Lagrangian, ensemble) | SEAC⁴RS (Stanford et al., in prep.) | observed profiles (TBD) | **needs forcing data** |
| 16 | continental sea-breeze convection (Lagrangian, ensemble) | TRACER (Matsui et al., in prep.) | observed profiles (TBD) | **needs forcing data** |

legoESM additionally runs DYCOMS-II **RF01** (`dycoms`), a neutral Ekman layer
(`ekman`), and plane-CRM drivers for GATE and LBA, none of which appear on the
slide.

## What the blocked cases are waiting on

None of rows 7–16 can be added without their forcing specification. Each needs
an initial sounding, a large-scale forcing time series and a surface boundary
condition, in a format one of the existing readers can ingest — either a gSAM
`snd`/`lsf`/`sfc`/`prm` deck (`data/les_cases/`, read by
`sam_case_forcing.py`) or a DEPHY-SCM driver file (read by `dephy_scm.py`).
Writing a sounding from a paper's figures is not an acceptable substitute; it
would silently produce a case that is not the published one.

Two of them are closer than the rest:

* **M-PACE** and the **Sandu & Stevens transition** cases are distributed in
  DEPHY-SCM format by the DEPHY/GASS community database, which this repo
  already has a loader for. They still need an LES arm before they count as
  runnable here.
* **RACORO** and **TWP-ICE** have cached decks for *neighbouring* campaigns
  (`ARM9707`, `TOGA`), which are different IOPs and must not be relabelled as
  the slide's cases.

## The aerosol column is a separate gap

Every legoESM LES case above prescribes a single cloud-droplet number
concentration (140 cm⁻³ for RF01, 55 cm⁻³ for RF02, 100 cm⁻³ for ASTEX) rather
than an aerosol size distribution. Nine of the sixteen rows specify observed or
estimated aerosol modes, and one specifies an ice-nucleating particle
concentration; running those as specified needs an activation path from aerosol
modes to droplet number, which the LES drivers do not have.

The droplet number is also *prescribed*, not predicted, so it never depletes by
collision–coalescence and there is no droplet–drizzle–liquid-water feedback.
For RF02 that is the difference between reproducing the case's first-hour mean
drizzle, which the fixed concentration does, and reproducing the thinning and
patchiness the drizzle causes later, which it cannot.
