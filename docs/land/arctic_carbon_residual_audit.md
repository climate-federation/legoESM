# Arctic / high-latitude carbon residual — audit finding (2026-07-15)

Companion to `carbon_equilibrium_audit.md`. Records the diagnosis of the residual
high-latitude carbon deficit in the science-grade global carbon IC, so the
investigation is not repeated. **Conclusion: the deficit is an input / structural
cold-tolerance limit, not a parameter bug — no parameter fix is warranted.**

## The deficit

Science-grade global carbon IC (real ERA5 climatology, 187 PFT×climate
archetypes; built by `scripts/data/build_global_carbon_ic.py`). Area-weighted
SOC 3.47, live biomass 7.88 kgC/m² (obs SOC ~9.5). Maps:
`scripts/plot/plot_global_carbon_ic.py`. The deficit is **latitudinally
localised** — not a global scaling error:

| band | SOC | live bio | % cells dead |
|------|-----|----------|--------------|
| tropics | 2.5 | 12.7 | 7 |
| N-temperate | 3.9 | 5.5 | 13 |
| **S-boreal 50–60°N** | **7.9** | **4.3** | **7 (healthy)** |
| N-boreal 60–70°N | 2.8 | 1.1 | 49 |
| high-arctic 70–90°N | 0.1 | 0.04 | 97 |

## Finding 1 — "boreal_forest death" was a harness artifact

The `scripts/validate/land_carbon_equilibrium.py` pixel harness reported
needleleaf_evergreen_boreal (boreal_forest) dead. It is **not**: the pixel
harness forces a near-constant `T_init` (no warm summer), whereas the real
ERA5-driven archetypes carry a seasonal cycle (Tsummer 284–293 K). In the real
climate S-boreal is healthy (SOC 7.9) and needleleaf_evergreen archetypes are
10/12 alive (SOC→14.4). This is the controlled-comparison trap: the pixel
changed the forcing, so its "dead" is a confound. **The pixel harness is not a
valid screen for any high-latitude PFT** (it kills even healthy boreal cells
gates-off).

## Finding 2 — larch death is climate-limited, NOT a parameter bug

`needleleaf_deciduous_boreal` (larch) equilibrates to biomass 0 / GPP 0 at every
climate. Ruled out, in order:

1. **Leaf economics.** Larch carried evergreen-needle economics
   (Vc_max25/LCMA/g1 = 43/70/6, ≈ needleleaf_evergreen 43/80/6) despite being
   deciduous; *Larix* is a high-SLA, high-rate deciduous conifer (LCMA ~40–45,
   g1 ~9; Kloeppel 1998, Reich GLOPNET 1997). But mapping larch to deciduous
   economics (46/45/9) leaves it **0/0, GPP=0.0 exact** — larch never grows
   leaves at all, so leaf economics are irrelevant. Not the cause.
2. **Phenology.** Identical to the healthy sibling broadleaf_deciduous_boreal
   (both take the deciduous branch of `compute_phenology`, keyed on
   `config.evergreen`; the `cold_deciduous` flag is inert when the opt-in
   dormancy gate is off). Not the cause.
3. **PFT-vs-climate swap — the decider.** Running the healthy
   broadleaf_deciduous_boreal PFT at **larch's exact climates** gives **0/0**
   (12/12 dead); the same PFT at its own (warmer) climates lives (GPP→807). So
   larch's cells are genuinely non-viable — the healthy tree dies there too.

There is a **model-wide deciduous viability threshold (~mat 277 K, ~4 °C annual
mean)**; larch's cells (mat 265–277 K) all fall below it. Even
broadleaf_deciduous_boreal is 8/12 dead at its own climates, so boreal-deciduous
cold tolerance is marginal model-wide (the DALEC cold-start / leaf-out lock in
cold climates: a cold-started deciduous stand can drop C_fol→0 faster than it can
regrow, and once LAI=0 → GPP=0 the state is locked).

## Productivity gates (opt-in NSC-gated R_maint + cold-deciduous dormancy)

Shipped opt-in / default-off in PR #1015. They do **not** cleanly resolve this:
gates-ON in the real archetype climate leave larch 0/0, and the apparent
temperate/tropical shifts in the ON-vs-shipped comparison are **confounded**
(the shipped OFF finidat predates the #897 canonical-FvCB merge). The pixel-
harness "tundra/shrub revival" was a cool-synthetic-climate artifact. **Keep the
gates opt-in / default-off; do not flip the defaults.**

## Recommendation

No parameter or gate change ships from this audit. The delivered IC is already
reasonable (SOC 3.47, S-boreal healthy); the coldest-boreal deficit (larch +
coldest 60–90°N cells) is a **characterised model cold-tolerance limit**, not a
mystery. The genuine remedy is a model-wide cold-climate deciduous leaf-out /
cold-start handling (carbon-conserving reserve-backed spring leaf-out that cannot
lock at LAI=0) — a deliberate structural sub-project with an uncertain payoff for
one high-latitude PFT, deferred here rather than forced.
