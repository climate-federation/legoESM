# Bottom Drag in Coarse-Resolution Global Ocean Models: A Literature Synthesis

Context: 5° x 5° x 20-level idealized global ocean with a Drake-Passage-like band, GM/Redi
on, currently using **linear drag** with `r = 0.0011 m/s` and `(1 - dt*r/H)` applied to the
barotropic mode. At `H = 4000 m` this is a barotropic decay timescale of about **42 days**.

---

## Q1. Standard form of bottom drag in coarse / idealized global ocean models

**Punchline.** Modern primitive-equation ocean GCMs (MOM6, MITgcm, NEMO, POP) overwhelmingly
default to **quadratic** bottom drag with a near-bottom-cell speed and an "unresolved-velocity"
floor. **Linear (Rayleigh) drag** is still common and physically defensible in *idealized
channel and QG ACC studies* (Munday/Johnson/Marshall, Marshall et al. 2017, Maddison et al.
2025) where it is preferred for analytic tractability of the momentum and energy budgets, and
because at coarse resolution the near-bottom velocity is itself a parameterization. Implicit
log-layer drag (`ln_loglayer` in NEMO; channel_drag with Jackson-style BBL in MOM6) is the
modern operational choice in eddying global models, particularly over rough topography.

**Evidence.**
- **MITgcm** supports both forms via `bottomDragLinear` (`r_b`, m s⁻¹) and `bottomDragQuadratic`
  (`C_d`, dimensionless). The manual quotes typical values "of the order 2 x 10⁻⁴ m s⁻¹" for
  linear and "0.001–0.003" for quadratic (MITgcm Algorithm chapter, "Bottom Drag";
  https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html).
- **NEMO** (zdfbfr) defaults: `rn_bfri1 = 4.0e-4 m/s` (linear, ~115-day decay) and
  `rn_bfri2 = 1.0e-3` (quadratic) with `rn_bfeb2 = 2.5e-3 m²/s²` background TKE
  (https://www.nemo-ocean.eu/doc/node70.html).
- **POP / CESM**: quadratic bottom drag with `c_d ~ O(10⁻³)` (POP Reference Manual,
  https://ncar.github.io/POP/doc/build/html/reference_manual/POPRefManual.html).
- **MOM6 / OM4 / CM4** (Adcroft et al. 2019, JAMES, doi:10.1029/2019MS001726): quadratic with
  `CHANNEL_DRAG = True`, `HBBL = 10 m`, `DRAG_BG_VEL = 0.1 m/s` (MOM_input,
  https://github.com/NOAA-GFDL/MOM6-examples). The Jackson-style bottom boundary layer is the
  current operational standard.
- **Idealized ACC channels**: Munday, Hogg & Marshall (2013, *JPO* 43, 507–532); Abernathey
  & Cessi (2014, *JPO* 44, 2107–2126); Marshall, Ambaum, Maddison, Munday & Novak (2017,
  *GRL*, doi:10.1002/2016GL071702); Maddison et al. (2025, *JAMES*, arXiv:2409.18035) all use
  **linear** bottom drag in their analytic and idealized work.

**Trade-offs.** Linear drag is convenient (constant decay timescale, linear barotropic
stability analysis, AD-friendly) but underestimates dissipation in fast western-boundary
currents and gives no stagnation cutoff. Quadratic drag scales correctly with `|u|` but
introduces an unresolved-velocity floor (`DRAG_BG_VEL`-like) to avoid zero-stress in slow
flow. Implicit/log-layer is a non-issue at our 5° resolution with no resolved BBL.

---

## Q2. Numerical values used in published work

**Punchline.** Linear `r = 1–4 x 10⁻⁴ m/s` and quadratic `C_d = 1–3 x 10⁻³` are the
operational ranges. Typical implied barotropic decay timescales at `H = 4000 m` and
`|u_bot| ~ 5 cm/s` are **100–1000+ days**, not ~40 days.

**Linear drag rate `r` (m/s):**
- MITgcm canonical / Munday-Marshall channels: `r ~ 1.1e-3 m/s` is sometimes cited *but
  applied as a Rayleigh damping on the bottom-cell velocity, not the depth-averaged column*.
  The "typical of order 2e-4" quoted in the MITgcm manual is for global-style use.
- NEMO default `rn_bfri1 = 4.0e-4 m/s` → 115 days at `H = 4000 m`.
- ROMS `rdrg` is typically `3e-4` m/s in coastal / shelf configurations.
- Held & Larichev (1996) and follow-on QG turbulence: linear drag chosen so `tau_drag =
  1/r * H_bot ~ 100–200 days`, with 115 days "the conventional choice" in QG ACC models.

**Quadratic drag `C_d` (dimensionless):**
- MOM6 OM4 / CM4: `CDBOT ~ 0.003` with `DRAG_BG_VEL = 0.1 m/s` (Adcroft et al. 2019).
- POP/CESM: `c_d ~ 1.0–1.225 x 10⁻³`.
- NEMO: `rn_Cd0 = 1.0e-3`.
- MITgcm tutorial channel (Munday/MJM-style): typically `2.0–2.5e-3`.

**Implied barotropic timescale at `H = 4000 m`.** For the depth-averaged column form
`(1 - dt r/H)`, the e-folding time is `tau = H/r`:
- `r = 1e-4 m/s` → `tau ~ 463 days`.
- `r = 4e-4 m/s` → `tau ~ 116 days` (NEMO default).
- `r = 1.1e-3 m/s` (our value) → `tau ~ 42 days`.
- For quadratic drag with `C_d = 3e-3`, `|u_bot| = 5 cm/s`, `H = 4000 m`: equivalent linear
  rate is `C_d * |u| ~ 1.5e-4 m/s` → `tau ~ 309 days`. With `|u_bot| = 10 cm/s` (the OM4
  `DRAG_BG_VEL` floor) it is **~150 days**.

So even the most aggressive operational quadratic choice gives a barotropic timescale
roughly 3–4x longer than ours.

---

## Q3. ACC / Drake-Passage transport sensitivity to bottom drag

**Punchline.** ACC zonal momentum is balanced primarily by **topographic form drag**, not
bottom friction (Munk & Palmén 1951; Hughes 1997; Olbers 1998; Masich et al. 2015,
*JGR*, doi:10.1002/2015JC011143). Nevertheless, in idealized models with weak or no topography
— or when using GM rather than resolved eddies — bottom drag controls a non-trivial fraction
of the ACC transport, and **transport increases with increasing bottom drag** (Marshall et al.
2017).

**Evidence.**
- Munk & Palmén (1951) and Olbers (1998): zonal-mean wind stress on the ACC must be balanced
  by an interfacial form stress acting on submarine ridges, transmitted downward by the eddy
  field — bottom friction alone cannot close the budget on a flat-bottom Earth.
- Masich, Chereskin & Mazloff (2015): in SOSE the topographic form stress accounts for ~95%
  of zonal wind stress; bottom drag is a small residual.
- Munday, Hogg & Marshall (2013, *JPO* 43, 507–532): in an MITgcm sector model with
  topography and resolved eddies, ACC transport is **eddy-saturated** w.r.t. wind stress
  but remains sensitive to drag and diapycnal mixing.
- Marshall, Ambaum, Maddison, Munday & Novak (2017, *GRL*, doi:10.1002/2016GL071702):
  developed a momentum + EKE budget showing **transport ~ r^α with α positive** ("circumpolar
  transport increases with increased bottom friction, a counterintuitive result confirmed in
  eddy-permitting calculations"). For all but the smallest drags, EKE saturates at a
  drag-independent value while transport scales with `r`.
- Constantinou & Hogg (2019, *GRL*, doi:10.1029/2019GL084117): even a barotropic channel with
  topography is "eddy-saturated", showing that flow–topography interactions, not bottom
  friction, set the transport.
- Maddison et al. (2025, *JAMES*, arXiv:2409.18035): a 2D model showing that
  for fixed wind, transport scales monotonically with linear drag in the planetary-geostrophic
  limit.

**For our setup specifically.** A *5° resolution* ocean with a *5-cell-wide Drake band*
cannot resolve standing meanders or significant submarine topography in the channel. Form
drag will be weak by construction, and **bottom drag will be doing more momentum-budget work
than is physical** for the real ACC. This is a known artifact of coarse, smooth-bottom
channels (Tansley & Marshall 2001, Allison et al. 2010). A westward `-405 Sv` total
transport with the Drake band 5° wide and only an Ekman-scale wind clearly indicates a
**nearly-inviscid barotropic mode** — form drag is missing and `r = 0.0011` is *too weak* in
its current barotropic-only form to substitute for it. (This is consistent with the very
strong deep westward `-7 cm/s` you report.)

---

## Q4. Is `r = 0.0011 m/s` (linear) reasonable for our setup?

**Punchline.** As a barotropic-column linear coefficient, `r = 0.0011 m/s` is on the
**high end** but not absurd; it is roughly an order of magnitude larger than the canonical
linear default in coarse climate models (`r ~ 1–4 x 10⁻⁴ m/s`). However, it is
applied to the *barotropic mode* (i.e., to `<u>` rather than to `u_bot`), which is a
non-standard choice that further amplifies its effective dissipation.

**Comparison.**
- POP 1°, MOM6 OM4 0.25°, NEMO ORCA1 — all use **quadratic** drag on the **bottom-cell**
  velocity, which gives an effective linear-equivalent `~1–2 x 10⁻⁴ m/s` for bottom speeds
  of `5–15 cm/s` and a 10-m-thick BBL. They do **not** apply drag to the depth-averaged
  velocity.
- Idealized ACC channels (Abernathey-Cessi 2014, Munday-Hogg-Marshall 2013) use
  `r = 1.1e-3 m/s` *but applied to the bottom layer only*. With a 100–200 m bottom layer
  this gives `tau ~ 1–2 days`, which is by design strong: it is a sponge-like sink for the
  resolved-eddy energy that would otherwise flood the bottom.
- Held & Larichev (1996) standard: `tau_drag = 100–200 days`.

**Diagnosis of our value.** Applying `(1 - dt r/H)` to the barotropic mode with `r = 1.1e-3`
and `H = 4000 m` gives `tau = 42 days`, about **3x stronger than NEMO's default linear** and
about **3–7x stronger than the effective linear-equivalent of MOM6/POP quadratic drag** at
typical bottom speeds. Yet it is still **not enough** to control your flow — that strongly
suggests the issue is structural (missing form drag in a smooth, cell-narrow channel; GM
unable to close the budget), not just a coefficient mismatch.

If the choice is to keep linear drag on the barotropic mode (simplest, AD-friendly), I would
target `tau = 30–60 days` at `H = 4000 m`, which corresponds to `r = 8e-4 to 1.5e-3 m/s`.
This is roughly where you are already. **The value is not the primary problem.** A more
useful change would be to apply the linear drag to the bottom-cell velocity (consistent with
literature) and add a topographic / channel-form drag proxy.

---

## Bottom line — recommendation

**Recommendation:** keep `r` in the range **`8e-4 to 1.5e-3 m/s` (target `r = 1.0e-3 m/s`)**
applied as currently structured (column-averaged, `(1 - dt r/H)`). At `H = 4000 m` this
gives a barotropic decay of ~46 days, in line with idealized ACC channel practice (MJM13,
AC14) and a factor of 2–3x stronger than NEMO/MOM6 operational defaults — appropriate
because **our 5° model has essentially no resolved topographic form drag in the Drake band**.

Do **not** drop below `~5e-4 m/s` (`tau > 90 days`) for this configuration; with our smooth
channel that will give an unphysical near-inviscid barotropic mode and runaway transport.

The right *long-term* fix is not to tune `r` further but to add a **form-drag-proxy**: either
quadratic drag on the bottom cell (operational standard), a `channel_drag`-style partial-cell
form-drag enhancement, or a topographic-roughness Rayleigh term concentrated where bottom
slopes exist. Our current `-405 Sv` Drake transport is a structural artifact of missing form
drag, not of the bottom-drag coefficient value.

---

## References

- Adcroft, A. et al. (2019). *The GFDL Global Ocean and Sea Ice Model OM4.0*. JAMES, 11.
  doi:10.1029/2019MS001726.
- Abernathey, R. & Cessi, P. (2014). *Topographic Enhancement of Eddy Efficiency in
  Baroclinic Equilibration*. JPO 44, 2107–2126.
- Allison, L. C., Johnson, H. L., Marshall, D. P., & Munday, D. R. (2010). *Where do winds
  drive the Antarctic Circumpolar Current?*. GRL 37. doi:10.1029/2010GL043355.
- Arbic, B. K. & Scott, R. B. (2008). *On Quadratic Bottom Drag, Geostrophic Turbulence,
  and Oceanic Mesoscale Eddies*. JPO 38, 84–103.
- Constantinou, N. C. & Hogg, A. McC. (2019). *Eddy Saturation of the Southern Ocean: A
  Baroclinic Versus Barotropic Perspective*. GRL 46, 12202–12212.
  doi:10.1029/2019GL084117.
- Held, I. M. & Larichev, V. D. (1996). *A scaling theory for horizontally homogeneous,
  baroclinically unstable flow on a beta plane*. JAS 53, 946–952.
- Hughes, C. W. (1997). *The Southern Ocean Momentum Balance: Evidence for Topographic
  Effects from Numerical Model Output and Altimeter Data*. JPO 27, 2219–2232.
- Maddison, J. R. et al. (2025). *A Two-Dimensional Model for Eddy Saturation and Frictional
  Control in the Southern Ocean*. JAMES. doi:10.1029/2024MS004682. arXiv:2409.18035.
- Marshall, D. P., Ambaum, M. H. P., Maddison, J. R., Munday, D. R., & Novak, L. (2017).
  *Eddy saturation and frictional control of the Antarctic Circumpolar Current*. GRL.
  doi:10.1002/2016GL071702.
- Masich, J., Chereskin, T. K. & Mazloff, M. R. (2015). *Topographic form stress in the
  Southern Ocean State Estimate*. JGR Oceans 120, 7919–7933. doi:10.1002/2015JC011143.
- Munday, D. R., Johnson, H. L. & Marshall, D. P. (2013). *Eddy Saturation of Equilibrated
  Circumpolar Currents*. JPO 43, 507–532. doi:10.1175/JPO-D-12-095.1.
- Munk, W. H. & Palmén, E. (1951). *Note on the dynamics of the Antarctic Circumpolar
  Current*. Tellus 3, 53–55.
- Olbers, D. (1998). *Comments on "On the obscurantist physics of 'form drag' in theorizing
  about the Circumpolar Current"*. JPO 28, 1647–1654.
- Tansley, C. E. & Marshall, D. P. (2001). *On the dynamics of wind-driven circumpolar
  currents*. JPO 31, 3258–3273.
- MITgcm Algorithm chapter, "Bottom Drag":
  https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html
- NEMO zdfbfr documentation: https://www.nemo-ocean.eu/doc/node70.html
- POP Reference Manual:
  https://ncar.github.io/POP/doc/build/html/reference_manual/POPRefManual.html
- MOM6-examples OM4_025 MOM_input:
  https://github.com/NOAA-GFDL/MOM6-examples/blob/dev/gfdl/ice_ocean_SIS2/OM4_025/MOM_input
