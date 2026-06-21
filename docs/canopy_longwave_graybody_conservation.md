# Two-Big-Leaf Longwave: gray-body conservation analysis and fix

**Status:** analysis / proposed fix. Intended to be implemented in **DifferBESS**
(`process/CanopyLongwaveRadiation.py`) first, then synced to legoESM
(`packages/land/legoesm/land/canopy/radiative_transfer.py`, which is a faithful
port of the DifferBESS routine).

**TL;DR.** The two-big-leaf longwave (LW) scheme in DifferBESS / legoESM (and in
CABLE, from which the idiom derives) uses a *near-black* approximation: each
surface **emits** `ε·σT⁴` but **absorbs** incident LW with absorptivity **1** and
**no reflection** is tracked. This violates Kirchhoff's law (absorptivity =
emissivity) for `ε<1`. At a hypothetical isothermal equilibrium the net LW is
`O(1−ε)·σT⁴` instead of exactly zero, which **warm-biases** the solved leaf/soil
temperatures and the diagnosed LST. This note (1) localises the defect, (2)
quantifies the bias, (3) compares CABLE and CLM5, and (4) gives a drop-in,
provably energy-conserving replacement that **keeps physical `εf, εs < 1`**,
adds the ground reflection **and** first-order leaf backscatter via a closed-form
geometric-series (canopy↔ground interreflection) closure, reduces to the current
scheme at `εf=εs=1` and to CLM5 at `εf=1`, and stays differentiable.

> **This document bundles a second, unrelated canopy fix** (**Part II**, at the
> end): the Penman-Monteith saturation **second-derivative** (`d²eₛ/dT²`,
> `ddesTc`) bug, plus a cross-module saturation-curve inconsistency surfaced by
> the preprocessing audit. Already fixed in legoESM (the `land/stable` merge);
> **DifferBESS still carries it** (`CanopyEnergyBalance.py:349-350`).

---

## 1. The scheme as implemented

`CanopyLongwaveRadiation` (DifferBESS) / `canopy_longwave_rt` (legoESM), fully
coupled branch. With `kd≈0.78` (diffuse), `kb=0.5/cosθ` (beam), effective LAI
`L_eff = LAI·CI`, gap fraction `gap_LW = exp(−kd·L_eff)`, total interception
`W_tot = 1 − gap_LW`, and the depth-resolved two-stream weights

```
W_sun_sky  = kd·(1 − e^{−(kb+kd)L_eff})/(kb+kd)
W_sun_soil = kd·e^{−kd L_eff}·L_eff·exprel((kd−kb)L_eff)      # = kd·(e^{−kb L}−e^{−kd L})/(kd−kb)
W_sh_sky   = W_tot − W_sun_sky
W_sh_soil  = W_tot − W_sun_soil
```

emitted flux densities

```
Ls     = εs·σ·Ts⁴          # soil emission     (carries εs)
Lf_Sun = εf·σ·Tf_Sun⁴      # sunlit-leaf emission (carries εf)
Lf_Sh  = εf·σ·Tf_Sh⁴       # shaded-leaf emission (carries εf)
```

and `La` = downwelling atmospheric LW (the raw forcing). The net absorbed fluxes
are

```
ALW_Sun  = W_sun_soil·(Ls − Lf_Sun) + W_sun_sky·(La − Lf_Sun)
ALW_Sh   = W_sh_soil ·(Ls − Lf_Sh ) + W_sh_sky ·(La − Lf_Sh )
ALW_Soil = W_sun_soil·Lf_Sun + W_sh_soil·Lf_Sh + gap_LW·La − Ls
```

with the top-of-canopy upward emission and LST diagnostic

```
Lcanopy_up = W_sun_sky·Lf_Sun + W_sh_sky·Lf_Sh
L          = Lcanopy_up + gap_LW·εs·σ·Ts⁴
E_eff      = (1 − gap_LW)·εf + gap_LW·εs
LST        = (L / (E_eff·σ))^{1/4}
```

---

## 2. The conservation defect

Two coupled problems, both visible in the `ALW_*` lines above:

1. **Absorptivity ≠ emissivity (Kirchhoff violation).** The incident terms `Ls`,
   `La` are absorbed with **absorptivity 1** (no `εf` in front), while the
   surface emits with `εf` (`Lf_*`). Kirchhoff's law requires a leaf's LW
   absorptivity to equal its emissivity `εf`. Likewise the soil absorbs
   `gap_LW·La` with absorptivity 1 but emits `Ls = εs·σTs⁴`.

2. **No reflection.** The `(1−ε)` fraction that a real gray surface *reflects*
   is simply dropped — neither the ground reflecting downwelling LW nor the
   leaves reflecting intercepted LW.

### Isothermal-equilibrium test (the diagnostic symptom)

Put leaves, soil and a black sky all at the same temperature `T`
(`La = σT⁴`, `εf=εs=ε`). Then `Ls = Lf_Sun = Lf_Sh = ε·σT⁴`:

```
ALW_Sun = W_sun_soil·(εσT⁴ − εσT⁴) + W_sun_sky·(σT⁴ − εσT⁴)
        = 0 + W_sun_sky·(1 − ε)·σT⁴            ≠ 0
```

The sunlit leaf *gains* `W_sun_sky·(1−ε)·σT⁴` at equilibrium — thermodynamically
impossible (a body in an isothermal enclosure has zero net radiative exchange
regardless of its emissivity). Summing all three components, the **system** gains

```
ΣALW = La − gap_LW·Ls − Lcanopy_up
     = σT⁴·[1 − gap_LW·εs − (1−gap_LW)·εf]
     ──ε≡εf≡εs──►  σT⁴·(1 − ε)                  (the un-reflected fraction of La)
```

i.e. the entire `(1−ε)` of incident LW that should have been reflected is instead
absorbed. This is why the existing `test_longwave_isothermal_blackbody_zero`
uses `ε=1` — at `ε=1` the term vanishes and the scheme *is* exactly
conservative. For `ε<1` it is not.

---

## 3. Magnitude of the bias

### 3.1 Over-absorption

Relative to a Kirchhoff-consistent gray surface (absorb `ε·incident`), the
near-black form over-absorbs. Holding the incident streams fixed, the
per-component absorptivity correction is

```
ΔALW_leaf = (1−εf)·[ W_sun_soil·Ls + W_sun_sky·La + W_sh_soil·Ls + W_sh_sky·La ]
ΔALW_Soil = (1−εs)·[ W_sun_soil·Lf_Sun + W_sh_soil·Lf_Sh + gap_LW·La ]   # canopy-incident + gap·La
```

(`ΔALW_Soil` must include the canopy LW reaching the soil, not just `gap·La`). At
a dense warm midday column (`εf=0.97, εs=0.96, La=340`) this is **~40 W m⁻²**
(leaf ≈22, soil ≈17). But that is only an *upper-bound indicator*: it counts all
the `(1−ε)` absorbed without crediting the LW the conservative scheme newly
**transmits and reflects**. The physically meaningful quantity is the **net
spurious LW the near-black scheme injects into the column**,
`ΣALW_old − ΣALW_new = LW_out_new − LW_out_old`. Computed directly (legoESM
harness, §5 scheme vs near-black):

| column | `ΣALW_old − ΣALW_new` |
|---|---|
| dense (LAI 4, La 340) | +12.9 W m⁻² |
| moderate (LAI 3, La 320) | +13.0 |
| humid (LAI 4, La 380) | +13.4 |
| sparse (LAI 1, La 300) | +13.4 |

So the near-black scheme retains **~13 W m⁻²** of LW that the conservative scheme
reflects/emits back to the sky — robust across canopy density and `La`. (The ~40
W m⁻² indicator over-counts because most of the "over-absorbed" LW is, in the
conservative scheme, re-transmitted or reflected rather than retained.)

> Table state (for reproducibility): `εf=0.97, εs=0.96, β=½, SZA=30°, Ts=305 K,
> Tf_Sun=300 K, Tf_Sh=297 K`; `CI=0.8` (dense/humid), `0.75` (moderate), `0.7`
> (sparse). The leaf-only old−new difference (used in §3.2) is `~4–8 W m⁻²`
> (≈6.8 dense), the remainder landing on the soil.

### 3.2 Translating to a temperature bias — coupling matters

The naïve radiative-equilibrium estimate `ΔT = ΔALW/(4 E_eff σ T³) ≈ 3–4 K`
is a large **over**-estimate: the surface is not radiatively isolated. Restoring
strengths at a typical state:

| channel | strength |
|---|---|
| radiative `4 E_eff σ T³` | ~6 W m⁻² K⁻¹ |
| sensible `ρ c_p / R_b` (`R_b≈40 s/m`) | ~30 W m⁻² K⁻¹ |
| latent (stomata × `de_sat/dT`) | ~30–50 W m⁻² K⁻¹ |

The leaf is coupled ~10–15× more strongly than radiation alone. Of the
~13 W m⁻² net column injection (§3.1), the **leaf** share (`ALW_Sun+ALW_Sh`,
old−new) is only ~4–8 W m⁻² (≈6.8 dense); over the ~70–90 W m⁻² K⁻¹ total leaf
restoring that is a **leaf warm bias of ~0.1 K** (the rest of the injection lands
on the soil). The often-quoted `~0.15–0.2 K` is an **upper bound** that assigns the
*entire* column forcing to the leaf; the precise leaf/soil split and LST effect
need a coupled rerun. The **sign is robustly warm** (the exact column-flux
difference in §3.1 is positive across all cases). A column LST bias
up to ~1 K at sparse / high-`La` sites is *plausible* — the soil restoring (ground
heat flux `G` + weaker turbulent coupling) is softer than the leaf's — but it is
not established by this back-of-envelope and should be confirmed against tower
`LW_out` (§6). Treat ~0.2–1 K (warm), increasing with `(1−ε)`, `La`, and where the
soil controls LST, as an order-of-magnitude expectation, not a derived number.

### 3.3 Consistency with DifferBESS field evaluation

The DifferBESS development journal documents a model-wide LST **warm** bias of
~+1.5 K (FC config) that the `kB⁻¹=0` + NKH aerodynamic changes roughly halved to
~+0.7 K — i.e. the dominant lever was the *turbulent* side. The near-black LW
over-absorption is a **secondary, structural** contributor to the residual warm
bias, consistent in sign with the estimate above.

---

## 4. Reference comparison

| model | canopy structure | emissivity treatment | reflection | conservative at ε<1 |
|---|---|---|---|---|
| **Ryu (2011) / DifferBESS / legoESM** | two-big-leaf | `ε` on emission, **absorptivity 1** | **none** | **no** (`O(1−ε)`) |
| **CABLE** (`cbl_radiation.F90`) | two-big-leaf | `Cemleaf/Cemsoil` on emission, absorptivity 1 | **none** | **no** (same closure) |
| **CLM5** (Tech Note §2.4.2) | single big-leaf | geometric `ε_v=1−e^{−(L+S)/μ̄}`, **absorptivity = emissivity** | **yes** `(1−ε_v)(1−ε_g)` | **yes** |

**CABLE uses the same near-black, no-reflection closure** (verified): mapping its
`flws=Cemsoil·σ·Tss⁴ → Ls`, `flwv=Cemleaf·flpwb → Lf`, `emair=fld/flpwb` so
`(emair−Cemleaf)·flpwb = fld−flwv → La−Lf`, `transd=e^{−kd L}`, `transb=e^{−kb L}`,
CABLE's sunlit `qcan(:,1,3)` is exactly `W_sun_soil·(Ls−Lf) + W_sun_sky·(La−Lf)`.
It is **not** term-for-term identical, however: CABLE uses a *single* canopy
emission temperature (`flwv`), so its shaded `qcan(:,2,3)=(1−transd)(flws+fld−
2·flwv)−qcan(sun)` is the single-`Tf` `2·Lf` form — our scheme improves on CABLE
with per-class `Lf_Sun/Lf_Sh`. The *emissivity closure* (near-black, no reflection)
is identical; the temperature structure is not.

**CLM5 is the conservative reference**, but single-leaf: leaves are **black**, the
canopy emissivity is the *geometric* interception `ε_v=1−e^{−(L+S)/μ̄}` (the
analogue of our `W_tot`), Kirchhoff is explicit ("these equations assume that
absorptivity equals emissivity"), and the ground reflects via
`L_g↑ = (1−ε_g)L_v↓ + ε_g σT_g⁴`. The fix below ports the CLM5 emissivity
bookkeeping onto the two-leaf kernels and additionally retains a (small) **leaf**
reflectance, which CLM5 omits.

---

## 5. The fix: conservative two-leaf gray-body longwave (keeps εf, εs < 1)

Keep the **physical** leaf and ground emissivities. A surface that emits with
`εf<1` must, by Kirchhoff + energy conservation, also **reflect** `(1−εf)`
(leaf LW *transmittance* ≈ 0, so the intercepted-but-unabsorbed fraction is
reflected, roughly Lambertian — not forward-transmitted). So the consistent
`εf<1` scheme carries a canopy reflectance `r_c>0`, which closes a canopy↔ground
interreflection loop that we sum in closed form.

### 5.1 Physical model

For a diffuse LW stream passing through the canopy, split into absorbed /
back-scattered / transmitted with (`β` = backscatter fraction of leaf scatter;
`β=½` for isotropic):

```
a_c = εf·W_tot                              # absorbed   (Kirchhoff: absorptivity = εf)
r_c = β·(1−εf)·W_tot                         # back-scattered (canopy LW reflectance)
t_c = gap_LW + (1−β)·(1−εf)·W_tot            # transmitted (gap + forward-scatter)
a_c + r_c + t_c = εf·W_tot + (1−εf)·W_tot + gap_LW = W_tot + gap_LW = 1   ✓
```

Ground: absorbs `εs`, emits `εs·σTg⁴`, reflects `ρ ≡ 1−εs`. Per-class canopy
emission to the sky / ground hemispheres (detailed balance, as the current code
splits `Lf`), with `B_x ≡ σT_x⁴`:

```
S↑ = εf·(W_sun_sky·B_Sun  + W_sh_sky·B_Sh )      # canopy emission upward (to sky)
S↓ = εf·(W_sun_soil·B_Sun + W_sh_soil·B_Sh)      # canopy emission downward (to ground)
```

Limits: `β=0` ⇒ `r_c=0` (forward-scatter closure, omits the leaf backscatter —
§5.5); `εf=1` ⇒ `r_c=0` and `a_c=W_tot` (black leaves, CLM5 — no leaf scatter to
model).

### 5.2 The canopy↔ground interreflection: full geometric-series closure

The canopy (reflectance `r_c`) and the ground (reflectance `ρ`) face each other,
so a stream bounces between them indefinitely. Let

```
D  ≡ total downward LW incident on the ground (all passes)
U  ≡ total upward   LW leaving  the ground (all passes)   (≡ U_g)
S_d ≡ t_c·La + S↓     # primary downward source: sky transmitted + canopy down-emission
```

Two balance statements encode the infinite interreflection:

```
ground:  U = εs·B_g + ρ·D                      # emit + reflect the incident D
canopy:  D = S_d + r_c·U                        # primary source + canopy reflecting U back down
```

**Bounce-by-bounce (the geometric series).** Iterate from `U⁽⁰⁾ = 0`,
`U⁽ⁿ⁺¹⟩ = εs·B_g + ρ(S_d + r_c·U⁽ⁿ⁾)`. The `n`-th canopy↔ground round trip
attenuates by `(ρ·r_c)`, so

```
U = (εs·B_g + ρ·S_d)·[ 1 + (ρ r_c) + (ρ r_c)² + … ]
  = (εs·B_g + ρ·S_d)·Σ_{n≥0} (ρ r_c)ⁿ
```

Splitting the two sources, `(εs·B_g + ρ·S_d)·(ρ r_c)ⁿ = εs·B_g·(ρ r_c)ⁿ +
ρ·S_d·(ρ r_c)ⁿ`: the **primary-source** part `ρ·S_d·(ρ r_c)ⁿ` has reflected off
the ground `n+1` times and off the canopy `n` times; the **ground-emission** part
`εs·B_g·(ρ r_c)ⁿ` has `n` of each (at `n=0` it is the unreflected ground
emission). The series converges whenever `ρ·r_c < 1`, which holds for all physical
emissivities — `ρ·r_c = (1−εs)·β(1−εf)·W_tot ≤ 6.0×10⁻⁴` at `εf=0.97, εs=0.96,
β=½`, and `≤ 0.125` even at `εf=εs=0.5`; it degenerates to `1` (zero denominator)
only at the unphysical corner `εf=εs=0, β=1, W_tot=1`. It sums to the closed form

```
U_g = (εs·B_g + ρ·S_d) / (1 − ρ·r_c)
```

(identical to solving the two balance equations simultaneously:
`U = εs B_g + ρ(S_d + r_c U) ⇒ U(1−ρ r_c) = εs B_g + ρ S_d`). Then

```
D_g = S_d + r_c·U_g
```

The single denominator `1/(1−ρ r_c)` *is* the resummed infinite interreflection —
exact, no truncation. Setting `r_c=0` collapses it to one ground bounce
(`U_g = εs B_g + ρ S_d`, `D_g = S_d`); see §5.5.

### 5.3 Net fluxes, LW_out, and the LST observation operator

The canopy absorbs `a_c` of each stream incident on it — `La` from above (split by
the sky kernels) and the converged `U_g` from below (split by the soil kernels) —
and emits `εf·B` to both hemispheres:

```
ALW_Sun  = εf·[ W_sun_sky·La + W_sun_soil·U_g − (W_sun_sky + W_sun_soil)·B_Sun ]
ALW_Sh   = εf·[ W_sh_sky ·La + W_sh_soil ·U_g − (W_sh_sky  + W_sh_soil )·B_Sh  ]
ALW_Soil = εs·( D_g − B_g )
```

**Top-of-canopy upward LW** (escapes to the atmosphere): canopy emission to sky +
ground upwelling transmitted out + sky directly reflected off the canopy top:

```
LW_out = S↑ + t_c·U_g + r_c·La
```

**LST is a diagnostic, not a calibration target (§6).** Crucially, the *physical*
effective emissivity of this column is **not** an emission-weighted average. The
fraction of incident `La` reflected back to the sky is

```
R_col = ∂LW_out/∂La = r_c + ρ·t_c² / (1 − ρ·r_c) ,     ε_eff,phys = 1 − R_col
```

For a dense canopy `t_c` is small, so `R_col → r_c` and `ε_eff,phys ≈ 0.985`
(`0.985` at `εf=0.97, εs=0.96`; ~`0.980` at `εf=0.96` — **only weakly dependent on
`εf`**, geometry-dominated) — the vegetated column is **near-black from above**,
not `0.96–0.98`. Therefore, to compare with a tower/satellite
LST that was *retrieved* assuming some `ε_cal≈0.96–0.98`, the model must **replicate
the retrieval operator** on its own `LW_out`:

```
LST_model = ( (LW_out − (1 − ε_cal)·La) / (ε_cal·σ) )^{1/4}     # same ε_cal the data used
```

i.e. subtract the assumed-reflected `(1−ε_cal)·La` and divide by `ε_cal·σ`, exactly
as the data processing did, so model and obs are treated identically irrespective
of the model's true `ε_eff,phys`. (The old emission-weighted
`E_eff=εf·W_tot+εs·gap` is **not** the column emissivity and should not be used as
a column emissivity.)

### 5.4 Properties (proofs)

Note `a_c + r_c + t_c = 1`, hence `a_c = εf·W_tot = 1 − r_c − t_c`.

**(a) Exact at isothermal equilibrium, any `εf, εs, β` with `ρr_c<1`.** `Tf_Sun=Tf_Sh=Tg=T`,
`La=σT⁴=B`: `S_d = t_c·B + εf·W_tot·B = (t_c + a_c)·B = (1−r_c)·B`, so
`U_g = (εs B + ρ(1−r_c)B)/(1−ρ r_c) = B·(εs + (1−εs)(1−r_c))/(1−(1−εs)r_c) = B`
(numerator `= εs + (1−εs) − (1−εs)r_c = 1 − ρ r_c`). Then `D_g = S_d + r_c B =
(1−r_c)B + r_c B = B`. Hence `ALW_Sun = εf[(W_sun_sky+W_sun_soil)B −
(W_sun_sky+W_sun_soil)B] = 0`, `ALW_Sh=0`, `ALW_Soil = εs(B−B)=0`, and
`LW_out = a_c B + t_c B + r_c B = B`. **All net fluxes vanish; the column radiates
as a blackbody at `T`.** ∎

**(b) Globally energy-conserving, all temperatures.** Required:
`ΣALW = La − LW_out`. Using `a_c = εf·W_tot`, `a_c+r_c+t_c=1`,
`D_g = S_d + r_c·U_g`, `U_g = εs·B_g + ρ·D_g`, the sum telescopes to
`ΣALW − (La − LW_out) = D_g − t_c·La − S↓ − r_c·U_g = D_g − S_d − r_c·U_g = 0`
(full algebra in the appendix). **Exact for any temperatures** — the
`1/(1−ρ r_c)` resummation carries all bounces, so there is no truncation error. ∎

**(c) Limits.** `r_c→0` (i.e. `β→0`) gives the single-ground-bounce form
`U_g=εs B_g+ρ S_d`. `εs=1` gives `U_g=B_g` (no ground reflection). At `εf=εs=1`
(and hence `r_c=0`, `t_c=gap_LW`) it is identical to today's near-black scheme
`W_sun_soil(B_g−B_Sun)+W_sun_sky(La−B_Sun)`, so the `ε=1` blackbody test is
preserved. ∎

**(d) CLM5 single-leaf limit.** Set `εf=1` (so `r_c=0`, `a_c=W_tot`,
`t_c=gap_LW`) and collapse sunlit+shaded to one canopy temperature
(`B_Sun=B_Sh=B_v`, `W_*→W_tot`): recovers CLM5's geometric `ε_v=W_tot`,
`L_g↑=(1−ε_g)L_v↓+ε_g B_g`, and the net forms. (For `εf<1` the scheme is *more*
general than CLM5 — it adds a leaf reflectance CLM5 omits.) ∎

**(e) Differentiable.** Only linear ops and one division by `1−ρ r_c` (bounded
away from 0); no new branches; the `kb=kd` removable singularity stays handled by
the `exprel/expm1` kernel form, so `jax.grad` (incl. wrt SZA) is finite. ∎

### 5.5 The leaf-scatter coefficient `β` (and the `r_c=0` / black-leaf options)

`β` sets how the intercepted-but-unabsorbed `(1−εf)·W_tot` splits between
backscatter (`r_c`) and forward-transmission. Three defensible choices:

- **`β = ½` (isotropic, recommended for `εf<1`).** Leaf LW reflectance is roughly
  Lambertian, so half the scattered flux returns. `r_c = ½(1−εf)W_tot ≈ 0.012`.
  The first-order canopy reflection it adds is `~r_c·La ≈ 4 W m⁻²` at dense canopy
  (`r_c` already contains `W_tot`) — comparable to the ground-reflection term, so
  worth keeping for a calibration-grade scheme. Cost: one extra division (`1/(1−ρ r_c)`); the
  multiple-bounce correction beyond first order is `ρ r_c ≤ 6.0×10⁻⁴` (≈5.5–5.7×10⁻⁴ for the dense examples; negligible
  but summed exactly anyway).
- **`β = 0` (`r_c=0`, forward-scatter).** Simplest: the loop opens, one ground
  bounce, `U_g=εs B_g+ρ S_d`. Still **exactly conservative**, but it routes the
  `~4–5 W m⁻²` of unabsorbed-intercepted LW *forward* (transmits it) instead of
  reflecting ~half of it back — i.e. it **omits the first-order leaf backscatter**.
  Acceptable only if that ~4–5 W m⁻² is below the calibration's discriminating
  power.
- **`εf=1` (black leaves, CLM5).** `(1−εf)=0`, so no leaf scatter at all and
  `r_c=0` exactly. Simplest and standard, but discards the physical leaf
  emissivity (≈3% emission). Viable because the column is near-black anyway (§5.3);
  pick this if minimal/standard is preferred over physical `εf`.

This document adopts **`εf<1` with `β=½` (`r_c>0`)** as the recommended scheme.

---

## 6. Calibration: target `LW_out`, not LST

The "observed LST" used for calibration is itself *derived* from tower `LW_out` by
assuming an effective surface emissivity `ε_cal≈0.96–0.98` and inverting. This
inversion is the only place the `0.96–0.98` number enters — it is a **retrieval
convention**, not the model's physical column emissivity (which is `≈0.985` for
dense canopy, weakly `εf`-dependent, §5.3).
Two consequences:

1. **Don't try to make an emission-weighted `E_eff` equal `ε_cal`.** The physical
   column emissivity `1−R_col` is geometry-dominated (≈0.985 for dense canopy at
   `εf=0.97, εs=0.96`, only weakly `εf`-dependent); matching `ε_cal` to an emission
   weighting is a category error. If you must produce a model LST, replicate the
   retrieval operator on the model `LW_out` with the **same `ε_cal`** the data used
   (§5.3); the residual emissivity-convention offset (using `ε_cal=0.97` to invert a
   column that is physically ≈0.985) is small — **~0.15–0.40 K** at 300 K
   (`+0.40 K` at `La=300`, `+0.15 K` at `La=400`) — and applies equally to model
   and obs, so it cancels in the comparison.

2. **Prefer calibrating `LW_out` directly** — measured by the 4-component
   radiometer at every EC site, no emissivity inversion. It is the matched pair for
   the conservative scheme:
   - the conservative `LW_out` (§5.3) includes the reflected `(1−εs)·LW_in` (and
     the small leaf-reflected `r_c·LW_in`);
   - the **near-black** scheme has no reflected term ⇒ its `LW_out` is too low by
     `~(1−ε_eff)·LW_in`, which it can only hide by **warming** the surface to emit
     the deficit — the warm-bias mechanism of §3. So an `LW_out` target *exposes*
     the defect; an LST target with a matched `ε` convention partly hides it.
   - `LW_out` is the actual land→atmosphere coupling flux (LST is only a
     diagnostic) and is hemispheric — a better match to a 1-D column than a
     *directional* satellite LST.
   - Calibrate jointly with `H, LE, GPP` (same towers) so the energy *partition*,
     and hence `T_surf`, is pinned — `LW_out` alone fixes the radiative sum, not the
     partition. Keep LST as cross-validation only.
   - Caveat: drive the model with the tower `LW_in` (the reflected term depends on
     it); EC turbulent-flux closure (~10–20%) does not affect the radiometer
     `LW_out`.

With `LW_out` as the target, `εf, εs` are **physical** parameters (surface-estimate
prior), not a retrieval convention, and the conservative scheme is required.

---

## 7. Implementation plan (DifferBESS first)

1. **`process/CanopyLongwaveRadiation.py`** — replace the coupled-branch `ALW_*`
   block (lines ~131–163) with §5.1–5.3:
   ```
   W_tot = 1 - gap_LW
   a_c = epsf * W_tot
   r_c = 0.5 * (1 - epsf) * W_tot                 # beta = 1/2 (isotropic)
   t_c = gap_LW + 0.5 * (1 - epsf) * W_tot         # = 1 - a_c - r_c
   rho = 1 - epss
   S_up   = epsf * (W_sun_sky*B_Sun  + W_sh_sky*B_Sh)
   S_down = epsf * (W_sun_soil*B_Sun + W_sh_soil*B_Sh)
   S_d = t_c*La + S_down
   U_g = (epss*B_g + rho*S_d) / (1 - rho*r_c)      # resummed interreflection
   D_g = S_d + r_c*U_g
   ALW_Sun = epsf*(W_sun_sky*La + W_sun_soil*U_g - (W_sun_sky+W_sun_soil)*B_Sun)
   ALW_Sh  = epsf*(W_sh_sky *La + W_sh_soil *U_g - (W_sh_sky +W_sh_soil )*B_Sh )
   ALW_Soil = epss*(D_g - B_g)
   LW_out = S_up + t_c*U_g + r_c*La
   ```
   Keep `εf, εs ≈ 0.97/0.96`. Update the docstring (remove the near-black caveat).
2. **Calibration** — switch the target to **`LW_out`** (§6), with `H, LE, GPP`;
   treat `εf, εs` as physical params (`0.96–0.98` prior). **Re-calibrate** — the
   near-black bias is baked into the current fit (§3.3), so any conservative scheme
   shifts `LW_out`/T and needs a re-fit.
3. **LST cross-check** — replicate the retrieval operator on `LW_out` with the
   calibration `ε_cal` (§5.3); do not calibrate on it.
4. **`decouple_soil_lw`** (VEG_ONLY) — explicitly remove the soil-origin
   absorption (the `W_*_soil·U_g` terms) so the canopy exchanges LW only with the
   sky: `ALW_i = εf·(W_i_sky·La − (W_i_sky+W_i_soil)·B_i)`. Setting `ρ=0` alone is
   **not** sufficient — leaves would still receive `U_g = εs·B_g` (ground emission)
   through the `W_*_soil·U_g` path.
5. **Tests** (non-vacuous): isothermal conservation **at εf=0.97, εs=0.96** (`ALW_*
   =0` to round-off — the current scheme fails this); global closure `ΣALW == La −
   LW_out`; `εf=εs=1` regression == current scheme; `r_c→0`/`εf→1` limits;
   flux-tower `LW_out`/LST regression.
6. **Sync to legoESM** once validated — port the block into
   `radiative_transfer.py` (`canopy_longwave_rt`), keep the per-field / `exprel` /
   conductance machinery, update the docstring, add the conservation test to
   `tests/land/unit/test_canopy_rt.py`.

---

## 8. Decision record

- The near-black scheme is a **faithful port of DifferBESS/CABLE** (same emissivity
  closure), correct at `ε=1`; the `O(1−ε)` non-conservation is shared by the whole
  two-leaf lineage (Ryu 2011 origin) — **not** a defect introduced by the legoESM
  sync.
- The §5 fix keeps **physical `εf, εs ≈ 0.96–0.98`** with **`r_c>0` leaf backscatter
  (`β=½`)** + ground reflection, summed by the closed-form `1/(1−ρ r_c)`
  interreflection closure. It is **provably, exactly conservative** (isothermal +
  global), reduces to the current scheme at `εf=εs=1` and to CLM5 at `εf=1`, and is
  differentiable.
- The model column is physically **near-black** (`ε_eff≈0.985`, weakly
  `εf`-dependent); LST comparison must
  replicate the tower retrieval operator on `LW_out`, and the **recommended
  calibration target is `LW_out`** (§6), not LST.
- It **changes answers** (adds reflection + `εf` absorptivity) ⇒ a
  **re-calibration-gated** upgrade.
- Plan: **fix + re-calibrate + validate in DifferBESS**, then sync to legoESM.
  Until then legoESM keeps the documented DifferBESS-faithful near-black scheme.

---

## Appendix — global conservation algebra (proof of §5.4b)

Let `S_up, S_down` as in §5.1, `a_c=εf·W_tot`, `a_c+r_c+t_c=1`,
`S_d=t_c·La+S_down`, `D_g=S_d+r_c·U_g`, `U_g=εs·B_g+ρ·D_g`, `ρ=1−εs`. The
per-class sums use `W_sun_sky+W_sh_sky = W_sun_soil+W_sh_soil = W_tot`, so
`ALW_Sun+ALW_Sh = a_c·La + a_c·U_g − (S_up + S_down)`. Then

```
ΣALW   = a_c·La + a_c·U_g − S_up − S_down + εs·D_g − εs·B_g
LW_out = S_up + t_c·U_g + r_c·La
```
Form `ΣALW − (La − LW_out)`:
```
= a_c·La + a_c·U_g − S_up − S_down + εs·D_g − εs·B_g − La + S_up + t_c·U_g + r_c·La
= (a_c + r_c − 1)·La + (a_c + t_c)·U_g − S_down + εs·D_g − εs·B_g
```
Use `a_c+r_c−1 = −t_c` and `a_c+t_c = 1−r_c`:
```
= −t_c·La + (1−r_c)·U_g − S_down + εs·D_g − εs·B_g
```
Substitute `εs·B_g = U_g − ρ·D_g` (from the ground balance) ⇒ `−εs·B_g = −U_g + ρ·D_g`:
```
= −t_c·La + (1−r_c)U_g − U_g − S_down + εs·D_g + ρ·D_g
= −t_c·La − r_c·U_g − S_down + (εs+ρ)·D_g
= −t_c·La − r_c·U_g − S_down + D_g          (εs+ρ = 1)
= D_g − (t_c·La + S_down) − r_c·U_g
= D_g − S_d − r_c·U_g = 0                    (D_g = S_d + r_c·U_g)   ∎
```

---

# Part II — Penman-Monteith saturation second derivative (`d²eₛ/dT²`) + curve consistency

**Separate from the longwave issue.** Bundled for the same DifferBESS fix pass.
Already fixed in legoESM (`energy_balance.py`, `canopy_met_variables`); DifferBESS
still carries it (`process/CanopyEnergyBalance.py:349-350`).

## II.1 Where it is — and a cross-module curve inconsistency

The Penman-Monteith leaf energy balance solves a quadratic in `LE` whose
coefficients depend on the first and second derivatives of the **saturation**
vapour pressure, `desTc=deₛ/dT` and `ddesTc=d²eₛ/dT²`. DifferBESS:

```python
es_c   = REF_SAT_VP * exp(17.67*(Tc−K) / ((Tc−K) + 243.5))            # MAGNUS  17.67 / 243.5
desTc  = es_c * 4098 * ((Tc−K) + 237.3)**(-2)                          # ← FAO-TETENS 4098 / 237.3
ddesTc = 4098 * ( desTc*((Tc−K)+237.3)**(-2) + (-2)*e_c*((Tc−K)+237.3)**(-3) )  # ← Tetens AND e_c
```

Three issues, in increasing severity:

1. **Curve mismatch within the routine.** `es_c` uses **Magnus** (`17.67/243.5`)
   but the derivatives use **FAO-Tetens** coefficients (`4098 ≈ 17.27·237.3 = 4098.17`,
   offset `237.3`). The derivative does not match the curve it differentiates.

2. **Cross-module curve inconsistency (surfaced by the preprocessing audit).** The
   forcing/preprocessing uses a *different* curve again — **Tetens**
   `0.6108·exp(17.27·T/(T+237.3))` in `process/Meteorology.py:24` and
   `notebooks/legacy/preprocessing/preprocessing_util.py:57` (and the dew-point
   inversion there uses the matching Tetens inverse). The preprocessing computes
   **no** `d²eₛ/dT²` — so the second-derivative bug lives only at runtime in
   `CanopyEnergyBalance.py` — but the model as a whole mixes **two saturation
   curves**: Tetens for the forcing VPD/Td, Magnus for the canopy `es_c`, and
   Tetens coefficients for the canopy derivative. (Note `4098 ≈ 17.27·237.3 = 4098.17`, so
   the existing derivative is internally Tetens-consistent — it is simply applied
   to a Magnus `es_c`.)

3. **`e_c` instead of `eₛ` (the serious one).** `d²eₛ/dT²` is a property of the
   **saturation** curve and must use `es_c`. DifferBESS uses `e_c` (the *actual*
   vapour pressure) in the second (negative) term. Since `e_c = RH·es_c`, that term
   is `~RH×` too small — ~40% low at `RH=0.6`. Its sign is negative, so the
   *complete* `ddesTc` ends up **~5% too high near 25 °C** (the positive first term
   dominates; the under-weighted negative term fails to subtract enough), and the
   error is spuriously humidity-dependent. The PM quadratic coefficients (`a, b, c`)
   contain `ddesTc` contributions (they are not wholly proportional to it), so the
   net effect on `LE`/leaf-T in the PM path is a modest, humidity-dependent bias.

## II.2 Correct derivation (one curve — Magnus)

With `eₛ = A·exp(B·Tc/(Tc+C))` (`Tc` in °C):

```
deₛ/dT   = eₛ · B·C / (Tc + C)²                       = eₛ · (BC) · (Tc+C)^(-2)
d²eₛ/dT² = (BC) · [ (deₛ/dT)·(Tc+C)^(-2)  −  2·eₛ·(Tc+C)^(-3) ]
```

For **Magnus** `B=17.67, C=243.5`: `BC = 17.67·243.5 = 4302.645` (note: **not**
4098), offset `C = 243.5` (**not** 237.3). `d²eₛ/dT²` uses **`eₛ`** throughout.

## II.3 The fix — Magnus everywhere

Resolution (confirmed): standardize the **whole DifferBESS pipeline on Magnus
`17.67/243.5`** — slightly more accurate than Tetens (Bolton 1980 ~0.2–0.3 % max
error over −35…+35 °C; Tetens `17.27/237.3` is ~0.1 % over 0–35 °C but degrades to
~2 % low by −35 °C) and **identical to legoESM's shared
`thermo.saturation_vapor_pressure`**, so the eventual sync needs no curve
conversion.

```python
_MAGNUS_B    = 17.67
_MAGNUS_C    = 243.5
_DESDT_COEFF = _MAGNUS_B * _MAGNUS_C          # = 4302.645  (B·C)

es_c   = saturation_vapor_pressure(Tc)        # shared Magnus curve, 17.67 / 243.5
desTc  = es_c * _DESDT_COEFF * (TcC + _MAGNUS_C) ** (-2)                         # deₛ/dT  [Pa K-1]
ddesTc = _DESDT_COEFF * ( desTc * (TcC + _MAGNUS_C) ** (-2)
                          + (-2.0) * es_c * (TcC + _MAGNUS_C) ** (-3) )          # d²eₛ/dT² [Pa K-2]
```
(`TcC = Tc − 273.15`.) Changes vs DifferBESS: prefactor `4098 → 4302.645`, offset
`237.3 → 243.5`, and `e_c → es_c`.

## II.4 DifferBESS edits + verification

- **`process/CanopyEnergyBalance.py:349-350`** → §II.3 form (define
  `_MAGNUS_B/_C/_DESDT_COEFF` once; reuse the same `17.67/243.5` `es_c` already
  uses).
- **Forcing curve** — switch `process/Meteorology.py:24` and
  `notebooks/legacy/preprocessing/preprocessing_util.py:57` (and the dew-point
  inverse) from Tetens `17.27/237.3` to Magnus `17.67/243.5`, so forcing VPD/Td and
  the canopy `es_c` share one curve. This shifts forcing VPD/Td by ~0.1–0.5 %,
  absorbed by the re-calibration; it is the change that makes the model
  curve-consistent end-to-end.
- **Verify**: central-difference `eₛ(T)` for `deₛ/dT`, `d²eₛ/dT²` vs `desTc/ddesTc`
  (legoESM matches to ~2e-10); plus an explicit `e_c`-independence check (`ddesTc`
  invariant to RH at fixed `T`).
- This is a correctness fix to the **PM** path; it does not change the **BT** path
  (the DifferBESS/legoESM default `LE_module="BT"` does not use `ddesTc`), so the
  production answer only moves where PM is selected (and wherever the forcing curve
  is touched).
