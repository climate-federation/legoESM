# Gaps 7 and 8: selectable GM/Redi coefficient machinery

Uncommitted capability addition, not climate gap closure. User requires no
commits, no default or production-card changes. This scope precedes unit
verification; no measurement lane or integration is run and no new climate
numbers are citable. No claim about job 9701531's executable.

## Inventory before edits

GM already offers constant, Visbeck, Treguier (generic and native NEMO slopes),
and runtime EKE/GEOMETRIC fields, with optional Hallberg resolution scaling.
The OMIP tripole recipe selects constant GM=Redi=600 m2/s, Visbeck/Treguier
both off, centered slopes. The runner exposes Treguier with cap 900 but a
DELIBERATE non-NEMO floor of 200; the recipe floor is zero. Thus gap 7's
coefficient-law selection already exists: `--gm-treguier --gm-kappa-min 0`.
Both Treguier implementations ALREADY multiply by min(1,abs(f/f20)). This
vanishes at the equator BEFORE the optional floor. Tropical taper CONFIRMED.

Redi already supports constant diffusivity, static cosine/metric scaling
(nn_aht_ijk_t=20), and runtime EKE/GEOMETRIC overrides. GM selection alone
leaves Redi constant. No nn_aht_ijk_t=21 law or OMIP selector existed.
Existing native tensor/MSC/K33 consumers already accept distinct east/north
face diffusivities. Tripole grid.f_v averages T-point Coriolis and is NOT
NEMO's ff_f; it cannot be substituted in the tropical enhancement.
Search covered lateral mixing, the model step, recipes, CORE-II CLI/build
and their tests. No automated gate certifies inventory completeness.

## Available oracle and executed source arms

Read NEMO **5.0.1** at
`/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1`.
`/burg-archive/glab/users/pg2328/nemo_orca1/ORCA1/EXPREF/namelist_cfg:324-343`
selects nn_aht_ijk_t=21, ln_ldfeiv=T, nn_aei_ijk_t=21 and both velocity/length
pairs 0.018 m/s and 100 km. `src/OCE/LDF/ldftra.F90:290-293,548-551`
forms the Laplacian scales with the factor 1/2, hence aht0=aei0=900 m2/s.
`:408-410,419-425` reuses the GM faces in the Redi branch; the alternative
standalone call to ldf_eiv(aht0) is NOT this selected arm.
`:426-435` applies max(0.2*aht0,GM_face) plus
(1-min(1,abs(f/f20)))*(aht0-0.2*aht0). The u expression uses ff_t,
the v expression ff_f (ORCA is lat/lon in the tropical band).
`:437-440` broadcasts down and masks every depth. `:700-708` confirms GM's
prior tropical DECREASE, cap, then masked face averaging. Redi enhancement
therefore remains reachable where GM vanishes. No final Redi cap is present:
do not invent one or apply the floor before face averaging.
GitHub #1455 could not be read (api.github.com connection failed); no posting.
These are source/namelist observations, not proof of a running binary's card.

## Added versus reused

Reuse the Treguier kernels, native slope diagnosis, GM face averaging, explicit
Redi tensor and shared a33/MSC split. Extract native GM diagnosis to share it
between explicit Redi and implicit K33 without duplicating coefficient physics.
Add selectable `redi_coefficient=constant|nemo21`, default constant; its
separate aht0 config defaults to the source-resolved ORCA1 scale. NEMO's fixed
floor fraction and taper latitude are documented module constants.

CORE-II adds `--redi-coefficient`, `--redi-aht0` and
`--gm-slope-positions`, all unset by default. Example opt-in:

```
--gm-treguier --gm-kappa-min 0 --gm-slope-scheme nemo_iso_lap \
--gm-slope-positions nemo_native --redi-coefficient nemo21
```

The law requires native slopes and the native rotated-Laplacian consumers;
it does not silently change slope schemes. Read exact gphif from the selected
mesh and use the run's rotation rate to populate the dedicated ff_f field.
The implicit consumer uses the same native GM diagnosis, including the
coefficient's eta time level, carried N2/e3w, EOS and rotation rate.
Unknown selections, inactive knobs, unavailable mesh metadata, other closures,
nonzero GM floor, competing overrides and multi-device use raise.

## Sign contracts and tests planned before execution

Coefficients have m2/s units and are nonnegative. Coriolis is signed;
ABS makes tropical enhancement symmetric between hemispheres. Native NEMO
slopes have the opposite sign to generic -grad(rho)/drho_dz slopes; the
existing native branch passes them directly, without an added negation.
Coefficient diagnosis squares slopes; reversing slope signs cannot change it.
Redi is down-gradient diffusion (the code returns divergence of positive
K*gradient); GM remains skew/adiabatic transport. K33 is nonnegative and the
same face fields must feed explicit tensor and implicit MSC/K33 terms.

CPU tests will check floor/enhancement at distinct stagger locations, equator,
midlatitude, nonzero/zero slopes and stratification, dry masks, cap, JIT/AD,
actual production dispatch/forwarding and fail-fast guards. Nondegenerate
fixtures must distinguish averaging before vs after the floor and separate
u/v enhancement. Mutation controls must detect reversion of each new behavior.

## Exact boundary: what this is NOT

Not NEMO 5.0.2 equivalence, an integration result, full ORCA operator fidelity,
or a climate/stability verdict. No default/card change, no commit. The new
Redi law only supports the already-implemented native NEMO tensor path with
Treguier enabled: NEMO's GM-disabled standalone ldf_eiv(aht0) branch is omitted.
No centered/triad/MPAS/cube Redi21, distributed ff_f sharding, or distributed
validation. Existing slope/N2/MLD/geometry/time-stepping conventions are not
promoted to oracle fidelity by this coefficient selection. Existing global
native operator boundary/fold behavior is not certified here. Bolus FCT,
face-average selection, MSC, native N2 and remaining operator choices retain
existing defaults; selecting the coefficient law is not a full ORCA card.
No new prognostic state. No standalone climate numbers or job-runtime claims.

## Validation receipts and default audit

Pending CPU tests and independent review. No independent approval claimed.
