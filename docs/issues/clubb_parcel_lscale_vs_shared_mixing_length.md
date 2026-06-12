# CLUBB parcel Lscale could supplant `_shared.mixing_length`

**Status:** open (enhancement idea). **Origin:** CLUBB single-file
condensation, iter C5 (2026-06-12; PORT_CLUBB.md goal item).

`clubb.py` section "Parcel buoyant-sorting mixing length" implements CLUBB's
nonlocal parcel buoyant-sorting `Lscale` (`mixing_length.F90`, golden-locked
vs CLUBB-JAX): an entraining parcel ascends/descends until its buoyancy is
exhausted; `Lscale_up`/`Lscale_down` are combined geometrically.

`atmosphere/physics/_shared.py:mixing_length` (used by `clubb_lite` and
others) is a simple Blackadar-style local length. The two were compared
during condensation and CONFIRMED different numerics, so the parcel version
stays inside `clubb.py` per the one-file-per-scheme convention.

**Idea:** the parcel buoyant-sorting length is physically richer (nonlocal,
stability-aware) and could eventually supplant the `_shared` Blackadar length
for other schemes. That would mean promoting it OUT of `clubb.py` into
`_shared` (or a dedicated shared module), wiring `clubb_lite`/others to it
behind a config switch, and re-validating those schemes (it is NOT a drop-in:
different magnitudes/profiles, hence different tuned coefficients).
