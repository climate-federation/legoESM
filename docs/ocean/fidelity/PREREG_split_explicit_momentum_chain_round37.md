# Preregistration: scoped pre-dyn_zdf momentum RHS accumulator, round 37

Date: 2026-08-30. Frozen before execution. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Scope and ordering

Round 36 exonerates wind placement and leaves the no-wind RHS first. Reuse
NEMO's cumulative DINO streams in `stpmlf.F90` execution order:

1. `D03`, after `dyn_adv` (`stpmlf.F90:265-270`);
2. `D04-D03`, `dyn_vor` (`:271-274`);
3. `D05-D04`, `dyn_ldf` (`:275-278`);
4. `D06-D05`, `dyn_hpg` (`:280-284`);
5. exact bundle `dyn_adv+dyn_hpg`, because NEMO attaches the KE gradient to
   `dyn_adv` while `MomentumTendencyDiagnostics` attaches it to `KE_PGF`;
6. the D06 four-term sum and its reconstruction of round-36 no-wind RHS.

Capture both U and V from the production public momentum diagnostics at the
matched day-180 Nnn/Nbb time levels. Individual adv/hpg rows are diagnostic
partials and cannot own a difference. The exact ownership ladder is the
adv+hpg group, vorticity/Coriolis, lateral friction, then the D06 sum, with
NEMO source order retained in the table.

## Bars and dispositions

Every 3-D tendency and exact group uses the ACCUMULATING `1e-12` class bar.
The first failing exact row owns the no-wind RHS interval. If all exact rows
pass but D06 fails, disposition is `OPEN_COMPOSITION`; if D06 passes but the
round-36 no-wind RHS fails, the remaining split-explicit removal/bottom-drag
composition is the owner. Partial adv/hpg failures alone never stop or own.

## Admission and controls

Bind round-36 SHA, entry/before restart, D03--D06 U/V streams, mesh, active
NEMO `stpmlf.F90` and momentum sources, production diagnostics modules,
scorer, and this registration. Require tracked-clean checkout, checkout-first
`PYTHONPATH`, CPU/fp64, session ID, exact cumulative differences, public
diagnostic closure, declared U/V staggering, and exact D06 reconstruction.
One-cell roll, sign reversal, wet NaN, and `2x` bar plants must fire for both
components. The failed legacy whole-ZDF top/bottom control is explicitly
unreachable from this scoped measurement and is not weakened or copied. No
new NEMO-writer SLOT is allocated because all required NEMO streams already
exist. The held twin-capture SLOT cascade is specified by the pre-run
amendment below.

## Pre-run implementation amendment: held twin capture and exact order

Frozen before any round-37 capture. Enumeration found that the retained
deterministic-writer stack already contains D03--D06 for both U and V, plus
the dedicated `keg`, `zad`, `vor`, `ldf`, and `hpg` increment streams. No new
NEMO writer is justified. The held cascade therefore builds and runs a
deterministic CPU twin-diagnostic producer twice, brackets its byte-identical
captures, then scores against those existing full-halo streams.

The earlier shorthand placing the `adv+hpg` group first is superseded by this
source-ordered exact ladder: (1) vertical-advection/ZAD, the separately
comparable part of `dyn_adv`; (2) vorticity/Coriolis; (3) lateral friction;
(4) the KE-gradient+HPG group at the `dyn_hpg` closure point; (5) D06 total.
The individual KEG and HPG partitions remain diagnostic partials. This is a
clarification of the registered bundle constraint, not a changed bar.

The active instrumented-source citations carried into every receipt are:

| exact row | NEMO source |
|---|---|
| vertical advection / ZAD | `stpmlf.F90:309-314`; `dynadv.F90:97-103` |
| vorticity / Coriolis | `stpmlf.F90:315-318`; `dynvor.F90:143-179` |
| lateral friction | `stpmlf.F90:319-322`; `dynldf.F90:79-115` |
| KE-gradient + HPG | `dynadv.F90:89-96`; `dynhpg.F90:117-133`; `stpmlf.F90:309-328` |
| D06 total | `stpmlf.F90:269-270,309-328` |

## Coverage amendment after the stopped first capture

The first held capture stopped before emitting metadata, exactly as its
closure gate required. Its retained binary arrays are diagnostic evidence,
not an admitted round-37 score: `total - diagnostic_sum` is nonzero only in
the U surface level (9,793 values, maximum `1.9218384941372795e-5 m s-2`),
while every deeper U level and all V values are exactly zero. This is the
signature of DINO's zonal-only explicit surface stress, already independently
established by round 36's exact-zero V wind control.

The active NEMO D03--D06 writer audit finds no omitted call:

| interval | call that can write momentum RHS | active DINO disposition |
|---|---|---|
| zero to D03 | `dyn_dmp` (`stpmlf.F90:292`), `dyn_asm_inc` (`:294-295`), `bdy_dyn3d_dmp` (`:297`), AGRIF sponge (`:302-303`) | inactive (`ln_dyndmp=F`, `ln_dyninc=F`, `ln_bdy=F`, no AGRIF) |
| D03 | `dyn_adv` (`stpmlf.F90:309-314`) -> KEG then ZAD (`dynadv.F90:86-103`) | active; dedicated KEG+ZAD identity |
| D03 to D04 | `dyn_vor` (`stpmlf.F90:315-318`; `dynvor.F90:143-179`) | active; planetary+relative EEN contribution |
| D04 to D05 | `dyn_ldf` (`stpmlf.F90:319-322`; `dynldf.F90:79-120`) | active |
| D05 to D06 | `dyn_osm` (`stpmlf.F90:323`), then `dyn_hpg` (`:324-328`; `dynhpg.F90:117-133`) | OSM inactive (`ln_zdfosm=F`); HPG active |
| after D06 | `dyn_spg` (`stpmlf.F90:332`), then surface stress inside `dyn_zdf` (`dynzdf.F90:353-363`) | outside the D03--D06 accumulator |

Thus neither a separate Coriolis metric term, an SSH-gradient split, nor
trend bookkeeping is missing from D03--D06. The missing public diagnostic is
`surface_stress_u/v`, applied by the twin after the enumerated slow terms and
by NEMO later inside `dyn_zdf`. Round 37 adds it as a coverage-only,
non-owning row; it is excluded from `mapped_d06` and remains disposed by
round 36 (`WIND_PLACEMENT_REFUTED`). The corrected closure bar is unchanged:
all named public components, now including surface stress, must reconstruct
`diagnostics.total_u/v` to `1e-15`, and the DINO V stress must remain exactly
zero. The exact D03--D06 ownership ladder and its `1e-12` bars do not change.
