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
SLOT is allocated: all NEMO streams already exist; only a CPU scorer is
required.
