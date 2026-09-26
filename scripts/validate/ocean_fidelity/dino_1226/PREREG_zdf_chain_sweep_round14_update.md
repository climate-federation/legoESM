# ZDF chain sweep round 14 update — remaining etau operand walk

Date frozen: 2026-08-28, after the first committed literal-EXP rerun and
before measuring these substitutions.

The literal ordinary-range glibc vector EXP reduced the row-18 composite from
173 to 21 failing columns but did not clear it.  Keep row 18 `DIVERGED` and
walk the remaining expression at `zdftke.F90:590-591` in source order.

Score, at the unchanged `1e-15` per-column bar and on all four focus columns:

1. literal EXP evaluated on the dumped NEMO argument (EXP ownership check);
2. literal EXP evaluated on legoESM's production argument (argument exposure);
3. exponent arguments using (a) the legacy `deg2rad`, (b) NEMO's literal
   `(rpi/180)*gphit` association, and (c) the grid's raw T-latitude radians;
4. dumped NEMO EXP substituted into the left-associated multiplication/add
   chain, one factor at a time: `rn_efr`, surface `en(1)`, EXP, ice fraction,
   masks, then addition to `en`.

The first substitution that clears 0/9,920 owns the remaining divergence.
If no small source-order construction clears it, stop at row 18 with the
first still-divergent operand and a next-round design; do not advance to row
19.  Controls and instrument brackets remain mandatory.
