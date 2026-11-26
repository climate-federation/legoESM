Verdict: **NOT CLEAN**.

1. **P2 — Incorrect gradient at the fcansno ramp join**, [interception.py:406](/glade/u/home/pg2328/legoesm/.claude/worktrees/canopy-snow/packages/land/legoesm/land/canopy/interception.py:406). At `snocan/cap == fcansno_ramp`, the power branch evaluates `maximum(base, x0)` at equality; JAX assigns its derivative ½. With default widths and capacity 1, autodiff returns **26.611004**, while the analytic slope and finite differences give **53.222008**. Guard the inactive power argument with `where` and test the join explicitly; current gradient tests miss this boundary.
