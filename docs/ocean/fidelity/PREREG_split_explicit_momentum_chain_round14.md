# Preregistration: post-metric row-1 recurrence, round 14

Date: 2026-08-29. Frozen before measurement.

## Question

Round 13 put every first-substep continuity operand 9.1--9.7 at its frozen
bar on the production path and released chain row 1.4.  This round asks whether
the complete split-explicit recurrence now carries that closure through NEMO's
final `puu_b`, `pvv_b`, `pssh`, `un_adv`, and `vn_adv` outputs.

The measurement reuses the committed
`split_explicit_momentum_chain_round6.py` wrapper unchanged.  It runs no NEMO
process and creates no new dump.  Its `forcing_only` arm substitutes the already
owned NEMO slow-momentum forcing at the production solver entry; the production
bridge supplies the literal BEFORE seed.  Its `forcing_and_seed` arm remains a
targeting discriminator and cannot promote a row across a failed literal seed.

## Frozen order and bars

The round-13 artifact must have SHA-256
`7224942cb8dbfa8522c6b61c3ee14a061f0372e46fc9ea00e59b3e30956aef68`, report
`ROW_1.3_ALL_OPERANDS_AT_BAR_RELEASE_1.4`, and have all rows 9.1--9.7 at bar.
The existing round-6 wrapper must run from this committed clean checkout on
CPU/fp64 with the `d180` dump lane and NEMO E3T mode `both`.

In the authoritative `forcing_only` arm, evaluate in this order:

1. row 1.2 `sshn_e/un_e/vn_e`, using the original POINTWISE `1e-15` bars;
2. row 1.3 `ssh/ub/vb` after substep 1, using the original ACCUMULATING
   `1e-12` bars and aggregate axes;
3. row 1.4 final `puu_b/pvv_b/pssh/un_adv/vn_adv`, using the original
   ACCUMULATING `1e-12` bars and aggregate axes.

Every field in a row must be `AT BAR`.  The first failed field stops the chain.
Rows 2--6 are released only if every authoritative field in rows 1.2--1.4 is at
bar.  The exact-seed arm, post-hoc component diagnostics, and aggregate
improvements cannot override this rule.

## Controls and provenance

Require exact held U/V forcing receipts, finite registered populations, the
wrapper's planted identity and binding perturbation controls, restored
monkeypatches, clean-before/clean-after receipts, checkout-local production
imports, and hashes of the restart, mesh, dumps, NEMO sources, production
modules, and scorers.  Any failure is `INVALID` and supplies no chain evidence.

This round changes no physics and authorizes no further fix.  If row 1.4 clears,
row 2 is measured next from its existing deterministic-writer dump.  If it does
not, the first failed row-1.4 operand becomes the next ordered peel.
