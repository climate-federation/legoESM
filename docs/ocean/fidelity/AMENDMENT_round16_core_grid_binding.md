# Amendment: bind the V-face metric owner in round-16 acceptance

Date: 2026-08-29. Frozen before the final CPU acceptance/replay rebind. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Adversarial evidence review found a provenance-only omission: the hardened
acceptance bound the DINO card, barotropic solver, bridge, NEMO readers, and
probe/scorer modules, but did not bind `legoesm.grids.latlon`, the core module
that constructs the corrected V-face `dx_v`.

The final acceptance must therefore additionally:

1. import `legoesm.grids.latlon` from beneath the measured clean checkout;
2. record its SHA-256 under production binding key `core_grid`;
3. require the combined adjudicator's production binding set to contain that
   exact key in addition to the previously frozen keys.

Then rerun the unchanged CPU/fp64 acceptance, recurrence, and combined
adjudication at one clean commit. All prior numeric bars, candidates, inputs,
controls, order, and dispositions remain frozen. This amendment authorizes no
physics change and no climate/GPU arm.
