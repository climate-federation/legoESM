# Preregistration — NEMO-testcases grid ladder G5, round 1

**Frozen:** 2026-10-15, before the G5 acquisition is run.

## Claim and exact target

G5 is the certified G4 resolved deck plus ORCA2's sea-ice selection: `nn_ice = 2`
and the complete `ORCA2_OMIP_L4/EXP00/namelist_ice_cfg` overlay, resolved against
`cfgs/SHARED/namelist_ice_ref`.  No other G4 switch may change.  For this round,
“OMT-0” means the target stated in the round brief: momentum advection, tracer
advection, lateral momentum/tracer diffusion, and bottom drag off; atmospheric
forcing and SI3 ice on.  The older ORCA2-lane Decision-103 object used the same
name for an all-forcing-off control; that historical object is not silently used
as this round's target.

The acquisition uses NEMO's own post-SI3, per-step ocean boundary records.  The
card does not implement a second sea-ice model.  It maps recorded combined ocean
stress, solar and nonsolar heat, `emp`, runoff, salt exchange, ice fraction, and
runoff tracer content.  It must also preserve `rCdU_ice` in the evidence bundle.

## Frozen prediction

1. The resolved configuration diff between G5 and the round-defined OMT-0
   target is empty.
2. At entry, T/S/U/V and the G4 free-surface perturbation are bitwise identical
   after applying the same SI3 initial snow/ice/pond global-mass adjustment used
   by the ORCA2 card.
3. The already-supported post-SI3 record mapping is exact for heat, freshwater,
   salt, ice fraction, runoff-tracer content, and combined ocean stress.
4. The first non-bitwise statement is predicted at the first external-mode use
   of NEMO's implicit ice-ocean top drag: `dynspg_ts.f90:1431-1432`.  The actual
   SI3-enabled executable sets `rCdU_top` from recorded `rCdU_ice` in
   `zdfdrg.f90:178-188`; legoESM presently has no top-drag forcing input.  This
   is a code-path prediction, not permission to omit the record.
5. As in G3/G4, the reference may stop while beginning `kt=11`; the expected
   admissible boundary is completed steps 1–10.  The script must discover the
   actual completed boundary, run twins only to it, and request restarts for
   every writable step from 2 through that boundary.  Step 1 is forbidden.

## Confirmers and falsifiers

The prediction is **confirmed** only if the admitted record proves all entry and
supported boundary mappings exact through the statement immediately before the
first non-bitwise top-drag consumer, while `rCdU_ice` is nonzero on some wet
ice-covered cell and the first mismatch is at the preregistered consumer.

It is **falsified** by any earlier state or supported-flux mismatch, any resolved
deck difference from the round-defined target, a zero/absent/non-finite
`rCdU_ice` record on all wet ice-covered cells, or exact agreement through the
first top-drag consumer.  A different completed-step boundary does not itself
falsify the physics prediction; it changes the admitted comparison horizon and
must be reported exactly.

No bar verdict is claimed in round 1.  This round stops for operator acquisition
and returns the absolute `run.sh` path.
