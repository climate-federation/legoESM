# Adversarial review packet 5 — prognostic sea-ice skin (round-4 fix)

Independent adversarial re-review. Round 4 (review-4.md) confirmed everything
resolved EXCEPT one compatibility edge: feature-on `(nCells,1)` SST/SIC. Fixed.

## Round-4 finding (the only open item) — `(nCells,1)` source broadcast
`_blend_T_sfc` passed the original 2-D `(nCells,1)` arrays to
`blend_surface_temperature`; against an `(nCells,)` skin, `(nCells,1)*(nCells,)`
broadcasts to `(nCells,nCells)`, which the retained `_ts0` guard then rejects.
The old scalar-`T_ice` blend tolerated `(nCells,1)` via the trailing reshape.

Fix — flatten SST/SIC to `(nCells,)` FIRST (your exact recommendation):
```python
            def _blend_T_sfc(_sst, _sic):
                _sst = jnp.asarray(_sst).reshape(-1)
                _sic = jnp.asarray(_sic).reshape(-1)
                _ice_component = self._ice_T_skin if _ice_skin_on else _T_ice
                _ts = blend_surface_temperature(_sst, _sic, _ice_component).reshape(-1)
                if _lapse_z is not None:
                    _ts = land_lapse_adjusted_surface_temperature(
                        _ts, _f_land_cells.astype(_ts.dtype),
                        _lapse_z.astype(_ts.dtype), _land_lapse_K_m)
                return _ts
```
Now `(nCells,)`, `(nCells,1)`, and any `(nCells,)`-reshapeable source all blend
elementwise; scalar/mis-counted sources are still caught by the raw-shape guard.
Byte-identical for the non-feature path (`.reshape(-1)` is a no-op on a valid
`(nCells,)` source, and reproduces the old `(nCells,1)->reshape(-1)` result).

New unit test `test_blend_flattens_2d_source_against_array_skin`:
```python
    sst = jnp.full((n, 1), 290.0); sic = jnp.full((n, 1), 0.5)
    skin = jnp.linspace(240.0, 250.0, n)                 # (n,) array skin
    out = blend_surface_temperature(sst.reshape(-1), sic.reshape(-1), skin)
    assert out.shape == (n,)                             # elementwise, not (n,n)
    assert blend_surface_temperature(sst, sic, skin).shape == (n, n)  # the bug
```

## Verification (srun, CPU x64)
- **258 passed** across the 5 baseline files (`test_mpas_ice_skin.py` = 23
  tests: helper physics, melt-band, restart-split invariance, autodiff, the
  `(nCells,1)` blend, validate_strict, CLI, subprocess wiring/checkpoint/restart,
  NaN-refusal).

## Complete round-by-round resolution (for your final judgement)
- codex-1 #1 diurnal snapshot -> per-step advance at DT.
- codex-1 #2 restart drop -> per-step advance (checkpoint holds fully-advanced
  skin) + per-step re-anchor.
- codex-1 #3 tas uses T_ice -> tas uses the skin.
- codex-1 #4 melt cap -> surface-melt cap T_freeze vs basal T_freeze_ocean.
- codex-1 #5 SIC timing -> per-step advance with current-day SIC + open-water snap.
- codex-1 #6 / codex-2 #3 laundering + load->save -> config-gated save + staged
  fallback (no silent strip).
- codex-1 #7 / codex-2 #2 conditional stability -> per-step re-anchor makes the
  feedback per-step (r*lambda = DT*lambda/C << 1); docstring restated.
- codex-2 #1 mid-day restart anchor branch -> per-step T_sfc re-anchor from
  cached SST/SIC vs restored skin.
- codex-3 shape-guard bypass -> raw SST/SIC shape validation.
- codex-3 NaN skin -> finiteness refusal + test.
- codex-4 (nCells,1) -> flatten inside _blend_T_sfc + test.

## Ask
Confirm the `(nCells,1)` fix is complete with no new regression, and state
whether the packet is now closed (no substantive findings). If any concern
remains, cite file:line.
