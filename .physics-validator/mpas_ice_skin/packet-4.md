# Adversarial review packet 4 — prognostic sea-ice skin (round-3 fixes)

Independent adversarial re-review. Round 3 (review-3.md) confirmed the core
per-step re-anchor fix correct and raised two defensive items + one acknowledged
pre-existing property. Both defensive items are now fixed. Verify, and confirm
whether anything substantive remains.

## Round-3 item 1 (NEW) — feature-on shape guard bypassable
With the prognostic skin an `(nCells,)` ice component, a SCALAR/mis-counted
SST or SIC would broadcast to `(nCells,)` and pass the output-only `_ts0` check
(the old scalar-`T_ice` blend caught it). Fix — validate the RAW shapes first
(`model_driver.py`, feature setup):
```python
            _sst0, _sic0 = self.get_sst_sic(START_DAY)
            for _nm, _arr in (("SST", _sst0), ("SIC", _sic0)):
                if tuple(jnp.asarray(_arr).reshape(-1).shape) != (_ncell,):
                    raise ValueError(f"MPAS {_nm} forcing ... != nCells={_ncell}; ...")
            _ts0 = _blend_T_sfc(_sst0, _sic0)
            if _ts0.shape != (_ncell,):
                raise ValueError(...)   # retained belt-and-suspenders
```
Runs for feature-on and -off SST-forced MPAS runs; valid `(nCells,)` sources
pass unchanged (256->257 tests still green, incl. the non-feature MPAS land/CMOR
runs). Also removed the redundant double `jnp.asarray` (`_compute_T_sfc` now
passes raw SST/SIC; `_blend_T_sfc` asarrays once) — output byte-identical.

## Round-3 item 2 (defensive) — NaN checkpoint skin poisons anchors
A NaN restored skin corrupts even open-water (sic=0) cells via `0*NaN` in the
setup blend. Fix — refuse a non-finite restored skin in the seed overlay:
```python
                    if _skin.shape != (_ncell,):
                        raise ValueError("... mesh mismatch.")
                    if not bool(jnp.all(jnp.isfinite(_skin))):
                        raise ValueError(
                            "checkpoint ice_T_skin has non-finite values — "
                            "refusing to resume from a corrupt skin.")
```
New test `test_restart_refuses_nonfinite_checkpoint_skin`: writes NaN into a
day-1 checkpoint's `ice_T_skin`, restarts, asserts the run exits non-zero with
the "non-finite" message. (A finite-but-out-of-range skin self-corrects: the
first per-step advance clips to `[floor, T_melt_surface]`; only NaN is
persistent, and NaN is what the setup blend propagates — so finiteness is the
load-time invariant that matters.)

## Round-3 acknowledged (no code change)
- Radiation-subcycle restart branch enters the skin update (restart step 0
  re-solves radiation while the straight run may hold SW/LW). Codex-3 agrees
  this is the PRE-EXISTING, state-wide radiation restart branch, not a
  stale-anchor branch. The skin inherits the model's existing restart tolerance;
  it does not amplify it. Declared in the report; the integration test asserts
  the ice-free open-water/restart wiring, and `test_restart_split_invariance`
  proves the advance itself is a pure carry (a checkpointed skin resumes
  exactly given the same fluxes).
- `load->save-without-run`: you confirmed no silent strip remains; the physstate
  staging raises first for that unsupported op. No change.
- Per-step re-blend perf: eager O(nCells) unfused blend+lapse each step, same
  class as the sibling per-step qv-smooth / hard-sat drains; small vs the MPAS
  dynamics, not claimed free.

## Verification (srun, CPU x64)
- **257 passed** across the 5 baseline files. `test_mpas_ice_skin.py` = 22 tests
  (helper physics + validate_strict + CLI + subprocess wiring/checkpoint/restart
  + NaN-refusal).

## Ask
Confirm items 1 and 2 are fully fixed with no new regression (esp. the raw-shape
guard not rejecting any valid `(nCells,)` or `(nCells,1)`-reshapeable source, and
the finiteness guard placement). State whether any substantive finding remains;
if not, say the packet is closed and why each prior finding is resolved.
