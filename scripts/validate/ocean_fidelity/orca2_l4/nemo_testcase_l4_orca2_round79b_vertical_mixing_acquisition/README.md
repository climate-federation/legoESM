# ORCA2 round-79b acquisition — the per-step vertical-mixing chain

## What it records

One file per rank per step, for the first ten steps, holding NEMO's own
diffusivities at every boundary of the compiled `zdf_phy`, in NEMO's execution
order, plus the turbulence closure's internals:

| boundary | NEMO statement (ORCA2 build, compiled `zdfphy`) | arrays recorded |
|---|---|---|
| after the closure | `:349-350`, the copy of the closure's coefficients into the working arrays | `avt_after_tke`, `avm_after_tke`, `avt_k`, `avm_k` |
| after the river mouths | `:355`, `avt = avt + 2 * rn_avt_rnf * rnfmsk * wmask` | `avt_after_rnf`, `rnfmsk` |
| after the convection arm | `:359`, `zdf_evd` | `avt_after_evd`, `avm_after_evd` |
| after the salt/heat split | `:363`, `zdf_ddm`, which also adds to the momentum coefficient (`zdfddm.f90:172`) | `avt_after_ddm`, `avs_after_ddm`, `avm_after_ddm` |
| after the internal waves | `:372`, `zdf_iwm` (de Lavergne; `zdfiwm.f90:314-316` adds the wave diffusivity to all three) | `avt_after_iwm`, `avm_after_iwm`, `avs_after_iwm` |

The momentum coefficient is captured at both the split and the wave arm because
both add to it, so neither increment can be read off the other's boundary.

The closure internals come from `zdftke.f90`'s `tke_avn`: the turbulent energy
`en`, the two mixing lengths `zmxlm`/`zmxld` (`:697-698` under `nn_mxl = 3`),
the dissipation length scale `dissl` (`:711`), and the resolved scalars
`rn_ediff` and `rmxl_min`.  The background profiles `avtb`, `avmb` and
`avtb_2d` come along so the closure's own floor arm (`:709-710`) can be
reproduced offline.

## Why

Round 78 showed that legoESM's ORCA2 card had been running an effective 1 m
turbulence mixing-length floor where NEMO's value for this deck is 1.0e-3 m,
and that correcting it made the ladder's end-of-step extrema worse.  A floor
that is NEMO's own number is not the defect, so the 1 m floor was covering an
error elsewhere.  This record lets each candidate be substituted into legoESM
one at a time.

## Shape of the record

`NEMO_L4_ZDFV_1` in sixteen bytes, then fifteen header integers (version, step,
the two time-level indices, the rank, the two global offsets, the three local
extents, the real width in bits, the array count, the resolved mixing-length
selector, and the two resolved switches), then for each array a sixteen-byte
name, seven integers (rank, three extents, three origins) and an
eight-byte-per-value payload.  The checker takes every name and every payload
length out of the file; it predicts no byte count and no header tuple.

## Files

* `zdfphy_round79b_writer.F90` — the write-only module.
* `zdfphy_round79b.patch` — additions-only, against `src/OCE/ZDF/zdfphy.F90`.
* `zdftke_round79b.patch` — additions-only, against `src/OCE/ZDF/zdftke.F90`.
* `run.sh` — the launcher (`--preflight-only`, `--run`, `--admit-existing`).
* `../nemo_testcase_l4_orca2_round79b_vertical_mixing_gate.py` — the admission
  gate; its five plants are proven to fire by
  `tests/ocean/fidelity/test_orca2_round79b_vertical_mixing_gate.py`.

## Admission

Note-AS: the patch is write-only, so all four ten-step restart files must be
byte-identical to the pinned record at
`phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`.
The five plants (header, field order, truncation, producer stamp, content) must
each make the gate refuse before the real admission runs.

## Command

```
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/orca2_r79b/repo/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round79b_vertical_mixing_acquisition/run.sh --run
```
