# Ralph loop task: drive every legoESM term to corr = 1.0 vs NEMO's own arrays (#1226)

## The bar (user directive, non-negotiable)

> "we need corr of 1, these minor differences will leak in as problems that we have
> been trying to chase down for very long now"

**1.0, not "close".** Today's evidence for why: N² is off by 54% at the top of the
chain, yet κ_GM reads 0.985 (column integrals average the error out) and the
slopes read 0.959 (N² sits in a denominator dominated by the slope cap). Three
layers of plausible-looking agreement hiding a factor-1.5 error in the input.
A term that "looks fine" downstream proves nothing about its inputs.

## Why a loop NOW (and not before)

The procedure is proven and mechanical: ~5 min per iteration, objective criterion,
no judgment needed per cycle. Yesterday this work had 3-hour feedback and no
established method — a loop then would have amplified false leads. It won't now.

## The proven cycle (do NOT re-derive this)

1. **Add a dump** to `~/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/MY_SRC/ldftra.F90`,
   inside `ldf_eiv_trp_MLF`, in the existing `IF( kt == kit000 .AND. cdtype == 'TRA' )`
   block. Units already used: 8801 eiv_u, 8802 eiv_v, 8803 wslpi, 8804 aeiu,
   8805 rn2b, 8806 e3w, 8807 gdept. Use 8808+.
   Pattern (open at `jk==1`, write every level, close at `jk==jpkm1`):
   ```fortran
   WRITE(88NN) ( ( <array>(ji,jj,jk), ji=1,jpi ), jj=1,jpj )
   ```
   Module arrays need a `USE` (e.g. `USE zdf_oce, ONLY : rn2b`).
2. **Rebuild**: `cd ~/oracle-builds/nemo5/nemo_5.0.2 && ./makenemo -n DINO -m linux_gfortran -j 8`
   (fast — only the changed file + dependents).
3. **Run**: `cd cfgs/DINO/RUN_GDB && ./nemo` — 1 rank, 3 steps, reads the REBUILT
   global restart `RUN_Y5_REBUILD/DINO_00057600_restart.nc`.
4. **Compare** in numpy. Dump layout is `[k][jj][ji]` float64,
   `jpi=56, jpj=203, jpkm1=35`; strip the `nn_hls=2` halo:
   ```python
   arr = np.moveaxis(np.fromfile(f, dtype=np.float64).reshape(35,203,56), 0, -1)[2:-2, 2:-2, :]
   ```
   legoESM side: bridge the SAME restart, `LEGOESM_NEMO_E3T=both`, call the leaf
   function directly (outside jit).
5. **Record** the correlation + ratio in this file and in memory.

Also available with NO rebuild: **gdb call tracing** (the binary is not stripped).
`gdb -batch -x cfgs/DINO/RUN_GDB/trace.gdb ./nemo` gives the per-step call order.

## Current scoreboard (all vs NEMO's own dumped arrays, same y5 restart state)

| term | corr | ratio | status |
|---|---|---|---|
| N² (`rn2b`) | **0.486–0.545** | **1.54–1.76** | OPEN — biggest gap, most upstream |
| `wslpi` (slope) | 0.9585 | 0.975 | OPEN |
| `aeiu` (κ_GM) | 0.9847 | 1.002 | OPEN (close, but not 1.0) |
| eiv transport | 0.7727 | 1.040 | OPEN — amplified by ψ's vertical difference |
| `e3w`, `gdept` ladders | ~1e-4 | 1.000 | **MATCHED** |

## N² — what is already eliminated (do NOT re-chase)

- **Formulation**: legoESM's adiabatic N² (0.5378/1.544) and its NEMO-faithful
  linearised-α/β `compute_buoyancy_frequency_nemo_bn2` (0.5450/1.5366) agree with
  EACH OTHER and disagree with NEMO identically. Not the α/β-vs-parcel choice.
- **e3w divisor / gdept ladder**: match NEMO to ~1e-4 (dumped and compared).
- **T/S time level**: identical result from `tn/sn` and `tb/sb` (0.486204 vs
  0.486212) — and verified non-vacuous (the before-state genuinely differs:
  max|ΔT| = 0.245 K).
- **Vertical index offset**: k-shift scan gives −1→0.36, 0→0.54, +1→0.62. No
  clean offset, but the sensitivity means **the level convention of the
  comparison itself is the prime remaining suspect** — re-derive which interface
  each side assigns N² to, from `eosbn2.F90` (NEMO computes only jk=2..jpkm1,
  surface/bottom zeroed in `istate.F90`) and from legoESM's function.

**START HERE**: settle the N² level convention. Everything downstream inherits it.

## Term list (work down; each is one cycle)

1. N² `rn2b` ← in progress
2. `wslpi` / `wslpj` slopes (after N² is 1.0 — it feeds them via `zdzr`)
3. `aeiu` κ_GM
4. eiv transport (should follow once 1–3 are 1.0)
5. tracer advection fluxes (`traadv_fct`: upstream, high-order, limited)
6. PGF (`hpg_sco`)
7. Coriolis / EEN vorticity
8. vertical mixing (`zdftke`), EVD trigger

## Guardrails (each earned by a failure TODAY)

- **Every change is a NEMO TRANSCRIPTION with `file:line`.** NEVER tune a
  coefficient to raise a correlation. If you cannot cite NEMO's line, stop.
- **Mask to WET/ACTIVE cells; use mean/p50/p99, not `max`.** An unmasked `max`
  caught dry-cell garbage and produced a false "+37%" lead.
- **Verify the comparison before believing it**: k-shift scan for alignment, and
  confirm the inputs genuinely DIFFER before reporting "no change" (a before/now
  test that returns identical numbers from identical inputs proves nothing).
- **Report inert fixes as inert.** The face-thickness slope bound was correct
  transcription and moved nothing (it only binds above ~70 m). Ship it, say so.
- **Faithful-but-worse is a signal, not a reason to revert.**
- **Record retractions in memory** next to the finding they replace. Two on the
  GM thread alone today.
- Fix as a **selectable card option**, defaults unchanged; direct test with a
  synthetic-violation check.
- Explicit git pathspecs, never `git add -A`. Do not stage
  `docs/ocean/fidelity/mitgcm_oracle_status.md` or `pyproject.toml`.

## ESCALATE to the human (do not decide alone)

- A hypothesis is **refuted** and the next step requires re-framing.
- A correlation cannot be raised without changing something NEMO does not do.
- A fix would alter legoESM defaults for non-oracle users.
- Two consecutive cycles produce no movement.

## Standing context

- The true-grid instability (`LEGOESM_NEMO_E3T=both` + `through_fct` grows to
  ~3 m/s; `centred` is stable) is UNEXPLAINED after ~12 internal eliminations.
  It is expected to fall out once the eiv chain reaches 1.0 — that is the bet
  this loop is testing.
- The vertical-grid fix is implemented and gated OFF (`mode="off"`) ONLY because
  of that instability. Flip to `"both"` the moment it is resolved.
- Full history: memory `project_dino_nemo_oracle.md` addenda 30–47.
- Skill: `.claude/skills/oracle-fidelity/SKILL.md` (Rule 0 = read the oracle's
  source first; Rule 1 = coverage, not a checklist).
