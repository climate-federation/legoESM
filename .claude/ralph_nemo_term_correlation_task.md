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

## Work in NEMO's OWN EXECUTION ORDER (user directive)

> "this bit by bit fix approach should start right at the start of when a model
> step cycle starts and fix every type of equation/terms it comes across...
> jumping the middle of a model step may lead to issues from earlier in the step
> that we didn't solve"
> "Don't go to next routine till a routine is 1.0 corr and ratio 1."

Order from `stpmlf.F90` (MLF; key_qco, key_vco_3d, no key_RK3).

## Current scoreboard (vs NEMO's own dumped arrays, same y5 restart state)

| # | routine | corr | ratio | status |
|---|---|---|---|---|
| 1 | `sbc` (utau/qsr/qns/sfx) | 1.000000 | 1.000000 | **PASS** |
| 2 | `eos_rab` (α, β) | 1.000000 | 1.000007 | **PASS** |
| 3 | `bn2` (`rn2b`) | 1.00000000 | 0.99997 | near — see below |
| 4 | `zdf_mxl` (`nmln`/`hmlp`) | 99.859% levels | — | near — 14/9920 cols |
| 5 | `ldf_slp` `wslpi` | 0.999177 | 0.998943 | **CLOSED** (\|x\| ratio) |
| 5 | `ldf_slp` `wslpj` | 0.998875 | 0.998447 | **CLOSED** |
| 5 | `ldf_slp` `uslp`  | 0.999340 | 0.998697 | **CLOSED** |
| 5 | `ldf_slp` `vslp`  | 0.999012 | 0.997594 | **CLOSED** |
| 6 | `ldf_eiv` (`aeiu`) | 0.9847 | 1.002 | OPEN |
| 10 | eiv transport | 0.7727 | 1.040 | OPEN |

### Item 3/4 — fixed 2026-07-27 (commit `634ee4aee`)

Two REAL transcription bugs, both found by reading the Fortran:

- **`zdfmxl.F90:96-101` bottom cap** — NEMO advances `nmln` on every level still
  BELOW threshold, clamped `MIN(jk,mbkt)+1`, so a never-reaching column ends at
  the SEAFLOOR, not the deepest interface. We had no cap → 339 columns 1-5
  levels too deep. **353 → 14 mismatched columns.**
- **`eosbn2.F90:1459` zrw depth** — α/β are evaluated at T-points and
  interpolated to the w-point using the TRUE `gdepw`, which equals the gdept
  midpoint ONLY on a uniform ladder. We passed the midpoint → 3.7e-4 median N²
  bias. MLD integrand: median rel err **3.66e-4 → 4.95e-5**, corr → **1.00000000**.

Also settled: **`hmlp = gdepw_0[nmln-1] * (1 + ssh/H)`** — validated by a control
test that reproduces NEMO's own dumped `hmlp` to **0.0 m** over 9920 columns.
The criterion itself is THICKNESS-FREE (eosbn2 divides by `e3w`, zdfmxl
multiplies it back — exact cancellation), so no z-star stretch belongs in the
integral; only in the depth.

**RETRACTED**: the earlier "N² off by 54%" (an off-by-one — correctly aligned it
was corr 1.000000) and "hmlp 14% too deep" (unvalidated depth lookup).

**NEXT (item 3 → 1.0)**: lego divides N² by the REFERENCE `e3w`; NEMO uses the
LIVE `e3w(jk,Kmm) = e3w_0*(1+r3t)` (`domzgr_substitute.h90:131`). That is the
3.0e-5 ratio deficit (ssh/H ~ 1e-4). It CANCELS in the MLD but NOT in the
slopes, so fix it before item 5. `col_stretch` in
`_nemo_mld_from_n2_integral` already shows how to recover `1+r3t` from
`z_coord.h_partial` without threading eta through signatures.

Eliminated for the residual 14 columns (do NOT re-chase): `mbkt` (0 disagreements
globally), `nlb10` (=2 on both sides), the level/index convention (validated).

### Item 5 `ldf_slp` — RETRACTION + re-baselined 2026-07-27

**"wslpi corr 0.9585" is WITHDRAWN.** Re-measured with a proper ±2 offset scan
over 342,134 wet W-points (production call chain, before-state, `bef.ssh`):

| offset | corr | ratio |
|---|---|---|
| −1 | 0.983733 | 1.2436 |
| **0** | **0.999110** | **1.0907** |
| +1 | 0.968548 | 0.9460 |

Sharply peaked at offset 0 ⇒ the convention is right and 0.9585 was itself an
alignment artifact. **The real gap is the RATIO: slopes 9.07% TOO LARGE**
(median rel err 0.83%, p90 3.35%; the p99 62% tail is near-zero-crossing cells,
not a fidelity signal). Corr is already ~1 — do NOT chase correlation here.

### ★★ METRIC LESSON — never score a SIGN-CHANGING field with a signed sum ★★

**"slopes 9% too large" was ALSO withdrawn.** `wslpi` changes sign, and the
signed sum near-cancels between shallow (negative-dominant) and deep
(positive-dominant) levels, so a ~1% asymmetric bias inflates ~9×:

    signed ratio  1.0907   <- ILL-CONDITIONED, meaningless here
    |wslpi| ratio 1.0121   <- the true aggregate magnitude bias
    RMS ratio     1.0079

**Always quote `|x|` and RMS ratios for sign-changing fields** (slopes,
velocities, tendencies). Keep signed sums only for sign-definite quantities.

### Item 5 RESOLVED 2026-07-27 (commit `874fb35c6`)

Root cause: the slope path fed a PARCEL-DISPLACEMENT (adiabatic) N², but NEMO's
`ldf_slp` consumes `rn2b` — the linearised α/β bn2 (`eosbn2.F90:1455-1468`).
The two diverge with PRESSURE, so the error GREW WITH DEPTH — matching the
measured signature exactly (bottom 8 levels carried ~60% of the excess).

New selectable `GMRediConfig.slope_n2` = `adiabatic` (default, byte-identical)
| `nemo_bn2`, surfaced as `DINOConfig.gm_redi_slope_n2`, on the kamm card.

                        adiabatic     nemo_bn2
    |wslpi| ratio        1.012076     0.998943
      levels k>=18       1.014793     0.999518   <- depth-growing bias GONE
      levels k<18        1.002542     0.996925
    corr                 0.999110     0.999177

ELIMINATED by measurement, do NOT re-chase: (a) the `S_max` cap — both sides use
0.01 from `rn_slpmax`, points at cap 530 lego vs 514 NEMO, <1% of the excess;
(b) the mixed-layer ramp — levels k<18 carry no excess.

NEMO re-instrumented + rebuilt with `wslpj`/`uslp`/`vslp` dumps (units 8818-8820)
so all four slope components can be checked, not just `wslpi`.
**BUILD GOTCHA**: `makenemo` needs the `nemo-build` conda env on PATH
(`Text::Balanced` Perl module); otherwise FCM dies with a confusing Perl error:
    env PATH="/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH" ./makenemo -n DINO -m linux_gfortran -j 8

## Term list (work down; each is one cycle)

1. ~~`sbc`~~ PASS · 2. ~~`eos_rab`~~ PASS · 3. `bn2` ← live e3w divisor
4. `zdf_mxl` (14 cols) · 5. `ldf_slp` · 6. `ldf_eiv` · 7. eiv transport
8. tracer advection (`traadv_fct`) · 9. PGF (`hpg_sco`)
10. Coriolis / EEN vorticity · 11. `zdftke`, EVD trigger

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


## ★ ONE ROOT CAUSE BEHIND THE items-3/4 RESIDUAL: static vs LIVE vertical grid

Checked lego's α/β against NEMO's own dumped `rab_b` (342,134 wet cells):
- **β is BIT-EXACT** (rel err 0.00e+00)
- α: median 4.70e-06, p99 1.11e-04
⇒ the EOS is NOT the floor.

`gdept` lego(static `t_depth_ref`) vs NEMO(live `gdept(Kmm)`):
**median rel 1.03e-04, max 6.32e-04** — the same order as EVERY remaining
residual (`bn2` ratio 3.0e-5, MLD integrand 4.95e-5, α p99 1.11e-4). It is all
`r3t = ssh/H`.

**NEXT for 1.0 on items 3/4**: thread `eta` so `gdept`/`e3w` are the LIVE
`gdept_0*(1+r3t)`. The stretch is column-UNIFORM (r3t is 2-D) ⇒ a per-column
scalar, not a new 3-D array. It CANCELS in the MLD criterion (thickness-free
identity) except via α/β's depth dependence, so expect more gain in `bn2` and
the slopes than in the 14 MLD columns.
**DO NOT use `z_coord.h_partial`** — it is the STATIC at-rest thickness and is
identically 1.0x (measured).

The slope residual (0.15-0.25% low) EXCEEDS this 1e-4 effect ⇒ a second,
unidentified contributor remains in `ldf_slp`.
