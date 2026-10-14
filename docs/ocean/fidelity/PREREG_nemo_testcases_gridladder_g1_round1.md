# PREREGISTRATION — grid ladder rung G1, round 1 (gridladder_r1)

Committed BEFORE any NEMO or legoESM measurement of G1. Nothing below is a
measured G1 result: no DOMAINcfg, makenemo or NEMO run exists yet. Numbers
quoted are either read from NEMO's source or computed offline from ORCA2's
shipped domain file, and each says which.

## 1. What G1 is (DECISION 117) and the operator picks

G1 = the TSUNAMI program on ORCA2's horizontal grid: T-pivot north fold,
cyclic east-west, flat bottom, one wet level, no surface forcing, no sea ice,
uniform T/S, an initial sea-level bump, split-explicit free surface, EEN
vorticity, constant vertical mixing, everything else off.

Operator picks inside D117, as built this round: depth 100 m and jpk = 2;
ORCA2's `nn_bt_flt = 3`, `rn_bt_alpha = 0.09`; `rn_Dt = 5400 s`; the bump is
TSUNAMI's formula and amplitude centred on (ji,jj) = (51,74) (180 E, equator);
compiled keys = the ORCA2_OMIP_L4 keys without key_si3 / key_top; the domain
file comes from NEMO's own tools/DOMAINcfg.

## 2. Four findings of the survey that change what G1 can be

**F1. An all-wet flat ocean on ORCA2's grid cannot run.** ORCA2's two northern
mesh poles sit on land; there `e1t` is 0.1747 m (offline, shipped domain file,
0-based (j,i) = (147,90), 100 W 70 N). NEMO's external-mode Courant number is
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1056-1080`. With every cell wet at
100 m the maximum is 14,891 at `rn_Dt = 5400`, `nn_e = 65` (the stop is at
0.9), and `ln_bt_auto` would ask for 1.2 million substeps. **Built instead:
ORCA2's own land-sea mask over the flat 100 m floor.** The same number is then
0.0781 (maximum at 0-based (1,49), `e1t = e2t = 47,096 m`, the Antarctic coast).
The south wall is therefore ORCA2's Antarctic coast, and ORCA2's coastline is a
further new thing versus TSUNAMI. UNASKED — see section 8 and DECISION_NEEDED.

**F2. ORCA2's RK3 deck does not run 5400 s.** It runs
`cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:28` (10800 s); 5400 s is the reference
namelist's value `cfgs/SHARED/namelist_ref:72`. ORCA2 also leaves the substep
count to `ln_bt_auto` (`cfgs/SHARED/namelist_ref:1099-1101`), which resolved
`nn_e = 65` in the admitted ORCA2 record. Built as briefed: `rn_Dt = 5400`,
`ln_bt_auto = .false.`, `nn_e = 65`. Both are one deck line each and the card
reads them from the deck.

**F3. With a domain file, NEMO never applies a user bump.**
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:395-400` sets ssh to zero under
`ln_read_cfg` and calls the user routine only otherwise. G1 therefore carries
the case-local copy of `src/OCE/IOM/restart.F90:457-461` with ONE added
statement (the user call after the zero). Without it the run is a motionless
ocean. UNASKED deviation, add-only, in the case's MY_SRC, never in NEMO `src/`.

**F4. The certified TSUNAMI card is not on this branch.** Searched
`grep -rli tsunami` over packages, scripts, tests and docs: only the two
advection-OFF arms (commit 1fe1448df) are here. `build_tsunami_zco_card`, its
one-level opt-in, its j-periodic option and its acquisition scripts live on
branch fidelity/nemo-testcases-tsunami (PR #1920). This round ports only the
one-level opt-in: the same keyword and default, laid out so that no previously
cited line of that file moves (a small merge conflict is expected). The one-level STEP is unproven
on this branch until that branch is merged here.

## 3. Survey

### (a) The ORCA2 L4 build
- Keys `cfgs/ORCA2_OMIP_L4/cpp_ORCA2_OMIP_L4.fcm:1`: key_si3 key_qco
  key_vco_1d3d key_RK3. G1 compiles key_qco key_vco_1d3d key_RK3.
- Its MY_SRC holds fifteen files (fourteen source overrides, four of them
  sea-ice, and one include). Its stprk3 was diffed against NEMO's and carries
  oracle writers; the others were not diffed. G1 carries none of them.
- Program: NEMO's RK3 driver. Split-explicit `nn_bt_flt`/`rn_bt_alpha` at
  `cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:365-366`.
- Resolved switches over the reference namelist: section 4's table.

### (b) The shipped TSUNAMI case
- State `tests/TSUNAMI/MY_SRC/usrdef_istate.F90:65-68` (T = 20, S = 30, rest)
  and bump `tests/TSUNAMI/MY_SRC/usrdef_istate.F90:94-102`; zero forcing
  `tests/TSUNAMI/MY_SRC/usrdef_sbc.F90:60-68`.
- G1 keeps verbatim: the T/S/velocity statements, the radius and profile
  statements of the bump, every zero-flux statement.
- G1 adapts: the bump's distance (degrees from the centre T point, longitude
  difference wrapped) and a land mask on the bump.
- G1 does NOT keep TSUNAMI's stprk3/stpmlf/diawri/usrdef_hgr/usrdef_zgr/
  usrdef_nam overrides. G1 runs NEMO's unmodified RK3 driver, as ORCA2 does.
  TSUNAMI's stripped driver drops the eos_rab/bn2/zdf_phy/lateral-physics
  calls, most diagnostics and the restart write; G1 executes them.
- NEMO's dispatch that makes the OFF switches real:
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynadv.f90:185`,
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:626`; the user routines are reached
  at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/istate.f90:129-133` and
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:442`.

### (c) tools/DOMAINcfg
- ORCA2 ships no coordinates or bathymetry file, only the finished domain file
  (180 x 148, SHA-256 7125f7a5...2839). DOMAINcfg reads the horizontal mesh
  from it directly: `tools/DOMAINcfg/src/domhgr.F90:484-502`.
- It recomputes Coriolis: `tools/DOMAINcfg/src/domhgr.F90:383-384`. NEMO then
  reads it back from the file:
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:233-237`.
- Uniform one-level grid: `tools/DOMAINcfg/src/domzgr.F90:425-437` with
  `ppkth = 0`, `pphmax = 100`, `jpkdta = 2`; full steps
  `tools/DOMAINcfg/src/domzgr.F90:961-984`.
- Mask: level bathymetry read at `tools/DOMAINcfg/src/domzgr.F90:625-629`.
  The input is 1 on ORCA2's wet cells, 0 on land (make_bathy_level.py).
- Two traps avoided: `cp_cfg` is not "orca", because
  `tools/DOMAINcfg/src/domzgr.F90:633-652` would write levels 15 and 12 into a
  one-level column; `rn_hmin = -1`, because
  `tools/DOMAINcfg/src/domzgr.F90:728-733` indexes `gdepw_1d(ik+1)`.
- The tool writes the old `jperio` variable
  (`tools/DOMAINcfg/src/domain.F90:417`), which NEMO 5.0.2 still decodes:
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domain.f90:595-604`. No strait file variable
  is written; NEMO treats it as optional
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:289-291`). 3-D thicknesses are
  read under key_vco_1d3d at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domzgr.f90:179-188`.
- Namelist: domaincfg_namelist_cfg (committed). Command, run by run.sh in
  `<evidence>/domaincfg`: `maketools -m conda-scalarmath -n DOMAINcfg`, then
  `mpirun -np 1 ./make_domain_cfg.exe` with coordinates.nc a symlink to the
  shipped domain file. The tool has never been built on this machine.

### (d) legoESM on this branch: what G1's card reuses
Searched `grep -n "^def \|^class "` in the recipe and `grep -rn` for
allow_single_level, j_periodic, eos_depth, nemo_barotropic_filter_alpha,
namelist parsers in packages/.
- Reused unchanged: tripolar grid loader `tripole.py:453`; ORCA2 mask and fold
  images `nemo_testcase_recipe.py:1290`; depth ladder
  `nemo_testcase_recipe.py:1271`; fold-owned F mask `vertical.py:281`; raw-mesh
  operand carrier `vertical.py:883` and its resolved builder `vertical.py:673`
  (D114); required alpha `ocean_model_latlon_cgrid.py:4415` (D115); the
  advection-OFF arms; the RK3 identity "orca2_vector_een_c2" with the OMT-0
  five-OFF selections.
- New: `nemo_testcase_recipe.py:3650` (builder), `:3562` deck reader, `:3599`
  bump, `:3863` validator. A separate builder rather than a parameterised
  ORCA2 card: the ORCA2 builder requires five input files G1 has no use for
  (T/S, viscosity, wave mixing) and builds ice and TKE state; the shared part
  is the geometry helpers, which are called, not copied.
- The card is dispatched by the shared card factory but is NOT listed in the
  shared validator's table: its validation is deck-bound and has its own pair
  (the precedent is the ORCA2 rung-0 and OMT cards, validated beside their
  gates). Listing it would also move lines that older receipts cite.
- No namelist reader existed in packages/ (three exist under scripts/ and are
  not importable from the package); a 20-line one was added.
- THE ONE new explicit selector: `vertical.py:1084` (one-level column), default
  off, selected only by this card at `nemo_testcase_recipe.py:3713`. No second
  builder, fold, mask or mesh machinery.

## 4. The G1 deck against its two parents

| switch | G1 | ORCA2 L4 | TSUNAMI |
|---|---|---|---|
| time step | 5400 | 10800 | 1000 |
| substeps | 65, stated | auto, resolves 65 | auto, resolves 6 |
| barotropic filter / alpha | 3 / 0.09 | 3 / 0.09 | 1 / 0 |
| domain | read, DOMAINcfg | read, shipped | analytic |
| levels | 1 wet | 30 wet, partial cells | 1 wet |
| initial T/S | uniform, user | data + damping | uniform, user |
| forcing | user, zero | CORE bulk | user, zero |
| forcing call period | 2 | 2 | 1 |
| ice / icebergs | none | SI3 / on | none |
| lateral slip | 2 | 2 | 0 |
| equation of state | EOS-80 | EOS-80 | simplified |
| momentum advection | OFF | vector | OFF |
| tracer advection | OFF | FCT | OFF |
| lateral mixing (both) | OFF | on | OFF |
| bottom drag | OFF | linear | OFF |
| vertical mixing | constant | TKE + extras | constant |
| BBL / geothermal / damping | off | on | off |
| step driver | NEMO's | NEMO's (instrumented) | case override |

More than one row differs from each parent, so no single G1 output difference
is attributable to one switch from this table alone.

## 5. Offline numbers (shipped domain file, not NEMO output)

- Wet cells 16,433 of 26,640; fold rows carry 70 / 69 / 66 wet cells; south
  row all land; no isolated wet cell (so DOMAINcfg is predicted to keep the
  mask unchanged).
- Bump: centre 179.99992 E, 0 N; radius 9.9416 degrees; 201 T cells, none on
  land; rows 62..86, columns 47..55 (one-based).
- Gravity-wave speed 31.3 m/s, 169 km per step. The bump's wave reaches the
  fold at about kt 40, the Antarctic coast at about kt 45 and the seam column
  at about kt 66 (distance / speed; PLAUSIBLE, not measured).

## 6. Predictions (frozen) and falsifiers

| id | prediction | falsifier |
|---|---|---|
| P1 | DOMAINcfg builds and writes a 180 x 148 x 2 file whose wet mask equals ORCA2's and whose thicknesses are all 100 m. | Any other shape, a changed mask, or a thickness that is not 100. |
| P2 | NEMO reports nn_e = 65 set by the namelist, passes its own 0.9 Courant stop, and completes 100 steps. (NEMO prints the Courant number only under ln_bt_auto, so 0.078 stays an offline number.) | A stop of any kind, or a different nn_e. |
| P3 | Geometry identity: 19 of 19 card arrays bit-identical to NEMO's mesh file (TSUNAMI had 35/35 on its own list). | Any unequal array; the first one named is the round-2 owner. |
| P4 | Fold statements predicted EXACT on G1: the T/U/V/F fold masks and the raw face-thickness operands (D114, round 238), the required alpha (D115, round 237), the fold exchange of ssh and of the barotropic pair. | A non-bit row inside the fold band whose seam and south rows are exact. |
| P5 | Fold statements G1 does NOT exercise (identified by their round titles, not re-read): the vector-form unit (rounds 212-217, 229), the tracer and FCT fold walks (224-227), the pressure-gradient fold split with real density (240), drag splits (250-252). They stay UNMEASURED on G1. | — |
| P6 | TSUNAMI-certified statements predicted exact given the merge in F4: the split-explicit substeps, EEN with planetary vorticity only, the cyclic i-exchange, the one-level column, both advection-OFF arms. | A non-bit row in the interior (away from fold, seam, coast) at kt = 1. |
| P7 | NOT predicted exact: the boxcar-free dissipative filter on one level and EOS-80 with a moving one-level depth were never run together; the first non-bit statement, if any, is expected there or at the coast. | — |

## 7. The discriminating quantity

At kt = 1 the wave has not reached any boundary, so absolute errors there are
at denormal scale and say nothing. The discriminator is therefore the COUNT of
bit-unequal sea-level cells per region, at the first step where it is nonzero:
fold band (the top three rows), seam (the first and last two columns), south
(the southernmost wet rows), interior (the rest). Fold defect: fold-band count
nonzero first, seam and south zero. Seam or south-wall defect: the reverse.
Interior nonzero first: neither, a program defect (P6/P7). The same three
maxima of |ssh - ssh_NEMO| are reported from kt 40 on, where they are finite.

## 8. Choices made this round

ASKED (operator picks, D117): 100 m; jpk = 2; filter 3 / alpha 0.09; rn_Dt
5400; bump formula, amplitude and (51,74); keys; DOMAINcfg.

UNASKED, each offered for revert:
1. ORCA2's coastline instead of an all-wet ocean (F1; all-wet cannot run).
2. Substeps stated as 65 with ln_bt_auto off (F2), not auto (would resolve 7).
3. The one added statement that applies the bump on a read domain (F3).
4. Bump distance in degrees with wrapped longitude; land mask on the bump.
5. EOS-80 (ORCA2's) rather than TSUNAMI's simplified equation of state.
6. No-slip lateral condition 2 (ORCA2's); inert here by reading, not measured.
7. NEMO's own step driver, not TSUNAMI's stripped override.
8. Forcing call period 2 (ORCA2's), so run lengths are even.
9. Single-rank runs; restarts every 10 steps for the passivity check.
10. Full-step vertical coordinate in DOMAINcfg with a level mask.
11. Porting the one-level opt-in from the TSUNAMI branch ahead of its merge.
