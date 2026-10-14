# Grid ladder rung G3 — round 1 preregistration (gridladder_r14)

Base: f6488baa7 (rungs G1 and G2 certified). No NEMO record of G3 exists;
nothing in this file was measured against NEMO. Evidence directory:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round14/.
NEMO citations are to the G1 build's compiled sources under
ORCA2G1_OMIP_L1_P3; the four files cited for the initial state, density and
pressure gradient (dtatsd, istate, eosbn2, dynhpg) are byte-identical in the
record binary's build ORCA2G1_OMIP_L1_P5 (cmp, this round). dtatsd, istate
and fldread are also byte-identical to ORCA2_OMIP_L4's own build.

## 1. The rung, and the operator's picks inside Decision 117

G3 = the certified G2 program with ONE switch, the initial state.

| item | G2 (certified) | G3 |
|---|---|---|
| initial T, S | uniform 20 °C, 30 (the case's usrdef_istate) | ORCA2's monthly climatology, read by NEMO's `&namtsd` |
| `ln_tsd_init` | .false. | .true. (ORCA2_OMIP_L4/EXP00/namelist_cfg:47) |
| `cn_dir`, `sn_tem`, `sn_sal` | not stated | ORCA2's lines, byte for byte apart from spacing (namelist_cfg:50, :54, :55) |
| climatology files | none | data_1m_potential_temperature_nomask.nc ca00905c27078305e80130ea6dd07fc575de2ae3257fad38003b329458cf7f7a; data_1m_salinity_nomask.nc ad648d972f0631bde7e1b598d98470a979b7f2649271fc9cf5ccfca158e31d0c |

No interpolation weights: ORCA2's deck names none (the file is already on
ORCA2's grid). The time interpolation is ORCA2's (monthly, climatological,
`ln_tint = T`), at ORCA2's calendar (nn_date0 = 010101, rn_Dt = 10800).

Operator picks (the user may redirect): damping OFF (`ln_tsd_dmp = .false.`,
`ln_tradmp = .false.`); the equation of state and pressure-gradient scheme as
ORCA2's deck selects them, already G2's (`ln_eos80`, `ln_hpg_sco`; no change
of scheme); everything else identical to the certified G2 deck; the G2 card
machinery with the G3 deck root, every G1 and G2 selector selected.

## 2. Deck switches that changed besides the initial state

**None.** Outside `&namtsd` the G3 deck's assignments equal G2's, which equal
G1's (committed test; the acquisition script refuses otherwise). Inside it:
`ln_tsd_init` moves to .true.; `cn_dir`, `sn_tem`, `sn_sal` appear with
ORCA2's values; `ln_tsd_dmp` stays .false.

## 3. FINDING: damping off also removes ORCA2's hand-made initial T/S edits

NEMO edits the ORCA2 climatology in the Alboran Sea and the Red Sea only
when damping data is on: `IF( nn_cfg == 2 .AND. ln_tsd_dmp )`
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dtatsd.f90:219`). With the operator's
pick, G3 does NOT start from ORCA2's real initial state: 1,283 temperature
and 720 salinity wet cells differ from it, the Red Sea's deep water by up to
20.5 °C (card, offline; the same counts the ORCA2 lane measured against
NEMO). With restoring off (`ln_tradmp = .false.`) the damping switch has no
other effect on the trajectory: dta_tsd is called each step only by the
restoring routine (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/tradmp.f90:181`).
Asked as DECISION_NEEDED; built as picked. The card follows NEMO's branch:
no edits when the deck's `ln_tsd_dmp` is false.

## 4. The record this acquisition gives, and its limit

- Binaries: G2's, unchanged (SHA-256 pinned; no build). `ln_tsd_init` is a
  namelist switch (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/istate.f90:117-118`).
- Runs: 2-step smoke (reference binary), then the 100-step reference, the
  100-step record (twin a, restarts every 10 steps), twin b (10 steps,
  restarts 2, 4, 6, 8, 10), twin c (10 steps, restarts 3, 5, 7, 9).
  G2 round 3 found a list of consecutive steps never opens; even and odd
  lists each do, so b and c together give every level at every step 2..10.
- **Limit.** Levels 2 to 30 are not witnessed at step 1 or at entry; level
  1 at entry is in the record binary's step-1 stage frame.

## 5. What is NEW, with no G1/G2 measurement behind it

Every density in G1 and G2 depended on depth only. With real T/S:

1. **The equation of state** on varying T, S: the in-situ density anomaly
   NEMO forms before the pressure gradient
   (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:137`, routine
   `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/eosbn2.f90:635-742`). Its polynomial
   evaluation order is the first candidate.
2. **The pressure gradient's downward recursion** with horizontal density
   differences, partial bottom cells included
   (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:397-413`).
3. **The baroclinic forcing of the barotropic solve**: the depth mean of a
   right-hand side that now holds a baroclinic pressure gradient
   (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:178-179`, certified on G2).
4. **N2** (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:160-161`). Read, not
   measured: under this deck its only reader is the mixed-layer depth
   (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/zdfmxl.f90:117`), which no prognostic
   reads (vertical mixing is constant). The record holds neither. Not scored.
5. **The initial state itself**: fldread's two-record weight at kt = 1 and
   the land mask after the copy (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dtatsd.f90:260-263`,
   `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dtatsd.f90:307-310`), through the ORCA2
   card's existing reader.

Splits ready for the first unequal cells: depth class (full / partial
bottom cell, the G2 ladder's level rows), and fold row / east-west seam /
south wall / coast / open interior (the G1 ladder's region rows).

## 6. Predictions (frozen before any G3 record exists)

- G3-P1 (log). NEMO's log states `ln_tsd_init = T`, `ln_tsd_dmp = F`, the
  release of the data arrays after the initial read, 31 levels, nn_e = 65.
  Falsifier: any of these absent (the script refuses).
- G3-P2 (initial state). Level-1 T and S in NEMO's step-1 entry frame equal
  the card's initial state on every wet cell, bit for bit. Falsifier: one
  unequal cell. (The ORCA2 lane matched the same reader against NEMO.)
- G3-P3 (first difference). If step 1 is not bit-identical, the first
  unequal field is level-1 U or V (the pressure gradient enters the 3-D
  right-hand side before the solve), and sea level and the barotropic
  velocities follow through the depth mean. Falsifier: sea level or the
  barotropic velocities unequal while level-1 U and V are equal.
- G3-P4 (tracers). Level-1 T and S after step 1 are bit-identical (no
  advection; the certified vertical diffusion). Falsifier: unequal T or S at
  step 1 while U and V are equal.
- G3-P5 (location). If a pressure-gradient statement is first, its first
  unequal faces include partial bottom cells. Falsifier: all first unequal
  faces in full cells only. PLAUSIBLE, no mechanism measured.
- G3-P6. No direction predicted for certification.

## 7. The card

ORCA2G3-ts: the G1/G2 builder on the G3 deck root (deck, shipped domain file
as domain_cfg.nc, the two climatology files). Its model configuration
equals G2's field for field (committed test); only the initial T and S
differ. A G3 deck with damping on, another file, another time step or start
date is refused (the reader's time weight is transcribed for this calendar
only). Declared not execution-ready (no record; density step unmeasured).
