# Grid ladder rung G2 — round 1 preregistration (gridladder_r10)

Base: 92435ddbd (rung G1 certified, rounds 1-9). No NEMO record of G2 exists;
nothing in this file was measured against NEMO. Evidence directory:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round10/.
NEMO citations are to the G1 build's compiled sources under
ORCA2G1_OMIP_L1_P3; the seven files cited are byte-identical in the two
builds the acquisition reuses (ORCA2G1_OMIP_L1_R4REF, ORCA2G1_OMIP_L1_P5;
compared with cmp this round).

## 1. The rung, and the operator's picks inside Decision 117

G2 = the certified G1 program with ONE switch, the domain.

| item | G1 (certified) | G2 |
|---|---|---|
| domain file | built by NEMO's domain tool: flat 100 m, 2 levels | ORCA2's shipped file, copied byte for byte: 31 levels, partial bottom cells |
| path | round 4's domaincfg/domain_cfg.nc | /data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/ORCA_R2_zps_domcfg.nc |
| SHA-256 | 73801c6f...deabdd | 7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839 |

Operator picks (the user may redirect): the shipped file used directly, no
domain-tool run; everything else identical to the certified G1 deck.

Everything else, stated: uniform T = 20 and S = 30; the same sea-level bump
formula at (ji,jj) = (51,74); zero forcing; no ice; momentum and tracer
advection OFF; lateral diffusion OFF; drag OFF; constant vertical mixing;
ORCA2's split-explicit settings; time step 10800 s; 100 steps; the same
record patches.

## 2. Survey of the shipped file (offline, read this round)

| fact | value |
|---|---|
| dimensions | x 180, y 148, z 31 |
| wet columns | 16,433 of 26,640; the same coast as G1 (mask equal) |
| wet levels per column | 3 to 30 |
| columns whose bottom cell is a partial cell | 15,554 of 16,433 |
| thinnest bottom cell | 1.31 m (0.041 of its level's full thickness) |
| cells above the bottom differing from the 1-D ladder | 0 |
| depth at the bump centre | 5,312 m, 30 wet levels |
| deepest / shallowest column | 5,500 m / 30.0 m |
| horizontal mesh, 16 fields | bit-identical to G1's file (round 1 receipt) |
| Coriolis fields | differ from G1's file by at most 2.7e-20 |
| extra variable | strait_shlat, values -1, 0.5, 1 |
| scalars G1's file has and this one lacks | jperio, jpiglo, jpjglo, jpkglo, ln_zco, ln_zps, ln_sco, ln_isfcav (global attributes instead) |

- NEMO sizes the domain from the file, not from the build
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domain.f90:581`), and takes the
  coordinate type from the file's attribute
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domzgr.f90:130-132`). The compiled keys
  are G1's. So the G1 binaries serve: no build.
- With this build's keys NEMO reads the four 3-D thicknesses
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domzgr.f90:185-188`); depths and the
  interface thickness stay on the 1-D ladder. Partial cells therefore enter
  through cell and face thicknesses and masks only. The density's depth is
  the 1-D depth times the stretch
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/eosbn2.f90:324`), the same at a partial
  cell as at a full one.
- Barotropic substeps: automatic, as in G1's deck. Offline, in NEMO's order:
  65, Courant number 0.7980181110475545. NEMO's own ORCA2 log (same file, same
  time step) prints 65 and 0.79801811104755460. G1: 13 and 0.7812.

## 3. Deck switches that changed besides the domain

**None.** The G2 deck's 129 assignments equal G1's (committed test; the
acquisition script refuses otherwise). Only the header and three comments
differ. NEMO needs no further switch for 31 levels: every vertical option was
already stated in G1's deck.

Three things change with the file and are NOT deck switches:

1. The substep count resolves to 65 instead of 13 (automatic, from depth).
2. The file carries strait_shlat. NEMO applies it when present
   (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dommsk.f90:289`): the vorticity mask at
   Gibraltar becomes 0.5 and at Bab el Mandeb 1, instead of 2 (the file's
   own history attribute names the two straits). This rides with the operator's pick of the shipped file. With
   momentum advection and lateral mixing OFF I know no reader of it in this
   program (PLAUSIBLE, read not measured). It is scored by the geometry row.
3. The Coriolis fields are this file's, not G1's (2.7e-20 apart).

Kept on purpose: the experiment name stays ORCA2G1, because the reused
binaries and the reused record checker name files by it.

## 4. The record this acquisition gives, and its limit

- Reference binary: 2-step smoke, then 100 steps. Record binary: 100 steps
  (twin a) and 10 steps (twin b). Stage frames at steps 1 to 50, final frame
  every step, solver substeps at steps 1, 2, 17, 31, 41.
- **Limit.** The record binary writes 2-D fields and LEVEL 1 of U, V, T, S.
  Levels 2 to 30 are only in NEMO's restart files: steps 10, 20, .., 100.
- One added output (UNASKED, offered for revert): twin b writes restarts at
  steps 2, 4, 6, 8, 10. The script judges that twin b's step-10 restart equals
  twin a's, so the added writes change nothing.
- Named, not built: a writer for all levels in the stage frames needs a new
  build. Asked for only if the walk cannot name a statement without it.

## 5. The card

ORCA2G2-zps is the G1 builder on the G2 deck root. Model configuration against
G1's, field by field: one field differs, the resolved substep count. Every G1
selector stays selected (the G1 selector check passes on the G2 card).

Changes to shared G1 code, each needed for more than one level:

| code | G1 form | now | effect on G1 |
|---|---|---|---|
| substep helper | column depth summed over all file levels of a wet column | summed over wet levels only | none (one level) |
| validator: T, S, thickness | level 1 checked | every level checked | none |
| domain reader | refuses anything but 2 levels | refuses anything but the rung's level count | none |
| strait mask input | none | the file's, or none when absent | none (G1's file has none) |

Declared not execution-ready: no record; multi-level partial-cell step
unmeasured.

## 6. Predictions (frozen before any G2 record exists)

Which G1-certified statements G2 exercises at full depth:

| statement (G1 round) | one level | 30 levels |
|---|---|---|
| density depth rounded before the scale (9) | one depth | 30 depths, to 5,250 m |
| stage T, U, V, F ratios interpolated (7, 9) | one thickness each | the same 2-D ratio on 30 thicknesses per column |
| vorticity term in written order (7) | one level | every level, with partial-cell thickness at the bottom |
| barotropic part removed after vertical mixing (7) | removes the whole velocity | removes a true vertical mean |
| stage-1 right-hand side handed whole (6) | one level | 30 levels |
| fold exchanges (2, 3, 6, 7) | one level | every level |
| solver statements (5, 6, 8) | 13 substeps, flat depth | 65 substeps, real depth |

What is new, with no G1 measurement behind it:

- the pressure gradient's downward recursion, levels 2 to 30
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:397-413`); G1 measured only
  the surface statement (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:380-394`);
- the solver's forcing as a 30-level thickness-weighted sum
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:178`), the bottom boundary of
  the barotropic and baroclinic split;
- bottom levels: T from the file
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domzgr.f90:575`), U and V as the
  shallower neighbour (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domzgr.f90:588-589`);
- the implicit vertical solves on real columns of 3 to 30 levels, the bottom
  one a partial cell, then the barotropic removal
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:160-161`);
- round 9's two open readers (stage transport geometry; the vertical solve's
  interface thickness), which had nothing to act on with one level.

Predictions:

- G-P1 (geometry). Every geometry row against NEMO's mesh file is equal:
  masks at T, U, V, F on 30 levels, bottom levels, the four thicknesses, depth
  sums, Coriolis. Falsifier: any unequal row. An unequal vorticity mask at the
  strait points would falsify item 2 of section 3 as handled.
- G-P2 (NEMO's log). 31 levels; 65 substeps; Courant number 0.7980 to four
  digits; the bump centre (51,74) at longitude 179.99992, latitude 0. Falsifier: any other value.
- G-P3 (NEMO runs). The 2-step smoke and the 100-step runs finish without an
  error line. Falsifier: NEMO stops. Then the deck needs a switch and section
  3 is wrong.
- G-P4 (twins). Twin a and twin b are byte-identical over steps 1 to 10, and
  the record build's restarts equal the reference build's at steps 10 and 100.
  Falsifier: any difference.
- G-P5 (T and S are not uniform after step 1). With advection OFF the tracer
  update divides by the new thickness, so T leaves 20 wherever the sea level
  moved, by the same factor at every level of a column. Falsifier: NEMO's
  level-1 T is exactly 20 on every wet cell after step 1.
- G-P6 (step 1, stage 1, level 1). The stage-1 right-hand side at level 1 is
  bit-identical: it is the G1-certified surface statement. Falsifier: an
  unequal face.
- G-P7 (**the first row that is not bit-identical**, PLAUSIBLE). Step 1, the
  solver's entry forcing (the substep record's first frame), on faces inside
  the bump (201 T cells around (51,74)). Reason: it is the first quantity that
  holds the 30-level pressure-gradient recursion and the vertical sum over
  partial-cell faces, and neither has been scored on this card. Falsifier:
  that frame is bit-identical. Then the next candidates, in NEMO's order: the
  stage-1 level-1 velocity, then stage 3's vertical solve.
- G-P8 (no direction). Whether step 1's final sea level is bit-identical: not
  predicted.
- G-P9 (tiny numbers). The registered G1 environment class reappears: NEMO
  keeps numbers below the smallest normal double at the edge of the signal,
  the card holds zero. Falsifier: no such cell in 100 steps.

Not predicted: the step at which any field first leaves the 1e-15 bar.

## 7. Choices

ASKED (round brief): shipped file directly; G1 deck otherwise; binary reuse;
smoke first; twins and the 100-step record; G1 card machinery; this file.

UNASKED, each offered for revert:
1. Twin b's restarts at steps 2, 4, 6, 8 (section 4).
2. The experiment name left as ORCA2G1 (section 3).
3. The card's case name, ORCA2G2-zps.
4. The four shared-code changes of section 5.
5. Two older line pins of the citation gate re-pinned to the edited lines.
