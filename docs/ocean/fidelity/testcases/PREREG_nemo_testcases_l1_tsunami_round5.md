# PREREGISTRATION — TSUNAMI lane, round 5 (build B4j: the j-periodic step)

Date 2026-10-08. Lane tip at start `4b991198b242`. Frozen before any
round-5 code or measurement exists. Evidence goes under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round5/`.

## 0. What was seen before this was written

Nothing was measured. Read only:

- NEMO's j-periodic exchange, single process (`ocean.output` lines 61-62:
  `jpni = jpnj = 1`; line 46 `nn_comm = 1` -> point-to-point exchange,
  `lbclnk.f90:203`). `l_SelfPerio(3:4) = l_Jperio .AND. jpnj == 1`
  (`mppini.f90:432`); every side with self-periodicity is filled
  `jpfillperio` (`lbclnk.f90:1868`); the southern halo copies the northern
  interior rows and the northern halo the southern interior rows
  (`lbclnk.f90:2026-2034`), by the plain assignment of
  `lbclnk.f90:2040-2045` — no sign. The sign `psgn` is used only by the
  north fold (`lbcnfd.f90:561`, the `'T','U','V','F'` cases at
  `lbcnfd.f90:584`, `lbcnfd.f90:639`, `lbcnfd.f90:684`), which runs only
  when `l_IdoNFold` (`lbclnk.f90:2111-2115`), and `l_IdoNFold` needs
  `l_NFold` (`mppini.f90:582`), false on TSUNAMI.
- So, per point type, on a doubly periodic domain: T, U, V and F halo rows
  are copies of the opposite interior rows, same index type, sign +1,
  offset exactly one period (Nj_0 rows). For legoESM's (ny+1) v-face
  layout this means the south face row 0 equals the north face row ny
  (the j analogue of the u-face column 0 = column nx closure the i-seam
  already uses).
- legoESM: the y-wrap exists as the process-global
  `halo_latlon.meridional_periodicity`; `pad_with_pole_bc_lat` wraps and
  `zero_polar_lat_ends` is a no-op under it; the face-mask builders open
  the boundary v-faces under it. The NEMO-literal split-explicit and qco
  metric statements build north/south neighbours with hard zero rows that
  never consult the flag, and the card's stored v-mask was built walled
  (rows 0 and 201 zero; read off the card, not a measurement of the step).

## 1. What will be built (card-selected; library default unchanged)

B4j: every j-neighbour statement on the TSUNAMI step's executed path routes
through the existing lat-axis pad (`pad_with_pole_bc_lat`, which already
wraps under the flag), and the card's masks open the j-seam v-faces when
`j_periodic`. With the flag off each rewritten statement reduces to its
current expression, so closed-basin cards are bit-identical by
construction, and that will be measured (section 2, P4).

## 2. Frozen predictions and falsifiers

- **P1 (unit, j-seam equivariance).** The round-2 strict xfail
  (`test_j_seam_is_translation_equivariant_bit_for_bit`) passes: one step
  of a bump straddling the j-seam equals the same bump moved into the
  interior, rolled back, bit for bit on eta, uu_b, vv_b. A plant that
  turns the j-wrap off makes it fail (nonzero count).
- **P2 (unit, transposition).** A bump symmetric under i<->j steps to
  fields with eta(j,i) = eta(i,j) bit for bit and uu_b/vv_b swapped under
  transposition with the f-plane rotation sign (u -> v, v -> u after the
  reflection i<->j reverses the rotation sense: test with f = 0 if the
  reflection is not a symmetry of the f-plane step). Falsifier: any
  unequal cell.
- **P3 (A/B on the 100-step record, INDEPENDENT label).** Arm OFF = the
  round-4 card (j-walled). Arm ON = the card with B4j. ON stays at the
  floor (normalised <= 1e-15, the trajectory gate's bar) on ssh, uu_b,
  vv_b, u, v at every kt = 1..100. Falsifier: any field over the bar at
  any kt; then the first such kt, field and cell are reported and the
  first non-bit statement is named with its NEMO citation. OFF must
  reproduce round 4 (first over the bar at kt = 15, ssh 2.7e-13 at
  (0, 39)); if it does not, the instrument is broken and P3 is void.
- **P4 (closed cards bit-identical).** The lane's cheapest certified-card
  gate (VORTEX / GYRE digests) gives the same verdict and the same digests
  before and after B4j. Falsifier: any changed digest that was not already
  red at the start commit.
- **P5 (kt = 1..10 ladder unchanged).** Given-NEMO-entry rhs stays at
  1.09e-19 (kt = 8: 1.63e-19) and the kt = 1..10 rows are unchanged to the
  bit (the j-seam is not reached before kt = 14). Falsifier: any row that
  moves.

## 3. What happens on each outcome

- P3 holds: TSUNAMI AT THE BAR for the key_RK3 deck; re-run the whole
  ladder (both labels) and the geometry gate as the certification set,
  register every row, write the ORCA2 pointer section.
- P3 fails: name the first non-bit statement (cited), keep B4j if P1, P4
  and P5 hold (it is NEMO's statement), status HELD on the record.
