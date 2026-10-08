# ORCA2 round 184 preregistration — atomic HPG fold/depth-average unit

Date: 2026-10-08. Base: `34fab9763c5a656632c1b0e21d866a767fe30e79`.
Claim label: **independent hierarchy rung 0** for rung-0 measurements and
**given NEMO's entry** for shipped rung-7 measurements. The two labels will
not be mixed in one table.

## Frozen scope

Round 183 proved that the first non-bit executable HPG statement is the
north-fold surface density product, but that raw HPG, reference `e3v`,
`vmask`, and `r1_hv0` form one cancelling unit. This round measures exactly
one private atomic arm containing all four NEMO statements:

1. the north-fold density neighbour consumed by the surface and interior V
   HPG products (`ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:409-416,445-453`);
2. the reference V thickness and its mask (`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/domain.f90:193-200` and
   `ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dommsk.f90:206-232`);
3. the stored reciprocal (`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/domain.f90:212-215`);
4. their one written depth average (`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215`).

The arm is private and indivisible. It makes no configuration choice, changes
no carried state, adds no stabiliser, and does not touch the shipped ORCA2
card's sea-ice selectors or `unmeasured_features`.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R184-P1 | The four arm operands are the admitted NEMO operands. | On the registered 68 northern V faces, north-fold HPG closes bit-for-bit; reference `e3v` closes 27 unequal cells, `vmask` closes 1,319, and `r1_hv0` closes 68. | Any registered operand remains unequal: **REFUTED**, hold and name that operand. |
| R184-P2 | The atomic depth-average endpoint is exact. | The completed V slow forcing closes 68/68 faces bit-for-bit against the admitted record. | Any face remains unequal, or any partial arm can be selected: **REFUTED**, hold. |
| R184-P3 | The atomic unit is a Decision-96 net improvement on both ORCA2 ladders. | First-over-bar row is toward or unchanged; a majority of moved rows is toward NEMO by RMS; no exact row is lost beyond the floor; rung-0 and rung-7 endpoint sea-surface maximum does not worsen. | Any predicate fails: keep the arm private, **HELD**, name the first worsened row and do not run landing gates. |
| R184-P4 | The private arm is non-vacuous and indivisible. | It moves the registered V slow forcing; a planted omitted operand and a planted endpoint ULP both make the classifier refuse. | A plant stays green: instrument invalid; quote no ladder number. |
| R184-P5 | If eligible to land, unaffected cards satisfy the standing shared gates. | GYRE passes its Decision-43/45/55/59 gate, DINO and both tanks remain inside their registered predicates, citations are mapped, and focused plus fidelity batteries have no new red. | Any shared gate fails: **HELD**; no production landing. |

The first ladder measurement is performed only after this file is committed.
Failed predictions remain in the receipt. No acquisition is preregistered:
rounds 170, 180, and 182 already contain the required rank-complete operands.
