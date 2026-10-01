# ORCA2 hierarchy decks round 4 preregistration — rung 8 admission and rung 7 internal-wave-off deck

Date: 2026-10-01

Base: `9599298bfd` (`fidelity/nemo-testcases-orca2-decks`).

Scope is NEMO-side only.  Round 3's frozen HD3-P3 and HD3-P5 predictions
govern admission of the operator-produced rung-8 record.  This round then
constructs hierarchy rung 7 by removing internal-wave-driven mixing and the
background reset that only its initializer performs.  It changes no legoESM
package, card, recipe, physics, configuration, threshold, or carried state.
Rungs 1 through 6 and main-lane rung 0 remain out of scope.

All record claims are labelled **independent**: NEMO starts from each rung's
own from-rest T/S initialization.

## Frozen oracle reading

The compiled vertical-physics manager reads `ln_zdfiwm` with the other
vertical-physics selectors and backgrounds
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:140-149`).
It initializes `avmb=rn_avm0`, `avtb=rn_avt0`, and the configured horizontal
tracer-background shape at `zdfphy.f90:205-228`.  The manager calls the
internal-wave initializer only when `ln_zdfiwm` is true and likewise calls the
per-step wave increment only in that arm (`zdfphy.f90:262-284,330-372`).

The internal-wave initializer overwrites those backgrounds with molecular
values `avmb=1.4e-6`, `avtb=1e-10`, and a uniform horizontal shape only when it
runs (`zdfiwm.f90:438-447`).  Therefore the false arm leaves the retained rung-8
namelist values `rn_avm0=1.2e-4`, `rn_avt0=1.2e-5`, `nn_avb=0`, and
`nn_havtb=1` in force.

The compiled TKE initializer reads `rn_emin=1e-6` from the inherited namelist.
It forces `rn_emin=1e-10` and `rmxl_min=1e-3` only when internal-wave mixing is
on.  In the false arm it retains `rn_emin=1e-6` and computes
`rmxl_min=1e-6/(rn_ediff*sqrt(rn_emin))=1e-2` for inherited `rn_ediff=0.1`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdftke.f90:751-783,835-856`).

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD4-P1 rung-8 admission | Round 3's operator record satisfies every frozen selector, completeness, and resolved-consequence predicate. | The committed round-3 gate passes normally and all nine record-stage plants fire. | Any plant stays green or any record predicate fails. |
| HD4-P2 one-module deck delta | Rung 7 differs from rung 8 in exactly one parsed assignment and physical line: `namzdf.ln_zdfiwm`, true to false. | Complete line and assignment-map gates report only that delta; every other assignment, build pin, input, and protocol value is inherited. | Any other assignment, physical line, CPP key, input, or run-protocol difference. |
| HD4-P3 internal-wave arm absent | Rung 7 does not initialize or execute internal-wave mixing. | Resolved output reports `ln_zdfiwm=false`, contains no internal-wave initializer report or reset message, and the source-path gate pins both guarded calls. | Either internal-wave initializer or per-step increment executes. |
| HD4-P4 inherited backgrounds restored | With the internal-wave reset skipped, rung 7 resolves `rn_avm0=1.2e-4`, `rn_avt0=1.2e-5`, `rn_emin=1e-6`, and `rmxl_min=1e-2`; the equatorial `nn_havtb=1` tracer-background shape remains selected. | The exact retained assignments, compiled branches, and resolved output all agree with those values and the standard false-arm mixing-length message. | Any value or background-shape selector differs, or the internal-wave reset message remains. |
| HD4-P5 retained wave namelist inert | The retained `namzdf_iwm` group and its six forcing descriptors are not read when `ln_zdfiwm=false`. | A committed syntactically invalid `namzdf_iwm` execution sentinel still reaches `STOP 0`, while the ordinary deck remains byte-retained outside the one selector. | NEMO parses the sentinel or any compiled unguarded call to `zdf_iwm_init` is found. |
| HD4-P6 record completeness | The reused two-rank instrumented binary produces 480 finite self-describing operand frames, finite T/U/V/W month products, finite fp64 step-240 ocean restarts, and a complete SHA-256 inventory. | Header-driven parsing reaches physical EOF for every rank-step frame and all product/restart/inventory checks pass. | Missing or malformed frame, bad payload, non-finite value, wrong step/dtype, or incomplete ledger. |
| HD4-P7 acquisition disposition | No rung-7 record exists before the operator run. | Preflight and every synthetic violation pass/fire; round ends `STOPPED_FOR_RECORD` with a committed launcher. | An already-existing admissible rung-7 record is found. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and landing predicate

The rung-7 gate must refuse a second deck delta, a missing selector delta, a
changed retained background, a wrong build pin, a readable invalid
`namzdf_iwm` sentinel, malformed or truncated self-describing payloads, a
missing/non-finite frame, a non-finite terminal state, a wrong terminal step,
a wrong resolved consequence, and an incomplete SHA-256 inventory.  Every
plant must fire.

This round lands rung-8 admission evidence plus the rung-7 deck, manifest,
gate, launcher, tests, and receipt only if rung 8 admits and rung-7 preflight
and plants pass.  Rung 7 remains **UNMEASURED** until its operator-run
acquisition is admitted.  No GYRE run is required because no model file
changes.
