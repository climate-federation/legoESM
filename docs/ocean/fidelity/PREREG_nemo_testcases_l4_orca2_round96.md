# Preregistration — ORCA2 round 96 rung-0 SPG acquisition repair

Date: 2026-10-02. Base: `90abce744`. Scope is ocean only. Any model number
eventually read from this acquisition is labelled **independent** because the
rung-0 state is NEMO's own from-rest state, not the shipped-card recorded-entry
twin.

## Prior evidence and frozen diagnosis

The operator ran round 95's committed launcher. It stopped at launcher line
210 before the instrument patch, custom rebuild, run-directory creation, or
MPI launch. The failed command applied the Decision-83 patch, which was made
against the admitted round-93 run namelist (SHA-256 `d25c69958...`), to the
configuration directory's stale `EXP00/namelist_cfg` (SHA-256 `c7d350404...`).
The admitted deck and config copy differ in the rung-0 module switches; two of
five patch hunks therefore refused. This is a launcher provenance defect, not
an ocean measurement. The partially created `ORCA2_OMIP_L4_R95SPG` target is
never reused.

## Frozen repair and measurements

1. The repaired launcher uses a new NEMO target and run-directory name. After
   cloning the source config it explicitly stages the admitted round-93
   `namelist_cfg`, pins its digest, then applies Decision 83 and pins the
   harmonized digest. It does not derive a deck from the stale config copy.
2. Preflight must syntax-check the same patched source used by the target and
   must fire the layout plant.
3. The operator-run acquisition must retain both rank-tagged self-describing
   SPG records and all twenty terminal restart shards. Admission requires the
   restarts to be byte-identical to round 93 and every record plant to fire.
4. Once admitted, the independent rung-0 substeps are walked in compiled
   order to the first active non-bit statement. If the walk reaches
   `dyn_cor_2D`, NEMO's four-cell masked `e3f_0vor` is tested as the already
   preregistered one-variable cross-operator arm.

## Predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R96-P1 | Round 95 produced no record and no custom binary. | Its evidence directory is empty; the target source is unpatched; the log has no MPI launch and refuses at line 210. | Any rank stream or custom compiled writer exists: **REFUTED**; inspect and admit only if its producer and passivity are sound. |
| R96-P2 | Staging the admitted namelist before applying Decision 83 repairs the launcher without changing the scientific deck. | Source digest and harmonized digest both pin, all five patch hunks apply at fuzz zero, and preflight passes. | A pin or hunk fails: **REFUTED**; stop and do not build or run. |
| R96-P3 | The writer is observational. | Twenty target restarts are byte-identical to round 93. | Any restart moves: **REFUTED**; withhold the record and repair the instrument. |
| R96-P4 | Round 95's first-boundary predictions remain unmeasured until the new record exists. | No ocean number is claimed from preflight or the failed target. | Any claim uses the failed target: **REFUTED** and retracted. |

## Landing and refusal bar

This repair changes no `packages/` file and lands no ocean statement. It makes
no selector, configuration, carried-state, stabilizer, sea-ice, threshold, or
`unmeasured_features` change. A later model landing still requires the ORCA2
rung-0 ladder and every shared-card gate. The held round-94 slow-depth patch is
not mixed into this acquisition.
