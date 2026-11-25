# PREREG — ORCA2 round 312: D126 surface-only runoff; D135 V1 TKE features under D122

Frozen before any code change or measurement (Claude round; codex review: paused (quota)).
Base `f075a5cd5` (round 311 landed).  CPU, fp64, libm, production JIT.

User decisions (2026-10-17, "yes to your picks"):

- D126: the ORCA2 card puts runoff at the surface only (`runoff_depth_spread_m = 0.0`),
  as NEMO's ORCA2 deck (namelist_ref `ln_rnf_depth = .false.`, `ln_rnf_depth_ini = .false.`;
  sbc_rnf_init sets `nk_rnf = 1`; sbc_rnf_div takes the surface-only branch).  Read off the
  round-311 resolved configs: only rung 10 resolves 150 m (inherited from the generic NEMO
  recipe default); rung 0 and OMT-0..4 already resolve 0.0.
- D135: rung 10 imports V1's certified TKE forms (gridladder-vmix `0b9443cef2`, V1 card
  `_orca2v1_tke_vertical`), never re-implemented:
  - ETAU group: `etau_mode="below_ml"` (nn_etau=1), `etau_htau_mode="latitude"` (nn_htau=1,
    namelist_ref), `nemo_etau_wmask`, `tke_htau_evaluation="nemo_scalar_libm"`,
    `tke_etau_exponential_evaluation="nemo_scalar_libm"`.
  - MXL group: `tke_mxl_raw_evaluation="nemo_literal"` (literal raw length) and
    `nemo_mxl_terminal_e3t` (dummy-bottom step; the code refuses it without the literal raw length).
  - LC: `tke_langmuir_evaluation="nemo_literal"`.
  - `nemo_mxlice = 2` stays 0 (needs the card's own ice thickness).

Inputs the imported forms read and rung 10's builder does not yet carry (read off the code):
`grid.native_lat_T_deg` (native gphit, literal htau) and `z_coord.nemo_e3t_jpk` (the
domcfg's level-jpk thickness, terminal step).  The import commit adds both to the ORCA2
builder; each is consumed only under its flag.

Arms on rung 10, both labels (control = the round-311 shipped card):
RNF; ETAU_CORE (below_ml + latitude); ETAU_WMASK / ETAU_HTAU / ETAU_EXP (core + one);
ETAU (all five); MXLRAW; MXL (raw + terminal); MXLTERM (terminal alone); LC; TKE (ETAU+MXL+LC); ALL (TKE+RNF).
Seven pairs: control vs D122 (the selection tree) on rung 0, OMT-1..4, rung 10 both labels.

| ID | Prediction | Falsifier |
|---|---|---|
| R312-P1 | On the import tree every shipped-card control reproduces round 311's `final/` D122 rows row for row (7 pairs). | any row differs. |
| R312-P2 | RNF moves >= 1 row on rung 10 under both labels, loses 0 exact rows, first over-bar row unchanged. | 0 rows moved, an exact row lost, or first over-bar away. |
| R312-P3 | ETAU moves >= 1 row on rung 10 under both labels and passes D122 (c)+(d). | 0 rows moved, or (c)/(d) fails, or a refusal. |
| R312-P4 | ETAU_HTAU and ETAU_EXP each differ from ETAU_CORE in >= 1 row (libm vs vectorised transcendental). ETAU_WMASK: no size prediction (registered). | either is row-identical to ETAU_CORE. |
| R312-P5 | MXLTERM refuses at build/trace (guard: terminal step requires the literal raw length); MXLRAW and MXL each move >= 1 row and pass D122 (c)+(d). | MXLTERM runs; MXLRAW/MXL move 0 rows or fail (c)/(d). |
| R312-P6 | LC moves >= 1 row and passes D122 (c)+(d). | 0 rows moved, or fails. |
| R312-P7 | TKE and ALL pass D122 (c)+(d) on rung 10 both labels; the seven-pair D122 decision passes. | any exact row lost / first over-bar away / decision fails. |
| R312-P8 | Selection-tree resolved configs vs base: rung 10 differs only in runoff depth and the selected TKE rows; rung 0 and OMT-0..4 identical. GYRE, DINO, VORTEX step jaxprs identical to base. | any other row, or any jaxpr line, differs. |

Landing: D126 lands (user decision) if R312-P2's (c)/(d) holds; otherwise STOPPED_FOR_DECISION.
Each D135 group lands on rung 10 if its row passes D122 (c)+(d) under both labels; groups
that fail stay off with their numbers.  Census toward/away registered, never a veto.
Not touched: G8 fold rows (D137 off), double diffusion (waits V4), G6 fold statements (wait G9).
