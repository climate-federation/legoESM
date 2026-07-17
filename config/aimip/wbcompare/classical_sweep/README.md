# Classical physics-combo sweep (WB2 + AIMIP)

Five CLASSICAL scheme combinations, each trained by the SAME run_aimip
`classical` variant (spectral primitive-equation dycore + RRTMGP radiation)
but with a different convection / turbulence / gwd / microphysics / cloud
combination. Every combo is scored on BOTH benchmarks:

  * the AIMIP-protocol scorecard written by `run_aimip` to the combo's
    `output_dir` (`aimip_scorecard.json`);
  * the WeatherBench-2 headline scorecard produced by the eval bridge
    `scripts/validate/run_aimip_wb2_eval.py`.

## Why one directory per combo (the merge-order trap)

`run_aimip`'s suite loader merges **shallow, in this order** (verified against
`scripts/run/run_aimip.py` L840-882):

    base  <-  suite.cfg_overrides  <-  variant_<variant>.yaml

with the overlay loaded from **the suite file's OWN parent directory**:

    overlay = _load_yaml(args.suite.parent / f"variant_{variant}.yaml")   # L879-880

The WB2 eval bridge reconstructs the identical merge
(`run_aimip_wb2_eval.py::merged_cfg_from_suite`, L161:
`Path(suite_path).parent / f"variant_{variant}.yaml"`).

Because the `variant_classical.yaml` overlay is merged **LAST**, any scheme it
sets WINS over `cfg_overrides`. If all combos shared the wbcompare
`variant_classical.yaml` (which hardcodes the winner schemes), every combo
would silently train the WINNER combo — defeating the sweep.

**Fix:** give each combo its OWN directory holding both `suite.yaml` and a
per-combo `variant_classical.yaml` that carries THAT combo's schemes. The
loader (train + eval) resolves `variant_classical.yaml` relative to the suite's
parent, so `.../<combo>/suite.yaml` picks up `.../<combo>/variant_classical.yaml`
— the combo's own schemes win the merge.

## Combos

| dir       | convection | turbulence  | gwd       | microphysics | cloud      | radiation |
|-----------|------------|-------------|-----------|--------------|------------|-----------|
| winner    | edmf       | louis       | mcfarlane | sundqvist    | xu_randall | rrtmgp    |
| tiedtke   | tiedtke    | louis       | mcfarlane | sundqvist    | xu_randall | rrtmgp    |
| bechtold  | bechtold   | louis       | mcfarlane | kessler      | xu_randall | rrtmgp    |
| sbm_ysu   | sbm        | ysu         | hines     | morrison     | sundqvist  | rrtmgp    |
| edmf_tke  | edmf       | tke         | mcfarlane | thompson     | xu_randall | rrtmgp    |

All combos: `aimip_variant=classical`, `aimip_radiation=rrtmgp`,
`aimip_n_epochs=12`, `eval_years=[2020]`, and the shared ACE2 loss block copied
from `config/aimip/wbcompare/suite.yaml`.

## Launch

Ginsburg:  `sbatch scripts/cluster/wb_forecast/classical_sweep.sbatch`
Derecho:   `qsub scripts/cluster/derecho/run_wb_classical_sweep.pbs`

Both loop the combos (train -> AIMIP scorecard -> WB2 eval -> comparison plot).
Subset with `WB_COMBOS="winner,tiedtke"`; eval-only with `WB_SKIP_TRAIN=1`.
