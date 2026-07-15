"""Post-run ClimateEval invocation for AMIP CMOR output.

ClimateEval depends on iris/ESMValTool, which live in a separate
environment from legoESM's JAX stack (never a legoESM dependency). This
module bridges the two via ``subprocess`` — ``climateeval``/``iris`` are
never imported into this process.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from legoesm.driver.config import ExperimentConfig

logger = logging.getLogger(__name__)

# scripts/validate/run_amip_climateeval.py, relative to this package module
# (packages/coupler/legoesm/driver/ -> repo root is 4 parents up). Mirrors
# the precedent in atmosphere/forcing/sam_case_forcing.py::_LOCAL_FORCING_CACHE.
RUNNER_SCRIPT = (
    Path(__file__).resolve().parents[4] / "scripts" / "validate" / "run_amip_climateeval.py"
)


def maybe_run_climateeval(config: ExperimentConfig, output_dir: str) -> int | None:
    """Run ClimateEval against ERA5 for a completed AMIP run's CMOR output.

    No-op (returns ``None``) when ``config.output.evaluation.enabled`` is
    False. Non-fatal on failure: logs a warning and returns the
    subprocess's exit code rather than raising, so a broken or absent
    ClimateEval installation never fails an otherwise-successful,
    multi-hour AMIP run.
    """
    evaluation = config.output.evaluation
    if not evaluation.enabled:
        return None

    cmor_dir = Path(output_dir) / "cmor"
    cmd = [
        evaluation.climateeval_python,
        str(RUNNER_SCRIPT),
        "--cmor-dir", str(cmor_dir),
        "--model-id", evaluation.model_id,
        "--experiment-id", evaluation.experiment_id,
        "--variant-id", evaluation.variant_id,
        "--data-root-dir", evaluation.data_root_dir,
        "--output-dir", str(output_dir),
    ]
    # Empty suites -> let the runner discover + run ALL bundled suites.
    if evaluation.suites:
        cmd += ["--suite", *evaluation.suites]
    if evaluation.timerange:
        cmd += ["--timerange", evaluation.timerange]
    if evaluation.fail_on_missing_data:
        cmd.append("--fail-on-missing-data")
    if evaluation.download_missing_data:
        cmd.append("--download-missing-data")

    logger.info("Running ClimateEval: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.warning(
            "ClimateEval evaluation failed (exit=%d); the AMIP run output "
            "itself is unaffected. Command: %s",
            result.returncode, " ".join(cmd),
        )
    return result.returncode
