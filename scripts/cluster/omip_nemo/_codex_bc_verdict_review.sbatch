#!/bin/bash
#SBATCH --job-name=cdxverd
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:25:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdxverd_%j.out
#
# Adversarial review of the CLAIM, before any production default is proposed.
# The claim is that a surface-turbulence boundary-condition flag causes an
# equatorial SST regression, read off a pre-registered band. codex is asked to
# break the inference, not to agree with it.
set -uo pipefail
cd /burg-archive/glab/users/pg2328/_omip_merge || exit 1
codex exec --sandbox read-only -C /burg-archive/glab/users/pg2328/_omip_merge 'ONE REVIEW, no follow-up. Attack this inference; do not agree with it. An ocean model is compared against a NEMO oracle over 30 days. A config flag selects where a turbulence closure holds its surface boundary value: value A (nemo_z0) matches the oracle code, value B (interior_pinned) does not. Two runs: SAME commit, SAME command line, the ONLY difference is that flag, verified by a path-scoped git diff --stat over packages/ src/ scripts/run/ scripts/validate/ between the two commits, which is empty. Both scored by one tool, one 42254-cell set, matched day 30. Equatorial Pacific SST bias: A +1.3396, B +1.1354. A cross-grid noise floor of 0.060 was estimated earlier from three independent meshes at matched days. BEFORE the run I pre-registered: B <= +1.15 means the flag causes the regression; B >= +1.29 means the flag is exonerated; between means no verdict. B landed at +1.1354 so I concluded the flag CAUSES it. Mixed-layer depth is unchanged between the arms (+0.87 vs +0.90 m). Name the strongest reasons this conclusion is wrong. Specifically: is a single-realization difference of 0.204 against a noise floor derived from a DIFFERENT source of spread (three meshes, not two seeds of one model) a valid significance test? Does one arm on one day license a causal claim at all? Does the unchanged mixed-layer depth argue for or against a physical mechanism, and could it indicate an implementation bug in the solve rather than a physics difference? Is there any way the flag changes something other than what I think it changes? Under 400 words, numbered points, harshest first.' 2>&1 | tail -80
rc=${PIPESTATUS[0]}
echo "=== codex exit: $rc ==="
exit $rc
