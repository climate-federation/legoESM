#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_rev
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_review_%j.log
# The SECOND adversarial reviewer (standing dual-review policy, main a6060d41d).
# GLM is reached on the CODING endpoint with ZAI_API_KEY -- see the module
# docstring of scripts/validate/glm_review.py for why the other key and the
# other endpoint both fail, and how.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_fv3jax_run7
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
echo "host: $(hostname)"
cd $REPO && echo "PINNED sha: $(git rev-parse HEAD)"
$PY $REPO/scripts/validate/glm_review.py \
    --prompt-file $G/prompt_strategy.txt \
    --model glm-5.2 --max-tokens 40000 \
    --out $G/review_strategy.md
echo "rc=$?"
echo GLM_REVIEW_DONE
