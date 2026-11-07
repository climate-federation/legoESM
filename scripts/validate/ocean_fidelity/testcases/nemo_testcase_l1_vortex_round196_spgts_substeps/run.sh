#!/usr/bin/env bash
set -Eeuo pipefail

# Bound entry point for the round-196 barotropic substep acquisition.  All
# build, admission, passivity and plant logic remains in the one shared VORTEX
# harness; this launcher only supplies the preregistered variant and run arm.
round196_launcher_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
exec "$round196_launcher_dir/../nemo_testcase_l1_vortex/run.sh" \
  --variant spgts --run "$@"
