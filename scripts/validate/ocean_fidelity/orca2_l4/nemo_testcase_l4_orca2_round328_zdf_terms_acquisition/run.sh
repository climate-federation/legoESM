#!/usr/bin/env bash
# Thin target selector for the shared, operator-reviewed round-328 acquisition.
set -Eeuo pipefail

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
export R328_ACQ_TARGET=zdfphy
exec "$here/../nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh" \
  "${1:---preflight-only}"
