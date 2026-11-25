#!/usr/bin/env bash
set -Eeuo pipefail
export R328_ACQ_TARGET=tke331
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
exec "$here/../nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh" "$@"
