# Shared checkout binding for the SWG_DG shell entry points.

swg_dg_bind_checkout() {
  local python=$1 here repo_root package_path
  here=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
  repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || {
    printf 'REFUSE_NOT_WORKTREE: tools are outside a Git checkout\n' >&2
    return 66
  }
  package_path=$repo_root/packages/core:$repo_root/packages/ocean:$repo_root/packages/atmosphere:$repo_root/packages/coupler:$repo_root/packages/ice:$repo_root/packages/land:$repo_root/packages/ml:$repo_root/packages/tools:$repo_root/src
  export PYTHONPATH=$package_path${PYTHONPATH:+:$PYTHONPATH}
  JAX_PLATFORMS=cpu "$python" -c \
    'import sys; from legoesm.ocean.fidelity.provenance import require_checkout_import; require_checkout_import(sys.argv[1])' \
    "$repo_root"
}
