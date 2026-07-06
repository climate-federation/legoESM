#!/bin/bash
# ===========================================================================
# Build aws-ofi-nccl (the NCCL <-> libfabric/CXI plugin) on NCAR Derecho.
#
# WHY: NCCL has no native Slingshot-11 support. Without this plugin every
# cross-node jax.distributed / shard_map collective falls back to TCP
# sockets over hsn (2-3x slower comm; ALCF measured 2-3x end-to-end on the
# same stack). With it, NCCL drives the Cassini NICs through libfabric
# ("Using network AWS Libfabric").
#
# RUN ON A DERECHO LOGIN NODE (no GPU needed to BUILD; ~5 min):
#   bash scripts/cluster/scaling_derecho/build_nccl_ofi.sh
#
# Then submit the multi-node job with the printed LEGOESM_NCCL_OFI_LIB:
#   qsub -v LEGOESM_NCCL_OFI_LIB=<printed lib dir> \
#        scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs
#
# Knobs (env overrides):
#   AWS_OFI_TAG     git tag to build (default: newest release tag)
#   PREFIX          install prefix (default /glade/work/$USER/nccl-ofi/<tag>)
#
# NOTE: modern aws-ofi-nccl (>= v1.8 era; verified against v1.20.0
# configure.ac) has NO --with-nccl — the plugin vendors the NCCL net-API
# headers and is dlopen'd AT RUNTIME by whatever NCCL the jax[cuda12]
# wheels ship (nvidia-nccl-cu12). Build deps are only libfabric + CUDA
# (+ hwloc).
#
# Reference recipe: github.com/benkirk/derecho-pytorch-mpi
# (utils/build_nccl-ofi-plugin.sh) — this script is the legoESM-shaped
# equivalent.
# ===========================================================================
set -euo pipefail

# Only gcc + cuda (for nvcc/headers) are needed — no conda env, no NCCL
# (see header note: the plugin has no NCCL build dependency).
module load gcc cuda 2>/dev/null || true

# --- Locate the pieces --------------------------------------------------
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
echo "CUDA_HOME=$CUDA_HOME"

# libfabric: Derecho's Cray-provided CXI-enabled install.
LIBFABRIC_HOME="${LIBFABRIC_HOME:-}"
if [ -z "$LIBFABRIC_HOME" ]; then
    for d in /opt/cray/libfabric/*/; do LIBFABRIC_HOME="${d%/}"; done
fi
[ -f "$LIBFABRIC_HOME/include/rdma/fabric.h" ] || {
    echo "ERROR: libfabric headers not found (looked in /opt/cray/libfabric/*)." >&2
    echo "       module load libfabric, or set LIBFABRIC_HOME." >&2
    exit 1
}
echo "LIBFABRIC_HOME=$LIBFABRIC_HOME"

# hwloc: required by the plugin's topology code.
module load hwloc 2>/dev/null || true
HWLOC_FLAG=""
if [ -n "${HWLOC_HOME:-}" ]; then
    HWLOC_FLAG="--with-hwloc=$HWLOC_HOME"
elif ! echo '#include <hwloc.h>' | gcc -E - >/dev/null 2>&1; then
    echo "WARNING: hwloc headers not on the default path; if configure fails,"
    echo "         module load hwloc (or set HWLOC_HOME) and rerun."
fi

# --- Fetch + pick tag -----------------------------------------------------
SRC="${TMPDIR:-/tmp}/aws-ofi-nccl-src.$$"
git clone --quiet https://github.com/aws/aws-ofi-nccl.git "$SRC"
cd "$SRC"
if [ -z "${AWS_OFI_TAG:-}" ]; then
    AWS_OFI_TAG=$(git describe --tags "$(git rev-list --tags --max-count=1)")
fi
git checkout --quiet "$AWS_OFI_TAG"
echo "aws-ofi-nccl tag: $AWS_OFI_TAG"

PREFIX="${PREFIX:-/glade/work/$USER/nccl-ofi/$AWS_OFI_TAG}"

# --- Build ------------------------------------------------------------------
./autogen.sh
# Flags verified against v1.20.0 configure.ac (m4/check_pkg_{libfabric,
# cuda,hwloc}.m4 + AC_ARG_ENABLE([tests])); there is deliberately NO
# --with-nccl (see header note).
./configure \
    --prefix="$PREFIX" \
    --with-libfabric="$LIBFABRIC_HOME" \
    --with-cuda="$CUDA_HOME" \
    $HWLOC_FLAG \
    --disable-tests
make -j 8
make install

echo
echo "============================================================"
ls -la "$PREFIX/lib/"
[ -e "$PREFIX/lib/libnccl-net.so" ] || {
    echo "ERROR: libnccl-net.so missing after install." >&2
    exit 1
}
echo "OK: aws-ofi-nccl installed."
echo
echo "Use it:"
echo "  qsub -v LEGOESM_NCCL_OFI_LIB=$PREFIX/lib \\"
echo "       scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs"
echo
echo "First-run verification: the job sets NCCL_DEBUG=INFO; the log MUST"
echo "print 'Using network AWS Libfabric' (not 'Socket')."
echo
echo "NOTE: rebuild this plugin whenever Derecho's system libfabric is"
echo "upgraded (the CXI provider ABI moves with it)."
echo "============================================================"
