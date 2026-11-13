#!/bin/bash
# Build the microhh rte-rrtmgp-cpp v0.3 shortwave ray-tracer oracle (fp64, CUDA).
# usage: build.sh <rte-rrtmgp-cpp checkout @ v0.3> <build dir>
# Sources are used unmodified except ONE harness line: the per-thread curand
# seed `rng(n)` gets a per-run offset so seed replicates are independent
# (v0.3 seeds by thread id only).  The patched copy is diffed in the log.
set -euo pipefail
R=$(cd "$1" && pwd); mkdir -p "$2"; B=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
echo "rte-rrtmgp-cpp $(git -C "$R" describe --tags) $(git -C "$R" rev-parse HEAD)"
K=$R/src_kernels_cuda_rt/raytracer_kernels.cu
sed 's/Random_number_generator<Float> rng(n);/Random_number_generator<Float> rng(n + harness_rng_offset);/' "$K" > "$B/raytracer_kernels_h.cu"
cat >> "$B/raytracer_kernels_h.cu" <<'EOT'
void harness_set_rng_offset(unsigned int off)
{ cudaMemcpyToSymbol(harness_rng_offset, &off, sizeof(off)); }
EOT
sed -i '0,/^namespace/s//__constant__ unsigned int harness_rng_offset;\nnamespace/' "$B/raytracer_kernels_h.cu"
# the patch must apply exactly once (replicate seed ranges: offset = seed*1000003
# > 256*512 threads, and the Sobol offset already advances by a full launch)
[ "$(grep -c 'rng(n + harness_rng_offset)' "$B/raytracer_kernels_h.cu")" = 1 ] || { echo "seed patch did not apply once"; exit 1; }
diff "$K" "$B/raytracer_kernels_h.cu" || true
nvcc --version | tail -1
INC="-I$R/include -I$R/include_rt -I$R/include_rt_kernels"
FL="-std=c++17 -expt-relaxed-constexpr -Xptxas -O3 -DNDEBUG -DUSECUDA -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86"
nvcc $FL -Xptxas -v $INC -c "$B/raytracer_kernels_h.cu" -o "$B/k.o" 2>&1 | grep -E "ray_tracer_kernel|registers" | head -8
nvcc $FL $INC -c "$R/src_cuda_rt/Raytracer.cu" -o "$B/r.o"
nvcc $FL $INC -c "$HERE/walk_driver.cu" -o "$B/d.o"
nvcc $FL "$B/k.o" "$B/r.o" "$B/d.o" -lcurand -o "$B/walk_driver"
echo "BUILD_OK $B/walk_driver"
