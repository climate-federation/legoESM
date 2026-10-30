// Host driver for the microhh rte-rrtmgp-cpp ray-tracer DEVICE FUNCTIONS: the verbatim
// headers (include_rt/raytracer_functions.h, include/types.h) and the create_knull_grid
// kernel (src_cuda_rt/Raytracer.cu, extracted verbatim by build.sh), compiled for CPU in
// double precision with __device__/__host__/__global__ defined empty.
// usage: oracle_driver <dir>   (little-endian; int32 headers, float64 arrays)
//  henyey_in.bin : n; g[n], u[n]                      -> henyey_out.bin : mu[n]
//  rayl_in.bin   : n; u[n]                            -> rayl_out.bin   : mu[n]
//  tau_in.bin    : n; u[n], k[n]                      -> tau_out.bin    : dn[n]
//                  dn = max(Float_epsilon, sample_tau(u) / k): the step of
//                  src_kernels_cuda_rt/raytracer_kernels.cu:212 (that expression, copied)
//  mie_in.bin    : n_mie, n_r, n; cdf[n_mie], ang[n_r*n_mie], u[n], reff[n] -> mie_out.bin : ang[n]
//  knull_in.bin  : nx,ny,nz,kx,ky,kz; kext[nx*ny*nz] (x fastest) -> knull_out.bin : kn[kx*ky*kz]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
using std::min; using std::max; using std::abs;
#include "raytracer_functions.h"
using namespace Raytracer_functions;
struct D3 { int x, y, z; };
static D3 blockIdx, threadIdx, blockDim;
#include "knull_kernel.inc"

static std::vector<double> rd(FILE* f, size_t n) { std::vector<double> v(n); if (n && fread(v.data(), 8, n, f) != n) exit(3); return v; }
static FILE* op(const std::string& p, const char* m) { FILE* f = fopen(p.c_str(), m); if (!f) { fprintf(stderr, "open %s\n", p.c_str()); exit(1); } return f; }
static void wr(const std::string& p, const std::vector<double>& v) { FILE* f = op(p, "wb"); fwrite(v.data(), 8, v.size(), f); fclose(f); }
static int ri(FILE* f) { int n; if (fread(&n, 4, 1, f) != 1) exit(3); return n; }

int main(int argc, char** argv) {
  std::string d = argv[1];
  { FILE* f = op(d + "/henyey_in.bin", "rb"); int n = ri(f); auto g = rd(f, n); auto u = rd(f, n); fclose(f);
    std::vector<double> o(n); for (int i = 0; i < n; ++i) o[i] = henyey(Float(g[i]), Float(u[i])); wr(d + "/henyey_out.bin", o); }
  { FILE* f = op(d + "/rayl_in.bin", "rb"); int n = ri(f); auto u = rd(f, n); fclose(f);
    std::vector<double> o(n); for (int i = 0; i < n; ++i) o[i] = rayleigh(Float(u[i])); wr(d + "/rayl_out.bin", o); }
  { FILE* f = op(d + "/tau_in.bin", "rb"); int n = ri(f); auto u = rd(f, n); auto k = rd(f, n); fclose(f);
    std::vector<double> o(n);
    for (int i = 0; i < n; ++i) { const Float tau = sample_tau(Float(u[i])); const Float k_ext_null = Float(k[i]);
      o[i] = max(Float_epsilon, tau / k_ext_null); }
    wr(d + "/tau_out.bin", o); }
  { FILE* f = op(d + "/mie_in.bin", "rb"); int nm = ri(f), nr = ri(f), n = ri(f);
    auto cdf = rd(f, nm); auto ang = rd(f, (size_t)nr * nm); auto u = rd(f, n); auto re = rd(f, n); fclose(f);
    std::vector<double> o(n);
    for (int i = 0; i < n; ++i) o[i] = mie_sample_angle(cdf.data(), ang.data(), Float(u[i]), Float(re[i]), nm);
    wr(d + "/mie_out.bin", o); }
  { FILE* f = op(d + "/knull_in.bin", "rb"); int h[6]; for (int i = 0; i < 6; ++i) h[i] = ri(f);
    auto ke = rd(f, (size_t)h[0] * h[1] * h[2]); fclose(f);
    std::vector<Float> kn((size_t)h[3] * h[4] * h[5]);
    Vector<int> gc{h[0], h[1], h[2]}, kg{h[3], h[4], h[5]};
    blockDim = {1, 1, 1}; threadIdx = {0, 0, 0};
    for (int z = 0; z < h[5]; ++z) for (int y = 0; y < h[4]; ++y) for (int x = 0; x < h[3]; ++x) {
      blockIdx = {x, y, z}; create_knull_grid(gc, Float(1e-3), kg, ke.data(), kn.data()); }
    wr(d + "/knull_out.bin", std::vector<double>(kn.begin(), kn.end())); }
  printf("ORACLE_DONE\n");
  return 0;
}
