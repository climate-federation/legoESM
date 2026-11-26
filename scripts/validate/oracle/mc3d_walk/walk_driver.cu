// Standalone driver for microhh rte-rrtmgp-cpp v0.3 Raytracer::trace_rays (fp64).
// Reads walk_in.bin, writes walk_out.bin (raw little-endian). See make_fixture.py.
//  in : int nx ny nz knx kny knz n_mie n_r ; uint64 ppp ; int seed ;
//       double dx dy dz zenith azimuth tod_dir tod_dif albedo ;
//       double tau_tot ssa_tot tau_cld ssa_cld asy_cld r_eff [nx*ny*nz each,
//       idx = x + y*nx + z*nx*ny] ; double mie_cdf[n_mie] mie_ang[n_r*n_mie]
//  out: tod_dn tod_up sfc_dir sfc_dif sfc_up [nx*ny] ; abs_dir abs_dif [nx*ny*nz]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "Array.h"
#include "Raytracer.h"
#include "gas_optics_rrtmgp_kernels_cuda_rt.h"

void harness_set_rng_offset(unsigned int off);

// Raytracer.cu zeroes its count arrays through these launchers; the gas-optics
// translation unit that defines them is not built here.
namespace Gas_optics_rrtmgp_kernels_cuda_rt
{
    void zero_array(const int ni, const int nj, Float* arr)
    { cuda_safe_call(cudaMemset(arr, 0, sizeof(Float)*ni*nj)); }
    void zero_array(const int ni, const int nj, const int nk, Float* arr)
    { cuda_safe_call(cudaMemset(arr, 0, sizeof(Float)*ni*nj*nk)); }
}

template<typename T> static T rd(FILE* f) { T v; if (fread(&v, sizeof(T), 1, f) != 1) exit(3); return v; }
static std::vector<Float> rdv(FILE* f, size_t n)
{ std::vector<Float> v(n); if (n && fread(v.data(), sizeof(Float), n, f) != n) exit(4); return v; }

int main(int argc, char** argv)
{
    if (argc != 3) { fprintf(stderr, "usage: walk_driver in.bin out.bin\n"); return 2; }
    FILE* f = fopen(argv[1], "rb"); if (!f) return 2;
    const int nx = rd<int>(f), ny = rd<int>(f), nz = rd<int>(f);
    const int knx = rd<int>(f), kny = rd<int>(f), knz = rd<int>(f);
    const int n_mie = rd<int>(f), n_r = rd<int>(f);
    const Int ppp = rd<unsigned long long>(f);
    const int seed = rd<int>(f);
    const Float dx = rd<double>(f), dy = rd<double>(f), dz = rd<double>(f);
    const Float zen = rd<double>(f), azi = rd<double>(f);
    const Float tdir = rd<double>(f), tdif = rd<double>(f), alb = rd<double>(f);
    const int ncol = nx*ny, nc = ncol*nz;
    auto tau_tot = rdv(f, nc), ssa_tot = rdv(f, nc), tau_cld = rdv(f, nc);
    auto ssa_cld = rdv(f, nc), asy_cld = rdv(f, nc), r_eff = rdv(f, nc);
    auto cdf = rdv(f, n_mie), ang = rdv(f, size_t(n_r)*n_mie);
    fclose(f);

    auto a2 = [&](const std::vector<Float>& v) {
        Array<Float,2> a({ncol, nz}); for (int i = 0; i < nc; ++i) a.v()[i] = v[i]; return Array_gpu<Float,2>(a); };
    std::vector<Float> zero(nc, Float(0.));
    Array_gpu<Float,2> tt = a2(tau_tot), st = a2(ssa_tot), tc = a2(tau_cld), sc = a2(ssa_cld),
                       ac = a2(asy_cld), re = a2(r_eff), ta = a2(zero), sa = a2(zero), aa = a2(zero);
    Array<Float,2> alb_c({1, ncol}); alb_c.fill(alb);
    Array_gpu<Float,2> alb_g(alb_c);
    Array<Float,2> cdf_c({n_mie > 0 ? n_mie : 1, 1});
    Array<Float,3> ang_c({n_mie > 0 ? n_mie : 1, n_r > 0 ? n_r : 1, 1});
    for (int i = 0; i < n_mie; ++i) cdf_c.v()[i] = cdf[i];
    for (size_t i = 0; i < ang.size(); ++i) ang_c.v()[i] = ang[i];
    Array_gpu<Float,2> cdf_g; Array_gpu<Float,3> ang_g;
    if (n_mie > 0) { cdf_g = Array_gpu<Float,2>(cdf_c); ang_g = Array_gpu<Float,3>(ang_c); }

    Array_gpu<Float,2> tod_dn({nx, ny}), tod_up({nx, ny}), sfc_dir({nx, ny}), sfc_dif({nx, ny}), sfc_up({nx, ny});
    Array_gpu<Float,3> abs_dir({nx, ny, nz}), abs_dif({nx, ny, nz});

    harness_set_rng_offset(unsigned(seed) * 1000003u);
    Raytracer rt;
    rt.trace_rays(seed, ppp, {nx, ny, nz}, {dx, dy, dz}, {knx, kny, knz},
                  cdf_g, ang_g, tt, st, tc, sc, ac, ta, sa, aa, re, alb_g,
                  zen, azi, tdir, tdif, tod_dn, tod_up, sfc_dir, sfc_dif, sfc_up, abs_dir, abs_dif);
    cuda_safe_call(cudaGetLastError());
    cuda_safe_call(cudaDeviceSynchronize());

    FILE* o = fopen(argv[2], "wb");
    auto w2 = [&](Array_gpu<Float,2>& g) { Array<Float,2> c(g); fwrite(c.ptr(), sizeof(Float), ncol, o); };
    auto w3 = [&](Array_gpu<Float,3>& g) { Array<Float,3> c(g); fwrite(c.ptr(), sizeof(Float), nc, o); };
    w2(tod_dn); w2(tod_up); w2(sfc_dir); w2(sfc_dif); w2(sfc_up); w3(abs_dir); w3(abs_dif);
    fclose(o);
    printf("WALK_OK seed=%d\n", seed);
    return 0;
}
