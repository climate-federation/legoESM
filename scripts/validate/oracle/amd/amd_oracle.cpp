// ORACLE: AMR-Wind v4.2.0 (Exawind/amr-wind, tag commit 0ee09c22b9afeca06231e4910276c0ba05fb52dd)
// src/turbulence/LES/AMD.H : function amd_muvel, body copied VERBATIM below.
// AUDIT stubs (harness only): amrex::Real=double, GpuArray, Array4 (component-major
// point arrays), amrex::max, direction_selector, interp::linear. Thermal input is zero
// and beta=0, so the buoyancy term vanishes (pure shear AMD, as in legoESM amd_nu_t).
// Layout convention verified from amr-wind src/fvm/gradient.H:
// gradphi(i,j,k, icomp*AMREX_SPACEDIM + idim) = d phi_icomp / d x_idim.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <vector>
#include <algorithm>
#define AMREX_SPACEDIM 3
#define AMREX_GPU_DEVICE
#define AMREX_FORCE_INLINE inline
namespace amrex {
using Real = double;
template <typename T, int N> struct GpuArray { T a[N]; const T& operator[](int i) const { return a[i]; } };
template <typename T> struct Array4 {
  const double* p; long np;  // component-major: p[n*np + i]
  double operator()(int i, int, int, int n) const { return p[n * np + i]; }
};
template <typename T> T max(T a, T b) { return std::max(a, b); }
}
namespace kynema_sgf {
inline int direction_selector(int i, int j, int k, int d) { return d == 0 ? i : (d == 1 ? j : k); }
namespace interp { inline double linear(const double*, const double*, const double*, double) { return 0.0; } }
}
inline double operator""_rt(long double x) { return (double)x; }

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real amd_muvel(
    int i,
    int j,
    int k,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx, // Grid spacing
    const amrex::Real beta, // Thermal expansion coefficient
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& gravity,
    const amrex::Real C, // Poincare const
    const amrex::Array4<amrex::Real const>& gradVel,
    const amrex::Array4<amrex::Real const>& gradT,
    const amrex::Real* gradTbar_coord_begin,
    const amrex::Real* gradTbar_coord_end,
    const amrex::Real* gradTbar,
    const int normal_dir,
    const amrex::Real nlo)
{
    amrex::Real num_shear = 0.0_rt;
    amrex::Real num_buoy = 0.0_rt;
    amrex::Real denom = 0.0_rt;
    const int n_ind = kynema_sgf::direction_selector(i, j, k, normal_dir);
    const amrex::Real h = nlo + ((n_ind + 0.5_rt) * dx[normal_dir]);
    const amrex::Real gradTbar_h = kynema_sgf::interp::linear(
        gradTbar_coord_begin, gradTbar_coord_end, gradTbar, h);
    for (int ii = 0; ii < AMREX_SPACEDIM; ++ii) {
        // This should operate only on the wall normal velocity
        num_buoy +=
            gradVel(i, j, k, (normal_dir * AMREX_SPACEDIM) + ii) *
            (gradT(i, j, k, ii) - ((ii == normal_dir) ? gradTbar_h : 0.0_rt)) *
            dx[ii] * dx[ii];

        for (int jj = 0; jj < AMREX_SPACEDIM; ++jj) {
            const amrex::Real diuj =
                gradVel(i, j, k, (ii * AMREX_SPACEDIM) + jj);
            const amrex::Real djui =
                gradVel(i, j, k, (jj * AMREX_SPACEDIM) + ii);
            denom += diuj * diuj;
            const amrex::Real sij = 0.5_rt * (diuj + djui);
            for (int kk = 0; kk < AMREX_SPACEDIM; ++kk) {
                const amrex::Real dkui =
                    gradVel(i, j, k, (ii * AMREX_SPACEDIM) + kk);
                const amrex::Real dkuj =
                    gradVel(i, j, k, (jj * AMREX_SPACEDIM) + kk);
                num_shear += dkui * dkuj * dx[kk] * dx[kk] * sij;
            }
        }
    }
    denom = amrex::max<amrex::Real>(
        std::numeric_limits<amrex::Real>::epsilon() * 1.0e1_rt, denom);
    num_shear *= -C;
    num_buoy *= C * beta * std::abs(gravity[normal_dir]);
    return amrex::max<amrex::Real>(
        std::numeric_limits<amrex::Real>::epsilon() * 1.0e1_rt,
        (num_shear + num_buoy) / denom);
}

// AUDIT driver: same binary input as the Vreman driver (int32 npts; a11..a33 (a_cd=du_c/dx_d),
// h1,h2,h3 as npts float64 each; float64 C). Per-point dx from h1..h3.
int main(int argc, char** argv) {
  FILE* f = std::fopen(argv[1], "rb"); int32_t n; std::fread(&n, 4, 1, f);
  std::vector<double> a(12L * n); std::fread(a.data(), 8, 12L * n, f);
  double C; std::fread(&C, 8, 1, f); std::fclose(f);
  std::vector<double> zeros(3L * n, 0.0), out(n);
  amrex::Array4<amrex::Real const> gv{a.data(), n}, gt{zeros.data(), n};
  amrex::GpuArray<double, 3> g{{0.0, 0.0, -9.81}};
  double tb = 0.0;
  for (long i = 0; i < n; ++i) {
    amrex::GpuArray<double, 3> dx{{a[9L * n + i], a[10L * n + i], a[11L * n + i]}};
    out[i] = amd_muvel((int)i, 0, 0, dx, 0.0, g, C, gv, gt, &tb, &tb, &tb, 2, 0.0);
  }
  f = std::fopen(argv[2], "wb"); std::fwrite(out.data(), 8, n, f); std::fclose(f);
  std::printf("AMD_ORACLE_DONE %d %g eps10=%g\n", n, C, std::numeric_limits<double>::epsilon() * 10.0);
  return 0;
}
