// Oracle driver for the legoESM SDM port: RUNS ERF 26.09
// (erf-model/ERF @ 6d09aeb120eb4eac388106004bfaaa7f795a2e26) super-droplet
// physics, compiled from the ERF headers in double precision on the host with
// minimal AMReX stubs (./stub).  Reads one task on stdin, prints one value per
// line (%.17e).  SI units (ERF's).
//
// Everything numeric is ERF code: TerminalVelocity, CollisionKernel (verbatim
// extraction, erf_kernels.H), dRsqdt / NewtonSolver / TI (header), the water
// MaterialPropertiesCore coefficients, erf_esatw, coalescence_rate and
// coal_update_attribs (verbatim extraction, erf_coal_update.H).  The glue that
// is NOT ERF code (AMReX particle containers, grid interpolation, RNG) is
// replaced by the explicit inputs read here and is marked HARNESS.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#define ERF_USE_PARTICLES 1
#include "AMReX_REAL.H"
#include "ERF_Constants.H"
#include "ERF_MicrophysicsUtils.H"
#include "ERF_MaterialProperties.H"
#include "ERF_TerminalVelocity.H"
#include "ERF_SuperDropletPCMassChange.H"
#include "erf_kernels.H"
#include "erf_coal_update.H"

using R = double;
static void out(double x) { std::printf("%.17e\n", x); }

// HARNESS: MaterialProperties::setProperties_H2O (ERF_MaterialProperties.cpp
// l.169-180) and setProperties_NaCl (l.222-227) transcribed field by field (the
// .cpp needs AMReX MultiFab machinery); the coefficient FUNCTIONS used below are
// the ERF header code.
static MaterialPropertiesCore h2o() {
  MaterialPropertiesCore m;
  m.m_density = rhor;
  m.m_ionization = 0;
  m.m_mol_weight = R(1.802e-02);
  m.m_lat_vap = L_v;
  m.m_lat_fus = lfus;
  m.m_Rv = R_v;
  m.m_Tb = R(373.15);
  m.m_Nav_by_molweight = MaterialPropertiesCore::s_N_av / m.m_mol_weight;
  m.m_is_water = true;
  return m;
}
static const R NACL_ION = 2, NACL_MW = R(5.844e-02);

int main() {
  std::string task; std::cin >> task;
  if (task == "KERNEL") {
    int n; std::cin >> n;
    CollisionKernel<R, 3> ck{};
    for (int k = 0; k < n; ++k) {
      R r1, r2, v1, v2, m1, m2, p, T; std::cin >> r1 >> r2 >> v1 >> v2 >> m1 >> m2 >> p >> T;
      R va[3] = {0, 0, -v1}, vb[3] = {0, 0, -v2}, x0[3] = {0, 0, 0};
      out(ck.golovin(r1, r2));
      out(ck.Longs(r1, r2, va, vb, x0, x0, SDKernelRelativeVelocityType::terminal_velocity));
      out(ck.Halls(r1, r2, va, vb, x0, x0, SDKernelRelativeVelocityType::terminal_velocity));
      out(ck.sedimentation(r1, r2, va, vb, x0, x0, SDKernelRelativeVelocityType::terminal_velocity));
      out(ck.Brownian_SeinfeldPandis(r1, r2, m1, m2, p, T));
    }
  } else if (task == "VTERM") {
    int n; std::cin >> n;
    TerminalVelocity<R> tv{rhor, R(917.0)};
    for (int k = 0; k < n; ++k) {
      R r, rho, p, T; std::cin >> r >> rho >> p >> T;
      out(tv.RogersYau(r)); out(tv.AtlasUlbrich(r)); out(tv.CloudRainShima(r, rho, p, T));
    }
  } else if (task == "ESATW") {
    // MaterialProperties.cpp l.29: psat = erf_esatw(T) * 100  [Pa]
    int n; std::cin >> n;
    for (int k = 0; k < n; ++k) { R T; std::cin >> T; out(erf_esatw(T) * R(100)); }
  } else if (task == "DRSQ") {
    // rhs_func / rhs_jac with the coefficients MassChange_LV builds for water
    // (MassChange.cpp l.72-77, 150-152; MaterialProperties.H coeff*()).
    const MaterialPropertiesCore w = h2o();
    SDMassChangeUtils_LV::dRsqdt<R> ode{w.m_lat_vap, therco, w.m_Rv, w.m_density};
    int n; std::cin >> n;
    for (int k = 0; k < n; ++k) {
      R u, S, T, p, msol; std::cin >> u >> S >> T >> p >> msol;
      R es = erf_esatw(T) * R(100);
      R Ns = msol * NACL_ION / NACL_MW;
      R a = w.coeffCurv(T), b = w.coeffVPSolute(), D = w.coeffMolecularDiffusion(T, p);
      out(ode.rhs_func(u, S, T, es, D, a, b, Ns, R(1)));
      out(ode.rhs_jac(u, T, es, D, a, b, Ns, R(1)));
    }
  } else if (task == "MASSCHANGE") {
    // SuperDropletPC::MassChange_LV per-particle body (MassChange.cpp
    // l.122-232) for a water droplet carrying one NaCl aerosol; ventilation off
    // (Initializations.cpp l.79 default).  HARNESS: the grid interpolation is
    // replaced by the explicit (S, T, p) of each particle, and the particle's
    // species masses are set so that SD_effective_radius == r_init.
    int method; R cfl, nrtol, natol, nstol; int nmax;
    std::cin >> method >> cfl >> nrtol >> natol >> nstol >> nmax;
    using namespace SDMassChangeUtils_LV;
    const MaterialPropertiesCore w = h2o();
    const R mat_density = w.m_density;
    dRsqdt<R> ode{w.m_lat_vap, therco, w.m_Rv, mat_density};
    NewtonSolver<dRsqdt<R>, R> newton{ode, nrtol, natol, nstol, nmax};
    int n; std::cin >> n;
    for (int k = 0; k < n; ++k) {
      R r_init, msol, S, T, p, dt; std::cin >> r_init >> msol >> S >> T >> p >> dt;
      const R e_sat = erf_esatw(T) * R(100);
      const R solute_moles = msol * NACL_ION / NACL_MW;
      const R coeff_curv = w.coeffCurv(T), coeff_sol = w.coeffVPSolute();
      const R coeff_moldiff = w.coeffMolecularDiffusion(T, p);
      TI<dRsqdt<R>, NewtonSolver<dRsqdt<R>, R>, R> ti{ode, newton, dt, 100,
          S, T, e_sat, coeff_moldiff, coeff_curv, coeff_sol, solute_moles,
          cfl, R(1e-40), R(1e-3), R(1e-6), false, false,
          false, R(6.954e7), R(1.963), R(1.069e3), R(0.702), R(20.0)};
      // HARNESS: species masses with SD_effective_radius(r_init) (Definitions.H l.233-258)
      R m_w = four_thirds_pi * rhor * r_init * r_init * r_init - msol;
      R r_sq = r_init * r_init; bool success = false;
      if (method == 0) ti.be(r_sq, success);
      else if (method == 1) ti.rk4(r_sq, success);
      else if (method == 2) ti.cn(r_sq, success);
      else if (method == 3) ti.dirk212(r_sq, success);
      else if (method == 4) ti.rk3bs(r_sq, success);
      R r_out = r_init;
      if (success) {
        R r_new = std::sqrt(r_sq);
        R d_mass = four_thirds_pi * mat_density * (r_new * r_new * r_new - r_init * r_init * r_init);
        m_w += d_mass;
        m_w = std::max(m_w, R(0));
        R m_t = m_w + msol;                                  // soluble aerosol: m_s
        r_out = std::cbrt(m_t / (four_thirds_pi * rhor));    // SD_effective_radius
      }
      out(r_out); out(success ? 1.0 : 0.0);
    }
  } else if (task == "COAL") {
    // Coalescence.cpp l.600-905 water-water path, with the shuffle replaced by
    // explicit pairs and the RNG draw by an explicit uniform (HARNESS).
    int kern, brown; R V, dt, p, T; int n; std::cin >> kern >> brown >> V >> dt >> p >> T >> n;
    std::vector<R> mult(n), rad(n), mass(n), vt(n), tfz(n, 0.0), sol(n);
    for (int k = 0; k < n; ++k) std::cin >> mult[k] >> rad[k] >> mass[k] >> vt[k] >> sol[k];
    int np; std::cin >> np;
    std::vector<int> partner(n, -1), prey(n, -1), npbin(n, 0);
    std::vector<R> rate(n, 0), rmndr(n, 0), uu(n, 0);
    CollisionKernel<R, 3> ck{};
    std::vector<int> PI_(np), PJ_(np); std::vector<R> U(np);
    for (int q = 0; q < np; ++q) std::cin >> PI_[q] >> PJ_[q] >> U[q];
    for (int q = 0; q < np; ++q) {                 // pairing, l.612-631
      int pi = PI_[q], pj = PJ_[q];
      if (pi == pj) continue;
      if (mult[pi] == 0 || mult[pj] == 0) continue;
      npbin[pi] = npbin[pj] = n;
      partner[pi] = pj; partner[pj] = pi;
      int i = (mult[pi] >= mult[pj]) ? pi : pj; int j = (i == pi) ? pj : pi;
      prey[i] = 1; prey[j] = 0; uu[i] = U[q];
    }
    for (int i = 0; i < n; ++i) {                  // rates, l.639-905
      if (partner[i] < 0) continue;
      if (!prey[i]) continue;
      int pi = i, pj = partner[i];
      R k_val = 0;
      if (kern == 0) k_val = ck.golovin(rad[pi], rad[pj]);
      else {
        R va[3] = {0, 0, -vt[pi]}, vb[3] = {0, 0, -vt[pj]}, x0[3] = {0, 0, 0};
        if (kern == 1) k_val = ck.Halls(rad[pi], rad[pj], va, vb, x0, x0, SDKernelRelativeVelocityType::terminal_velocity);
        else if (kern == 2) k_val = ck.sedimentation(rad[pi], rad[pj], va, vb, x0, x0, SDKernelRelativeVelocityType::terminal_velocity);
        else if (kern == 3) k_val = ck.Longs(rad[pi], rad[pj], va, vb, x0, x0, SDKernelRelativeVelocityType::terminal_velocity);
      }
      if (brown) k_val += ck.Brownian_SeinfeldPandis(rad[pi], rad[pj], mass[pi], mass[pj], p, T);
      R prob_ij = k_val / V;
      R prob_sd_ij = std::max(mult[pi], mult[pj]) * prob_ij;
      R ns = npbin[i];
      R scaling = myhalf * ns * (ns - 1) / std::floor(myhalf * ns);
      R scaled = prob_sd_ij * scaling;
      RandomEngine eng{uu[i]};
      R gamma = coalescence_rate(eng, scaled * dt);
      if (gamma > 0) {
        rate[pi] = std::min(gamma, std::floor(mult[pi] / mult[pj])); rate[pj] = rate[pi];
        rmndr[pi] = mult[pi] - rate[pi] * mult[pj]; rmndr[pj] = rmndr[pi];
      } else { partner[pi] = -1; partner[pj] = -1; }
    }
    // ParallelFor over particles (l.919-940): every particle reads the
    // pre-update partner state; emulate with per-particle copies.
    std::vector<R> mult1 = mult, rad1 = rad, mass1 = mass, sol1 = sol;
    SDSpeciesMassArr spm{};
    for (int i = 0; i < n; ++i) {
      int j = partner[i]; if (j < 0) continue;
      std::vector<R> m_ = mult, r_ = rad, ms_ = mass, s_ = sol, t_ = tfz;
      SDAerosolMassArr ae{}; ae[0] = s_.data();
      coal_update_attribs(i, j, prey.data(), rate.data(), rmndr.data(), ms_.data(), r_.data(),
                          t_.data(), m_.data(), 0, spm, 1, ae);
      // write back exactly what coal_update_attribs(i, j) writes: particle i
      // when the remainder is positive; i AND j from the prey when it is zero.
      if (rmndr[i] > 0 || prey[i] == 1) { mult1[i] = m_[i]; rad1[i] = r_[i]; mass1[i] = ms_[i]; sol1[i] = s_[i]; }
      if (prey[i] == 1 && rmndr[i] == 0) { mult1[j] = m_[j]; rad1[j] = r_[j]; mass1[j] = ms_[j]; sol1[j] = s_[j]; }
    }
    for (int k = 0; k < n; ++k) { out(mult1[k]); out(rad1[k]); out(mass1[k]); out(sol1[k]); }
  } else {
    std::fprintf(stderr, "unknown task %s\n", task.c_str()); return 2;
  }
  return 0;
}
