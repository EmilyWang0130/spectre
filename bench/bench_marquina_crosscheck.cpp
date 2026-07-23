// Distributed under the MIT License.
// See LICENSE.txt for details.
//
// Numerical cross-validation of SpECTRE's Marquina flux against the WHISKY
// reference on IDENTICAL input states (unlike the throughput benchmarks, which
// only share input distributions). For each random flat-metric, normal-along-x
// hydro state pair (u_L, u_R) it compares, between the two implementations:
//   1. the characteristic speeds lambda_+, lambda_-, lambda_0 of u_L;
//   2. the acoustic right eigenvectors R+, R- of u_L (both D-normalized);
//   3. the assembled Marquina numerical flux (D, S_x, S_y, S_z, tau).
//
// SpECTRE's WeakInertial dg_boundary_terms output IS the Marquina numerical
// flux F*. WHISKY assembles F* = 0.5*(F(u_L).x + F(u_R).x - dissipation); we
// form the same combination using SpECTRE's own physical fluxes F(u).x for the
// two sides and WHISKY's dissipation, so the only thing under test is the
// scheme, not the flux/EOS definitions (identical conserved states and
// analytic IdealFluid Gamma=2 EOS quantities are fed to both).

#include <charm++.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"

#include "bench_marquina_common.hpp"

extern "C" void CkRegisterMainModule(void) {}

// WHISKY diagnostic kernel (bench/whisky_crosscheck.F90).
extern "C" void whisky_marquina_diag(
    double rhoL, double vxL, double vyL, double vzL, double epsL, double pressL,
    double cs2L, double dpdepsL, double densL, double sxL, double syL,
    double szL, double tauL, double rhoR, double vxR, double vyR, double vzR,
    double epsR, double pressR, double cs2R, double dpdepsR, double densR,
    double sxR, double syR, double szR, double tauR, double* lam_l,
    double* reivecp_l, double* reivecm_l, double* leivecp_l, double* leivecm_l,
    double* diss);

namespace {
namespace bm = bench_marquina;
using bm::Marquina;
namespace grmhd_tags = grmhd::ValenciaDivClean::Tags;

struct Stats {
  double max_abs = 0.0;
  double max_rel = 0.0;
  std::vector<double> abs_samples{};
  std::vector<double> rel_samples{};
  void add(double got, double ref) {
    const double ad = std::abs(got - ref);
    const double rd = ad / (std::abs(ref) + 1.0e-300);
    max_abs = std::max(max_abs, ad);
    max_rel = std::max(max_rel, rd);
    abs_samples.push_back(ad);
    rel_samples.push_back(rd);
  }
  double pct(std::vector<double>* v, double p) {
    if (v->empty()) {
      return 0.0;
    }
    const size_t k =
        static_cast<size_t>((p / 100.0) * static_cast<double>(v->size() - 1));
    std::nth_element(v->begin(), v->begin() + k, v->end());
    return (*v)[k];
  }
  void report(const char* name) {
    std::printf(
        "  %-10s abs: p50=%.2e p99=%.2e max=%.2e | rel: p50=%.2e "
        "p99=%.2e max=%.2e\n",
        name, pct(&abs_samples, 50), pct(&abs_samples, 99), max_abs,
        pct(&rel_samples, 50), pct(&rel_samples, 99), max_rel);
  }
};
}  // namespace

int main() {
  constexpr size_t kN = 20000;
  constexpr double kGamma = 2.0;
  const EquationsOfState::IdealFluid<true> base_eos{kGamma, 0.0};
  const std::unique_ptr<EquationsOfState::EquationOfState<true, 3>> eos =
      base_eos.promote_to_3d_eos();

  std::mt19937 gen(20240716);
  bm::PrimitiveStore prim_l;
  bm::PrimitiveStore prim_r;
  bm::generate(&prim_l, &gen, kN);
  bm::generate(&prim_r, &gen, kN);

  // Crafted positive-control states in slots 0..3 (printed as examples):
  //   0: identical L=R subsonic  -> trivial interface, both schemes give F(u)
  //   1: identical L=R supersonic -> trivial interface, both give F(u)
  //   2: supersonic L!=R (all lambda>0 on both sides) -> SpECTRE gives the pure
  //      upwind F(u_L); WHISKY adds max-|lambda| viscosity so it differs
  //   3: subsonic L!=R -> both differ from central and from each other
  const auto set_state = [](bm::PrimitiveStore* s, size_t i, double rho,
                            double vx, double eps) {
    s->rho[i] = rho;
    s->vx[i] = vx;
    s->vy[i] = 0.0;
    s->vz[i] = 0.0;
    s->eps[i] = eps;
  };
  set_state(&prim_l, 0, 1.0, 0.1, 0.5);
  set_state(&prim_r, 0, 1.0, 0.1, 0.5);
  set_state(&prim_l, 1, 1.0, 0.3, 0.01);
  set_state(&prim_r, 1, 1.0, 0.3, 0.01);
  set_state(&prim_l, 2, 2.0, 0.3, 0.01);
  set_state(&prim_r, 2, 1.0, 0.29, 0.01);
  set_state(&prim_l, 3, 2.0, 0.10, 0.5);
  set_state(&prim_r, 3, 1.0, -0.05, 0.4);

  bm::SideState interior;
  bm::SideState exterior;
  interior.allocate(kN);
  exterior.allocate(kN);
  bm::BoundaryCorrection bc;
  bc.allocate(kN);

  // Interior = u_L (normal +x), exterior = u_R (normal -x).
  bm::fill_side(&interior, prim_l, 0, kN, +1.0, *eos);
  bm::fill_side(&exterior, prim_r, 0, kN, -1.0, *eos);
  bm::run_package_data<Marquina>(&interior, *eos);
  bm::run_package_data<Marquina>(&exterior, *eos);
  bm::run_boundary_terms<Marquina>(&bc, interior, exterior);

  // SpECTRE packaged interior characteristic data (full Marquina).
  const auto& speeds =
      get<typename Marquina::CharacteristicSpeeds>(interior.package_data);
  const auto& right_ev =
      get<typename Marquina::RightCharacteristicFields>(interior.package_data);
  using HydroSpeed = grmhd::ValenciaDivClean::HydroSpeed;
  using HydroVectorR = grmhd::ValenciaDivClean::HydroVectorR;

  Stats lam_p, lam_m, lam_0;
  Stats f_d, f_sx, f_sy, f_sz, f_tau;
  Stats rvecp, rvecm;

  const int n_examples = 4;
  const char* example_labels[4] = {
      "identical L=R subsonic (trivial interface)",
      "identical L=R supersonic (trivial interface)",
      "supersonic L!=R, all lambda>0 (SpECTRE -> pure upwind F(u_L))",
      "subsonic L!=R (generic)"};
  std::printf(
      "=== SpECTRE Marquina vs WHISKY cross-check (%zu identical states) ===\n",
      kN);

  for (size_t i = 0; i < kN; ++i) {
    // Analytic IdealFluid Gamma=2 EOS quantities (match SpECTRE's IdealFluid).
    auto eos_quant = [](double rho, double eps, double* press, double* cs2,
                        double* dpde) {
      const double p = (kGamma - 1.0) * rho * eps;
      const double dp = (kGamma - 1.0) * rho;
      const double h = 1.0 + eps + p / rho;
      *press = p;
      *dpde = dp;
      *cs2 = ((kGamma - 1.0) * eps + p / (rho * rho) * dp) / h;
    };
    double pL, cL, dL, pR, cR, dR;
    eos_quant(prim_l.rho[i], prim_l.eps[i], &pL, &cL, &dL);
    eos_quant(prim_r.rho[i], prim_r.eps[i], &pR, &cR, &dR);

    // Identical conserved states: use SpECTRE's own tilde_* (flat metric).
    const double densL = get(interior.tilde_d)[i];
    const double sxL = interior.tilde_s.get(0)[i];
    const double syL = interior.tilde_s.get(1)[i];
    const double szL = interior.tilde_s.get(2)[i];
    const double tauL = get(interior.tilde_tau)[i];
    const double densR = get(exterior.tilde_d)[i];
    const double sxR = exterior.tilde_s.get(0)[i];
    const double syR = exterior.tilde_s.get(1)[i];
    const double szR = exterior.tilde_s.get(2)[i];
    const double tauR = get(exterior.tilde_tau)[i];

    std::array<double, 3> lam{};
    std::array<double, 5> rp{}, rm{}, lp{}, lm{}, diss{};
    whisky_marquina_diag(prim_l.rho[i], prim_l.vx[i], prim_l.vy[i],
                         prim_l.vz[i], prim_l.eps[i], pL, cL, dL, densL, sxL,
                         syL, szL, tauL, prim_r.rho[i], prim_r.vx[i],
                         prim_r.vy[i], prim_r.vz[i], prim_r.eps[i], pR, cR, dR,
                         densR, sxR, syR, szR, tauR, lam.data(), rp.data(),
                         rm.data(), lp.data(), lm.data(), diss.data());

    // --- eigenvalue comparison (u_L) ---
    lam_p.add(speeds.get(HydroSpeed::LambdaPlus)[i], lam[0]);
    lam_m.add(speeds.get(HydroSpeed::LambdaMinus)[i], lam[1]);
    lam_0.add(speeds.get(HydroSpeed::NormalDotVelocity)[i], lam[2]);

    // --- assembled Marquina flux: 0.5*(F(u_L).x + F(u_R).x - diss) ---
    // SpECTRE physical x-fluxes (raw, +x column) for both sides.
    const double fpl_d = interior.flux_tilde_d.get(0)[i];
    const double fmi_d = exterior.flux_tilde_d.get(0)[i];
    const double fpl_sx = interior.flux_tilde_s.get(0, 0)[i];
    const double fmi_sx = exterior.flux_tilde_s.get(0, 0)[i];
    const double fpl_sy = interior.flux_tilde_s.get(1, 0)[i];
    const double fmi_sy = exterior.flux_tilde_s.get(1, 0)[i];
    const double fpl_sz = interior.flux_tilde_s.get(2, 0)[i];
    const double fmi_sz = exterior.flux_tilde_s.get(2, 0)[i];
    const double fpl_tau = interior.flux_tilde_tau.get(0)[i];
    const double fmi_tau = exterior.flux_tilde_tau.get(0)[i];

    const double fw_d = 0.5 * (fpl_d + fmi_d - diss[0]);
    const double fw_sx = 0.5 * (fpl_sx + fmi_sx - diss[1]);
    const double fw_sy = 0.5 * (fpl_sy + fmi_sy - diss[2]);
    const double fw_sz = 0.5 * (fpl_sz + fmi_sz - diss[3]);
    const double fw_tau = 0.5 * (fpl_tau + fmi_tau - diss[4]);

    f_d.add(get(bc.tilde_d)[i], fw_d);
    f_sx.add(bc.tilde_s.get(0)[i], fw_sx);
    f_sy.add(bc.tilde_s.get(1)[i], fw_sy);
    f_sz.add(bc.tilde_s.get(2)[i], fw_sz);
    f_tau.add(get(bc.tilde_tau)[i], fw_tau);

    // --- acoustic R+/R- (D-normalized; SpECTRE (0,1,2,3,4)=D,Sx,Sy,Sz,tau
    //     maps to WHISKY (0,1,2,3,4) after dropping the DYe column). ---
    for (size_t c = 0; c < 5; ++c) {
      rvecp.add(right_ev.get(HydroVectorR::Rplus, c)[i], rp[c]);
      rvecm.add(right_ev.get(HydroVectorR::Rminus, c)[i], rm[c]);
    }

    if (static_cast<int>(i) < n_examples) {
      std::printf(
          "\n-- example %zu [%s]\n   rhoL=%.3f vxL=%.3f epsL=%.3f | "
          "rhoR=%.3f vxR=%.3f epsR=%.3f\n",
          i, example_labels[i], prim_l.rho[i], prim_l.vx[i], prim_l.eps[i],
          prim_r.rho[i], prim_r.vx[i], prim_r.eps[i]);
      std::printf("   lambda+  SpECTRE=% .8e  WHISKY=% .8e\n",
                  speeds.get(HydroSpeed::LambdaPlus)[i], lam[0]);
      std::printf("   lambda-  SpECTRE=% .8e  WHISKY=% .8e\n",
                  speeds.get(HydroSpeed::LambdaMinus)[i], lam[1]);
      std::printf("   lambda0  SpECTRE=% .8e  WHISKY=% .8e\n",
                  speeds.get(HydroSpeed::NormalDotVelocity)[i], lam[2]);
      std::printf("   R+ (D,Sx,Sy,Sz,tau):\n");
      std::printf("     SpECTRE % .6e % .6e % .6e % .6e % .6e\n",
                  right_ev.get(HydroVectorR::Rplus, 0)[i],
                  right_ev.get(HydroVectorR::Rplus, 1)[i],
                  right_ev.get(HydroVectorR::Rplus, 2)[i],
                  right_ev.get(HydroVectorR::Rplus, 3)[i],
                  right_ev.get(HydroVectorR::Rplus, 4)[i]);
      std::printf("     WHISKY  % .6e % .6e % .6e % .6e % .6e\n", rp[0], rp[1],
                  rp[2], rp[3], rp[4]);
      std::printf("   flux (D,Sx,Sy,Sz,tau):\n");
      std::printf("     F(u_L).x % .6e % .6e % .6e % .6e % .6e\n", fpl_d,
                  fpl_sx, fpl_sy, fpl_sz, fpl_tau);
      std::printf("     SpECTRE  % .6e % .6e % .6e % .6e % .6e\n",
                  get(bc.tilde_d)[i], bc.tilde_s.get(0)[i],
                  bc.tilde_s.get(1)[i], bc.tilde_s.get(2)[i],
                  get(bc.tilde_tau)[i]);
      std::printf("     WHISKY   % .6e % .6e % .6e % .6e % .6e\n", fw_d, fw_sx,
                  fw_sy, fw_sz, fw_tau);
    }
  }

  std::printf("\n=== eigenvalue agreement (u_L) ===\n");
  lam_p.report("lambda+");
  lam_m.report("lambda-");
  lam_0.report("lambda0");
  std::printf(
      "=== acoustic right-eigenvector agreement (u_L, D-normalized) "
      "===\n");
  rvecp.report("R+");
  rvecm.report("R-");
  std::printf("=== assembled Marquina numerical flux agreement ===\n");
  f_d.report("F[D]");
  f_sx.report("F[Sx]");
  f_sy.report("F[Sy]");
  f_sz.report("F[Sz]");
  f_tau.report("F[tau]");

  const double flux_max = std::max(
      {f_d.max_abs, f_sx.max_abs, f_sy.max_abs, f_sz.max_abs, f_tau.max_abs});
  const double lam_max =
      std::max({lam_p.max_abs, lam_m.max_abs, lam_0.max_abs});
  std::printf("\nSUMMARY: max |lambda diff| = %.3e, max |flux diff| = %.3e\n",
              lam_max, flux_max);
  return 0;
}
