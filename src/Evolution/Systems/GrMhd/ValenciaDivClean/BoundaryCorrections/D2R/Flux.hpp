// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include <array>
#include <string>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/D2R/Types.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE {

// Compiled correctness backend: flat SR, gamma-law 1<Gamma<=2, common Bn.
// Numerical entropy support is not an interval or full-update certificate.
struct FaceResult {
  bool supported = false;
  bool active = false;
  bool causal_fallback = false;
  // A literal, or (an unsupported entropy model's composite reason) this
  // thread's buffer, valid until the next such face in the thread.
  const char* status = "";
  State flux{};
  State baseline{};
  State direction{};
  State left_tau{};
  State right_tau{};
  std::array<double, 2> bounds{};
  double beta = 0.;
  double alpha = 0.;
  // The consistency bound's scaling of the response direction (1 = not
  // bound); the bound itself is C s_max |U_R - U_L|, see Flux.cpp.
  double consistency = 1.;
  // The resolved-jump weight min(1, (|dU| / (eta |U|))^2) applied to the
  // response (1 = not damped; eta = SPECTRE_D2R_JUMP_FLOOR, 0 = off).
  double jump_weight = 1.;
  int core_calls = 0;
  int native_status = 0;
  int baseline_outcome = -1;
};

FaceResult solve_face(const State& left, const State& right, double bn,
                      double gamma);

}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE

struct PlutoD2RDiagnostics;
// The batch entry points of each precision build; the extern "C" functions
// below dispatch to one of them (SPECTRE_D2R_PRECISION: unset or "double" ->
// d2r_detail, "mp" -> the 160-digit oracle d2r_detail_mp).
namespace grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail {
int pluto_d2r_flux_impl(int npts, const double* left, const double* right,
                        double gamma, double* flux, double* pressure);
void reset_diagnostics_impl();
PlutoD2RDiagnostics diagnostics_impl();
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail
namespace grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail_mp {
int pluto_d2r_flux_impl(int npts, const double* left, const double* right,
                        double gamma, double* flux, double* pressure);
void reset_diagnostics_impl();
PlutoD2RDiagnostics diagnostics_impl();
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail_mp

extern "C" {
// PLUTO shim-compatible eight-component interface. P=(rho,vn,vt,vb,Bn,Bt,Bb,p),
// F=(D,mn,mt,mb,Bn,Bt,Bb,tau). Complete normal momentum is included, so the
// separate pressure is zero. Unsupported points are NaN for the outer HLL
// fallback; the integer return is the number of supported points.
int pluto_d2r_flux(int npts, const double* left, const double* right,
                   double gamma, double* flux, double* pressure);
struct PlutoD2RDiagnostics {
  int faces;
  int supported;
  int active;
  int causal_fallback;
  int unsupported;
};
void pluto_d2r_reset_diagnostics();
PlutoD2RDiagnostics pluto_d2r_diagnostics();
}
