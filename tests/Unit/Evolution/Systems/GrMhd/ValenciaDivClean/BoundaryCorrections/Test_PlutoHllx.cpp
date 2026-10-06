// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoHllx.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/pluto/pluto_hlld_shim.h"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

namespace {
namespace helpers = TestHelpers::evolution::dg;
namespace bc = grmhd::ValenciaDivClean::BoundaryCorrections;

using PlutoFlux = int (*)(int, const double*, const double*, double, double*,
                          double*);

// One interface: PLUTO RMHD primitives {rho, vx, vy, vz, Bx, By, Bz, p_gas},
// normal along x, and the flux and total pressure PLUTO returns for it.
struct ReferenceCase {
  std::string name;
  std::array<double, 8> left;
  std::array<double, 8> right;
  double adiabatic_index;
  // {RHO, MX1, MX2, MX3, BX1, BX2, BX3, ENG} then the pressure term, for the
  // seven-wave solver (hllx) and for PLUTO's HLLD on the same face
  std::array<double, 9> hllx;
  std::array<double, 9> hlld;
};

// Reference fluxes from the standalone face driver
// (notes/projects/active/hllx_rmhd/scripts/pluto_face_driver/rmhd_face_driver.c,
// gcc -O2 -ffp-contract=off) built from the PLUTO 4.4-patch4 release tree with
// hllx.c added to it (local PLUTO repository commit dca07c7, 2026-10-06), which
// prepares each state the way PLUTO's own States/flat_states.c does and calls
// HLLX_Solver and HLLD_Solver directly, not through the shim. The numpy
// reference (rmhd_py/hllx_rmhd.py) agrees with these to 1e-14. Configuration:
// the vendored pluto/definitions.h.
//
// What each face exercises: st1 (Balsara 1) and the mild face reach a
// seven-wave root, so hllx differs from HLLD by 1e-2 there; bal5 reaches one
// at intermediate slow speeds (continuation); bx0 has B^x = 0, where the slow
// fronts coalesce with the contact and hllx keeps HLLD's flux exactly; on the
// large-B_x Dai-Woodward face PLUTO's HLLD secant fails at its first probe and
// both are the HLL flux (this is the initial face only: in the evolution of
// that tube hllx resolves the slow shocks that HLLD smears).
const std::array<ReferenceCase, 5> reference_cases{{
    {"st1 (Balsara 1)",  // hllx outcome: hllx
     {{1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0}},
     {{0.125, 0.0, 0.0, 0.0, 0.5, -1.0, 0.0, 0.1}},
     2.0,
      {{2.3394947029238541e-01, -9.4737768390383748e-01,
       -3.5417383179998840e-01, 0.0, 0.0, 2.5034348690188002e-01, 0.0,
       4.8371253910887413e-01, 1.6250000000000000e+00}},
      {{2.4453400539364434e-01, -9.6296630705276220e-01,
       -3.6853136801338132e-01, 0.0, 0.0, 2.4936651533216450e-01, 0.0,
       4.9904982334815257e-01, 1.6250000000000000e+00}}},
    {"bal5 (Balsara 5)",  // hllx outcome: hllx-cont
     {{1.08, 0.4, 0.3, 0.2, 2.0, 0.3, 0.3, 0.95}},
     {{1.0, -0.45, -0.2, 0.2, 2.0, -0.7, 0.5, 1.0}},
     5.0 / 3.0,
      {{-9.3924482892466937e-02, -8.1472333565555655e-01,
       2.3059513832197176e+00, -1.3184281762926977e+00, 0.0,
       3.4429144684791713e-01, -4.4867526951955261e-01, -1.0594219147932524e+00,
       2.9182750000000004e+00}},
      {{-8.2986651036617565e-02, -4.4472312969017525e-01,
       2.2415290642175152e+00, -1.2856772860785275e+00, -2.0170121894108265e-16,
       3.4026459025312045e-01, -4.4049785498744509e-01, -1.0136171975617605e+00,
       2.9182750000000004e+00}}},
    {"bx0 (B^x = 0)",  // hllx outcome: hlld-bx0
     {{1.0, 0.1, 0.2, -0.1, 0.0, 1.0, 0.5, 1.0}},
     {{0.5, -0.1, 0.0, 0.2, 0.0, -0.5, 0.3, 0.5}},
     5.0 / 3.0,
      {{1.6752728460253963e-01, -1.5991856496983331e-01, 1.3719621332967982e-01,
       -9.2961665958901415e-02, 0.0, 1.6242372862707688e-01,
       8.1211864313538440e-02, 5.9778662004548178e-01, 1.5987499999999999e+00}},
      {{1.6752728460253963e-01, -1.5991856496983331e-01, 1.3719621332967982e-01,
       -9.2961665958901415e-02, 0.0, 1.6242372862707688e-01,
       8.1211864313538440e-02, 5.9778662004548178e-01, 1.5987499999999999e+00}}},
    {"dwbigbx (ZPL26 large-B_x Dai-Woodward, lam = 100)",  // hllx outcome: hll
     {{1.08, 0.012, 0.0001, 0.005, 0.5641895835477563, 0.010155412503859613, 0.005641895835477563, 9.499999999999999e-05}},
     {{1.0, 0.0, 0.0, 0.0, 0.5641895835477563, 0.011283791670955126, 0.005641895835477563, 0.0001}},
     5.0 / 3.0,
      {{2.6159988337563032e-02, -3.1504628232975779e-01,
       -6.0297654621202564e-03, -1.4415165239525425e-03, 0.0,
       -2.4453405034741767e-04, -1.3766225838565253e-03, 1.3095371346998835e-05,
       1.5932407588473776e-01}},
      {{2.6159988337563032e-02, -3.1504628232975779e-01,
       -6.0297654621202564e-03, -1.4415165239525425e-03, 0.0,
       -2.4453405034741767e-04, -1.3766225838565253e-03, 1.3095371346998835e-05,
       1.5932407588473776e-01}}},
    {"mild face 0 (seven-wave root)",  // hllx outcome: hllx
     {{0.5694935630934498, 0.004014453606245651, 0.059431691807168265, 0.189882725762603, -0.6752715219083351, 0.37974792728641055, 0.14798803413693906, 0.030001655811010528}},
     {{0.286313851637016, 0.09271697196233884, -0.08557999545150088, 0.17167786160990514, -0.6752715219083351, 0.18418346361074558, 0.03798081265203704, 0.04480638970977093}},
     2.0,
      {{1.8897906461003026e-02, -4.5580836771582994e-01, 2.1722868763062000e-01,
       7.5098099107924920e-02, 0.0, 7.3691336529367227e-02,
       1.6161744563624439e-01, 4.2857833326765842e-02, 3.2988321666838161e-01}},
      {{2.9856746433569770e-02, -4.6385569356016537e-01, 2.1947697494940560e-01,
       7.8198191050208699e-02, 7.8256380772450675e-17, 7.4662254976905124e-02,
       1.6266575735502820e-01, 4.4434044256312075e-02, 3.2988321666838161e-01}}},
}};

// Does the vendored PLUTO, through the shim, still give unmodified PLUTO's
// flux? Several batch sizes: 1 is the scalar path, 169 a realistic face, and
// 2500 makes the shim chunk (its capacity is 1024).
// The seven-wave flux is a Newton root found to a residual of 1e-10 relative
// to the face's own scales, so a different compiler or contraction policy may
// stop at an iterate that differs from the reference by more than round-off;
// 1e-9 is still five orders below the hllx - HLLD difference it is meant to
// pin. HLLD's values reproduce to all digits, as in Test_PlutoHlld.
const Approx hllx_approx = Approx::custom().epsilon(1.0e-9).scale(1.0);

void test_matches_reference_pluto() {
  // PLUTO raises FPEs in its allocator and root finds; see Test_PlutoHlld.
  const ScopedFpeState fpe_off(false);
  for (const auto& reference : reference_cases) {
    CAPTURE(reference.name);
    for (const auto& [solver_name, solver, expected, tolerance] :
         {std::tuple{std::string{"hllx"}, PlutoFlux{&pluto_hllx_flux},
                     reference.hllx, hllx_approx},
          std::tuple{std::string{"hlld"}, PlutoFlux{&pluto_hlld_flux},
                     reference.hlld, approx}}) {
      CAPTURE(solver_name);
      for (const int npts : {1, 169, 2500}) {
        CAPTURE(npts);
        const auto num_points = static_cast<size_t>(npts);
        std::vector<double> batch_l(num_points * 8);
        std::vector<double> batch_r(num_points * 8);
        std::vector<double> flux(num_points * 8);
        std::vector<double> press(num_points);
        for (size_t i = 0; i < num_points; ++i) {
          for (size_t nv = 0; nv < 8; ++nv) {
            batch_l[i * 8 + nv] = gsl::at(reference.left, nv);
            batch_r[i * 8 + nv] = gsl::at(reference.right, nv);
          }
        }
        CHECK(solver(npts, batch_l.data(), batch_r.data(),
                     reference.adiabatic_index, flux.data(),
                     press.data()) == 0);
        for (size_t i = 0; i < num_points; ++i) {
          for (size_t nv = 0; nv < 8; ++nv) {
            CHECK(flux[i * 8 + nv] == tolerance(gsl::at(expected, nv)));
          }
          CHECK(press[i] == tolerance(expected[8]));
        }
      }
    }
  }
}

// hllx.c keeps file-scope state (the outcome counters, the current face's S_c
// and B_x, the seven-wave switch) and lazily-allocated function statics (Uhll,
// Fhll, Vhll, lamL, lamR), all made _Thread_local in the vendored copy.
// As in Test_PlutoHlld, every thread solves a DIFFERENT state and is compared
// with that state's own single-threaded answer; threads solving the same
// state could not detect a race.
void test_thread_safety(const PlutoFlux solver, const std::string& name) {
  CAPTURE(name);
  const ScopedFpeState fpe_off(false);
  constexpr size_t num_threads = 8;
  constexpr size_t npts = 169;
  const auto make_state = [](size_t seed, std::vector<double>* l,
                             std::vector<double>* r) {
    std::mt19937 gen(static_cast<unsigned>(seed) + 1u);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::array<double, 8> sl{};
    std::array<double, 8> sr{};
    sl[0] = 0.5 + dist(gen);
    sr[0] = 0.5 + dist(gen);
    for (size_t k = 1; k < 4; ++k) {
      gsl::at(sl, k) = -0.3 + 0.6 * dist(gen);
      gsl::at(sr, k) = -0.3 + 0.6 * dist(gen);
    }
    sl[4] = -1.0 + 2.0 * dist(gen);
    sr[4] = sl[4];  // normal B is continuous
    for (size_t k = 5; k < 7; ++k) {
      gsl::at(sl, k) = -1.0 + 2.0 * dist(gen);
      gsl::at(sr, k) = -1.0 + 2.0 * dist(gen);
    }
    sl[7] = 0.5 + dist(gen);
    sr[7] = 0.5 + dist(gen);
    l->resize(npts * 8);
    r->resize(npts * 8);
    for (size_t i = 0; i < npts; ++i) {
      for (size_t k = 0; k < 8; ++k) {
        (*l)[i * 8 + k] = gsl::at(sl, k);
        (*r)[i * 8 + k] = gsl::at(sr, k);
      }
    }
  };
  // One solve of state `seed` on the calling thread.
  const auto solve = [&solver, &make_state](const size_t seed) {
    std::vector<double> l{};
    std::vector<double> r{};
    std::vector<double> f(npts * 8);
    std::vector<double> pr(npts);
    make_state(seed, &l, &r);
    solver(static_cast<int>(npts), l.data(), r.data(), 5.0 / 3.0, f.data(),
           pr.data());
    return f;
  };
  std::vector<std::vector<double>> reference(num_threads);
  for (size_t t = 0; t < num_threads; ++t) {
    reference[t] = solve(t);
  }
  std::vector<double> worst(num_threads, 0.0);
  std::vector<std::thread> threads{};
  threads.reserve(num_threads);
  for (size_t t = 0; t < num_threads; ++t) {
    threads.emplace_back([t, &solve, &reference, &worst]() {
      const std::vector<double> f = solve(t);
      double w = 0.0;
      for (size_t i = 0; i < f.size(); ++i) {
        const double ref = reference[t][i];
        w = std::max(w, std::abs(f[i] - ref) / (1.0 + std::abs(ref)));
      }
      worst[t] = w;
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  for (size_t t = 0; t < num_threads; ++t) {
    CHECK(worst[t] == 0.0);
  }
}

// End to end: dg_package_data on each side, then dg_boundary_terms, must hand
// back PLUTO's own flux. This exercises what the shim test cannot: the
// adiabatic index recovered from (rho, eps, p), the rotation into the normal
// frame and back, the pressure term added onto the normal momentum, and which
// PLUTO solver the variant selects. The normal is +x, so PLUTO's frame is the
// inertial one (make_triad gives t1 = y, t2 = z) and the weak-form correction
// is the flux itself.
void test_end_to_end() {
  for (const auto& reference : reference_cases) {
    CAPTURE(reference.name);
    const auto& expected = reference.hllx;
    const double gamma = reference.adiabatic_index;
    const auto eos =
        EquationsOfState::IdealFluid<true>{gamma}.promote_to_3d_eos();
    const bc::PlutoHllx solver{1.0e-30, 1.0e-12};

    struct Packaged {
      Scalar<DataVector> tilde_d{DataVector{1_st, 0.0}};
      Scalar<DataVector> tilde_ye{DataVector{1_st, 0.0}};
      Scalar<DataVector> tilde_tau{DataVector{1_st, 0.0}};
      tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{1_st, 0.0};
      tnsr::I<DataVector, 3, Frame::Inertial> tilde_b{1_st, 0.0};
      Scalar<DataVector> tilde_phi{DataVector{1_st, 0.0}};
      Scalar<DataVector> nf_tilde_d{DataVector{1_st, 0.0}};
      Scalar<DataVector> nf_tilde_ye{DataVector{1_st, 0.0}};
      Scalar<DataVector> nf_tilde_tau{DataVector{1_st, 0.0}};
      tnsr::i<DataVector, 3, Frame::Inertial> nf_tilde_s{1_st, 0.0};
      tnsr::I<DataVector, 3, Frame::Inertial> nf_tilde_b{1_st, 0.0};
      Scalar<DataVector> nf_tilde_phi{DataVector{1_st, 0.0}};
      Scalar<DataVector> largest_out{DataVector{1_st, 0.0}};
      Scalar<DataVector> largest_in{DataVector{1_st, 0.0}};
      Scalar<DataVector> fast_out{DataVector{1_st, 0.0}};
      Scalar<DataVector> fast_in{DataVector{1_st, 0.0}};
      tnsr::i<DataVector, 3, Frame::Inertial> normal{1_st, 0.0};
      Scalar<DataVector> flatness{DataVector{1_st, 0.0}};
      Scalar<DataVector> rho{DataVector{1_st, 0.0}};
      tnsr::I<DataVector, 3, Frame::Inertial> velocity{1_st, 0.0};
      Scalar<DataVector> pressure{DataVector{1_st, 0.0}};
      Scalar<DataVector> lorentz_factor{DataVector{1_st, 0.0}};
      Scalar<DataVector> eps{DataVector{1_st, 0.0}};
    };
    // Only the primitives reach PLUTO; the conserved variables and their
    // fluxes feed the HLL layer (TildeYe, TildePhi, the normal field), which
    // is not checked here, so they are left at zero.
    const auto package = [&](const std::array<double, 8>& prim,
                             const double normal_sign) {
      Packaged out{};
      const double rho = prim[0];
      const double pressure = prim[7];
      const double eps = pressure / ((gamma - 1.0) * rho);
      const double v_squared =
          prim[1] * prim[1] + prim[2] * prim[2] + prim[3] * prim[3];
      const Scalar<DataVector> rest_mass_density{DataVector{1_st, rho}};
      const Scalar<DataVector> specific_internal_energy{DataVector{1_st, eps}};
      const Scalar<DataVector> pressure_scalar{DataVector{1_st, pressure}};
      const Scalar<DataVector> lorentz_factor{
          DataVector{1_st, 1.0 / std::sqrt(1.0 - v_squared)}};
      const Scalar<DataVector> electron_fraction{DataVector{1_st, 0.1}};
      const Scalar<DataVector> temperature =
          eos->temperature_from_density_and_energy(
              rest_mass_density, specific_internal_energy, electron_fraction);
      tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{1_st, 0.0};
      tnsr::i<DataVector, 3, Frame::Inertial> velocity_one_form{1_st, 0.0};
      tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{1_st, 0.0};
      for (size_t i = 0; i < 3; ++i) {
        spatial_velocity.get(i) = gsl::at(prim, 1 + i);
        velocity_one_form.get(i) = gsl::at(prim, 1 + i);
        magnetic_field.get(i) = gsl::at(prim, 4 + i);
      }
      tnsr::i<DataVector, 3, Frame::Inertial> normal_covector{1_st, 0.0};
      tnsr::I<DataVector, 3, Frame::Inertial> normal_vector{1_st, 0.0};
      get<0>(normal_covector) = normal_sign;
      get<0>(normal_vector) = normal_sign;
      const Scalar<DataVector> lapse{DataVector{1_st, 1.0}};
      const tnsr::I<DataVector, 3, Frame::Inertial> shift{1_st, 0.0};
      const Scalar<DataVector> zero_scalar{DataVector{1_st, 0.0}};
      const tnsr::i<DataVector, 3, Frame::Inertial> zero_covector{1_st, 0.0};
      const tnsr::I<DataVector, 3, Frame::Inertial> zero_vector{1_st, 0.0};
      const tnsr::Ij<DataVector, 3, Frame::Inertial> zero_flux_s{1_st, 0.0};
      const tnsr::IJ<DataVector, 3, Frame::Inertial> zero_flux_b{1_st, 0.0};
      solver.dg_package_data(
          make_not_null(&out.tilde_d), make_not_null(&out.tilde_ye),
          make_not_null(&out.tilde_tau), make_not_null(&out.tilde_s),
          make_not_null(&out.tilde_b), make_not_null(&out.tilde_phi),
          make_not_null(&out.nf_tilde_d), make_not_null(&out.nf_tilde_ye),
          make_not_null(&out.nf_tilde_tau), make_not_null(&out.nf_tilde_s),
          make_not_null(&out.nf_tilde_b), make_not_null(&out.nf_tilde_phi),
          make_not_null(&out.largest_out), make_not_null(&out.largest_in),
          make_not_null(&out.fast_out), make_not_null(&out.fast_in),
          make_not_null(&out.normal), make_not_null(&out.flatness),
          make_not_null(&out.rho), make_not_null(&out.velocity),
          make_not_null(&out.pressure), make_not_null(&out.lorentz_factor),
          make_not_null(&out.eps), zero_scalar, zero_scalar, zero_scalar,
          zero_covector, magnetic_field, zero_scalar, zero_vector, zero_vector,
          zero_vector, zero_flux_s, zero_flux_b, zero_vector, lapse, shift,
          velocity_one_form, rest_mass_density, electron_fraction, temperature,
          spatial_velocity, specific_internal_energy, pressure_scalar,
          lorentz_factor, normal_covector, normal_vector, std::nullopt,
          std::nullopt, *eos);
      return out;
    };
    const auto interior = package(reference.left, 1.0);
    const auto exterior = package(reference.right, -1.0);

    Scalar<DataVector> g_d{DataVector{1_st, 0.0}};
    Scalar<DataVector> g_ye{DataVector{1_st, 0.0}};
    Scalar<DataVector> g_tau{DataVector{1_st, 0.0}};
    tnsr::i<DataVector, 3, Frame::Inertial> g_s{1_st, 0.0};
    tnsr::I<DataVector, 3, Frame::Inertial> g_b{1_st, 0.0};
    Scalar<DataVector> g_phi{DataVector{1_st, 0.0}};
    solver.dg_boundary_terms(
        make_not_null(&g_d), make_not_null(&g_ye), make_not_null(&g_tau),
        make_not_null(&g_s), make_not_null(&g_b), make_not_null(&g_phi),
        interior.tilde_d, interior.tilde_ye, interior.tilde_tau,
        interior.tilde_s, interior.tilde_b, interior.tilde_phi,
        interior.nf_tilde_d, interior.nf_tilde_ye, interior.nf_tilde_tau,
        interior.nf_tilde_s, interior.nf_tilde_b, interior.nf_tilde_phi,
        interior.largest_out, interior.largest_in, interior.fast_out,
        interior.fast_in, interior.normal, interior.flatness, interior.rho,
        interior.velocity, interior.pressure, interior.lorentz_factor,
        interior.eps, exterior.tilde_d, exterior.tilde_ye, exterior.tilde_tau,
        exterior.tilde_s, exterior.tilde_b, exterior.tilde_phi,
        exterior.nf_tilde_d, exterior.nf_tilde_ye, exterior.nf_tilde_tau,
        exterior.nf_tilde_s, exterior.nf_tilde_b, exterior.nf_tilde_phi,
        exterior.largest_out, exterior.largest_in, exterior.fast_out,
        exterior.fast_in, exterior.normal, exterior.flatness, exterior.rho,
        exterior.velocity, exterior.pressure, exterior.lorentz_factor,
        exterior.eps, dg::Formulation::WeakInertial);

    // PLUTO carries the total pressure outside the momentum flux; the wrapper
    // adds it back onto the normal component.
    CHECK(get(g_d)[0] == hllx_approx(expected[0]));
    CHECK(get<0>(g_s)[0] == hllx_approx(expected[1] + expected[8]));
    CHECK(get<1>(g_s)[0] == hllx_approx(expected[2]));
    CHECK(get<2>(g_s)[0] == hllx_approx(expected[3]));
    CHECK(get<1>(g_b)[0] == hllx_approx(expected[5]));
    CHECK(get<2>(g_b)[0] == hllx_approx(expected[6]));
    CHECK(get(g_tau)[0] == hllx_approx(expected[7]));
  }
}

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.PlutoHllx",
                  "[Unit][GrMhd]") {
  PUPable_reg(bc::PlutoHllx);
  test_matches_reference_pluto();
  test_thread_safety(&pluto_hllx_flux, "hllx");
  test_end_to_end();

  MAKE_GENERATOR(gen);
  using system = grmhd::ValenciaDivClean::System;
  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>>
      ranges{std::array{0.3, 1.0}, std::array{0.01, 0.02}};
  const tuples::TaggedTuple<hydro::Tags::GrmhdEquationOfState> volume_data{
      EquationsOfState::IdealFluid<true>{4.0 / 3.0}.promote_to_3d_eos()};
  // Conservation: the numerical flux must be single valued across the
  // interface.
  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen), bc::PlutoHllx{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);
  // Factory creation + round trip through a base-class pointer.
  const auto pluto_hllx =
      TestHelpers::test_factory_creation<evolution::BoundaryCorrection,
                                         bc::PlutoHllx>(
          "PlutoHllx:\n"
          "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
          "  LightSpeedDensityCutoff: 1.0e-8\n");
  CHECK(dynamic_cast<const bc::PlutoHllx&>(*pluto_hllx) ==
        bc::PlutoHllx{1.0e-30, 1.0e-8});
  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen), dynamic_cast<const bc::PlutoHllx&>(*pluto_hllx),
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);
  CHECK_FALSE(bc::PlutoHllx{1.0e-30, 1.0e-8} !=
              bc::PlutoHllx{1.0e-30, 1.0e-8});
  CHECK(bc::PlutoHllx{1.0e-30, 1.0e-8} != bc::PlutoHllx{2.0e-30, 1.0e-8});
  CHECK(bc::PlutoHllx{1.0e-30, 1.0e-8} != bc::PlutoHllx{1.0e-30, 2.0e-8});
  test_serialization(bc::PlutoHllx{1.0e-30, 1.0e-8});
}
}  // namespace
