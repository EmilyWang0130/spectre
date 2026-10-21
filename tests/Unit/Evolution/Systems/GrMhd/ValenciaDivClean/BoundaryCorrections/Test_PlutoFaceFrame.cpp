// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoD2R.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoFaceFrame.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoHllc.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoHlld.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoHllx.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/pluto/pluto_hlld_shim.h"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"

// Tests of the GR face frame of the PLUTO-backed boundary corrections
// (`PlutoFaceFrame.hpp`; the contract is the HLLx campaign's
// gr_tetrad_adapter.md). The limiting cases of its section 7 each get a test:
// flat space, a lapse alone, a shift alone (and the equivalent mesh motion), a
// constant curved metric with a uniform state, and Minkowski space in linear
// non-Cartesian coordinates -- the last the only reference that reaches the
// HLLD and HLLx branches on a curved face.

namespace {
namespace bc = grmhd::ValenciaDivClean::BoundaryCorrections;
using Eos = EquationsOfState::EquationOfState<true, 3>;
using Matrix3 = std::array<std::array<double, 3>, 3>;
constexpr double adiabatic_index = 5.0 / 3.0;

// ---------------------------------------------------------------------------
// States and geometry of one face point.
// ---------------------------------------------------------------------------
struct Geometry {
  double lapse = 1.0;
  std::array<double, 3> shift{};
  Matrix3 metric{{{{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}};
  // The mesh velocity; zero means a static mesh (no mesh velocity at all).
  std::array<double, 3> mesh_velocity{};
};

struct Primitive {
  double rho = 1.0;
  double pressure = 1.0;
  std::array<double, 3> velocity{};  // coordinate v^i
  std::array<double, 3> field{};     // undensitized B^i
  double phi = 0.0;                  // undensitized Phi
  double ye = 0.1;
};

struct Packaged {
  Scalar<DataVector> d{DataVector{1_st, 0.0}};
  Scalar<DataVector> ye{DataVector{1_st, 0.0}};
  Scalar<DataVector> tau{DataVector{1_st, 0.0}};
  tnsr::i<DataVector, 3> s{1_st, 0.0};
  tnsr::I<DataVector, 3> b{1_st, 0.0};
  Scalar<DataVector> phi{DataVector{1_st, 0.0}};
  Scalar<DataVector> nf_d{DataVector{1_st, 0.0}};
  Scalar<DataVector> nf_ye{DataVector{1_st, 0.0}};
  Scalar<DataVector> nf_tau{DataVector{1_st, 0.0}};
  tnsr::i<DataVector, 3> nf_s{1_st, 0.0};
  tnsr::I<DataVector, 3> nf_b{1_st, 0.0};
  Scalar<DataVector> nf_phi{DataVector{1_st, 0.0}};
  Scalar<DataVector> largest_out{DataVector{1_st, 0.0}};
  Scalar<DataVector> largest_in{DataVector{1_st, 0.0}};
  Scalar<DataVector> fast_out{DataVector{1_st, 0.0}};
  Scalar<DataVector> fast_in{DataVector{1_st, 0.0}};
  tnsr::i<DataVector, 3> normal{1_st, 0.0};
  Scalar<DataVector> flatness{DataVector{1_st, 0.0}};
  Scalar<DataVector> iface_lapse{DataVector{1_st, 0.0}};
  Scalar<DataVector> iface_shift_n{DataVector{1_st, 0.0}};
  tnsr::ii<DataVector, 3> iface_metric{1_st, 0.0};
  Scalar<DataVector> rho{DataVector{1_st, 0.0}};
  tnsr::I<DataVector, 3> velocity{1_st, 0.0};
  Scalar<DataVector> pressure{DataVector{1_st, 0.0}};
  Scalar<DataVector> lorentz{DataVector{1_st, 0.0}};
  Scalar<DataVector> eps{DataVector{1_st, 0.0}};
};

// The boundary correction of every evolved variable at one point, flattened:
// D, Ye, Tau, S_x..z, B^x..z, Phi.
using Correction = std::array<double, 10>;

double norm_inf(const Correction& c) {
  double m = 0.0;
  for (const double x : c) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

// A unit normal covector (in gamma) along the covector `direction`.
std::array<double, 3> unit_normal(const Geometry& geometry,
                                  const std::array<double, 3>& direction) {
  tnsr::ii<double, 3> g{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      g.get(i, j) = gsl::at(gsl::at(geometry.metric, i), j);
    }
  }
  const auto [det, inv] = determinant_and_inverse(g);
  (void)det;
  double norm_squared = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      norm_squared +=
          inv.get(i, j) * gsl::at(direction, i) * gsl::at(direction, j);
    }
  }
  std::array<double, 3> n{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(n, i) = gsl::at(direction, i) / std::sqrt(norm_squared);
  }
  return n;
}

// Package one side, with conserved variables and normal fluxes from
// SpECTRE's own ConservativeFromPrimitive and ComputeFluxes (and the
// mesh-velocity shift F^i -= v_g^i U that SpECTRE applies before the
// boundary correction sees the fluxes).
template <typename Solver>
Packaged package(const Solver& solver, const Primitive& prim,
                 const Geometry& geometry,
                 const std::array<double, 3>& normal_covector, const Eos& eos) {
  Packaged out{};
  tnsr::ii<DataVector, 3> metric{1_st, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      metric.get(i, j) = gsl::at(gsl::at(geometry.metric, i), j);
    }
  }
  const auto [det, inv_metric] = determinant_and_inverse(metric);
  const Scalar<DataVector> sqrt_det{sqrt(get(det))};
  const Scalar<DataVector> lapse{DataVector{1_st, geometry.lapse}};
  tnsr::I<DataVector, 3> shift{1_st, 0.0};
  tnsr::I<DataVector, 3> velocity{1_st, 0.0};
  tnsr::I<DataVector, 3> field{1_st, 0.0};
  tnsr::i<DataVector, 3> velocity_one_form{1_st, 0.0};
  tnsr::i<DataVector, 3> normal{1_st, 0.0};
  tnsr::I<DataVector, 3> normal_vector{1_st, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    shift.get(i) = gsl::at(geometry.shift, i);
    velocity.get(i) = gsl::at(prim.velocity, i);
    field.get(i) = gsl::at(prim.field, i);
    normal.get(i) = gsl::at(normal_covector, i);
  }
  double v2 = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      velocity_one_form.get(i) += metric.get(i, j) * velocity.get(j);
      normal_vector.get(i) += inv_metric.get(i, j) * normal.get(j);
      v2 += metric.get(i, j)[0] * gsl::at(prim.velocity, i) *
            gsl::at(prim.velocity, j);
    }
  }
  REQUIRE(v2 < 1.0);
  const Scalar<DataVector> lorentz{DataVector{1_st, 1.0 / std::sqrt(1.0 - v2)}};
  const Scalar<DataVector> rho{DataVector{1_st, prim.rho}};
  const Scalar<DataVector> pressure{DataVector{1_st, prim.pressure}};
  const Scalar<DataVector> eps{
      DataVector{1_st, prim.pressure / ((adiabatic_index - 1.0) * prim.rho)}};
  const Scalar<DataVector> ye{DataVector{1_st, prim.ye}};
  const Scalar<DataVector> phi{DataVector{1_st, prim.phi}};
  const Scalar<DataVector> temperature =
      eos.temperature_from_density_and_energy(rho, eps, ye);

  Scalar<DataVector> u_d{DataVector{1_st, 0.0}};
  Scalar<DataVector> u_ye{DataVector{1_st, 0.0}};
  Scalar<DataVector> u_tau{DataVector{1_st, 0.0}};
  tnsr::i<DataVector, 3> u_s{1_st, 0.0};
  tnsr::I<DataVector, 3> u_b{1_st, 0.0};
  Scalar<DataVector> u_phi{DataVector{1_st, 0.0}};
  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&u_d), make_not_null(&u_ye), make_not_null(&u_tau),
      make_not_null(&u_s), make_not_null(&u_b), make_not_null(&u_phi), rho, ye,
      eps, pressure, velocity, lorentz, field, sqrt_det, metric, phi);
  tnsr::I<DataVector, 3> f_d{1_st, 0.0};
  tnsr::I<DataVector, 3> f_ye{1_st, 0.0};
  tnsr::I<DataVector, 3> f_tau{1_st, 0.0};
  tnsr::Ij<DataVector, 3> f_s{1_st, 0.0};
  tnsr::IJ<DataVector, 3> f_b{1_st, 0.0};
  tnsr::I<DataVector, 3> f_phi{1_st, 0.0};
  grmhd::ValenciaDivClean::ComputeFluxes::apply(
      make_not_null(&f_d), make_not_null(&f_ye), make_not_null(&f_tau),
      make_not_null(&f_s), make_not_null(&f_b), make_not_null(&f_phi), u_d,
      u_ye, u_tau, u_s, u_b, u_phi, lapse, shift, sqrt_det, metric, inv_metric,
      pressure, velocity, lorentz, field);

  std::optional<tnsr::I<DataVector, 3>> mesh_velocity{};
  std::optional<Scalar<DataVector>> normal_dot_mesh_velocity{};
  if (geometry.mesh_velocity != std::array<double, 3>{}) {
    mesh_velocity = tnsr::I<DataVector, 3>{1_st, 0.0};
    normal_dot_mesh_velocity = Scalar<DataVector>{DataVector{1_st, 0.0}};
    for (size_t i = 0; i < 3; ++i) {
      const double vg = gsl::at(geometry.mesh_velocity, i);
      mesh_velocity->get(i) = vg;
      get(*normal_dot_mesh_velocity) += vg * normal.get(i);
      f_d.get(i) -= vg * get(u_d);
      f_ye.get(i) -= vg * get(u_ye);
      f_tau.get(i) -= vg * get(u_tau);
      f_phi.get(i) -= vg * get(u_phi);
      for (size_t j = 0; j < 3; ++j) {
        f_s.get(i, j) -= vg * u_s.get(j);
        f_b.get(i, j) -= vg * u_b.get(j);
      }
    }
  }

  solver.dg_package_data(
      make_not_null(&out.d), make_not_null(&out.ye), make_not_null(&out.tau),
      make_not_null(&out.s), make_not_null(&out.b), make_not_null(&out.phi),
      make_not_null(&out.nf_d), make_not_null(&out.nf_ye),
      make_not_null(&out.nf_tau), make_not_null(&out.nf_s),
      make_not_null(&out.nf_b), make_not_null(&out.nf_phi),
      make_not_null(&out.largest_out), make_not_null(&out.largest_in),
      make_not_null(&out.fast_out), make_not_null(&out.fast_in),
      make_not_null(&out.normal), make_not_null(&out.flatness),
      make_not_null(&out.iface_lapse), make_not_null(&out.iface_shift_n),
      make_not_null(&out.iface_metric), make_not_null(&out.rho),
      make_not_null(&out.velocity), make_not_null(&out.pressure),
      make_not_null(&out.lorentz), make_not_null(&out.eps), u_d, u_ye, u_tau,
      u_s, u_b, u_phi, f_d, f_ye, f_tau, f_s, f_b, f_phi, lapse, shift,
      velocity_one_form, metric, rho, ye, temperature, velocity, eps, pressure,
      lorentz, normal, normal_vector, mesh_velocity, normal_dot_mesh_velocity,
      eos);
  return out;
}

template <typename Solver>
Correction boundary(const Solver& solver, const Packaged& l, const Packaged& r,
                    const dg::Formulation formulation) {
  Scalar<DataVector> g_d{DataVector{1_st, 0.0}};
  Scalar<DataVector> g_ye{DataVector{1_st, 0.0}};
  Scalar<DataVector> g_tau{DataVector{1_st, 0.0}};
  tnsr::i<DataVector, 3> g_s{1_st, 0.0};
  tnsr::I<DataVector, 3> g_b{1_st, 0.0};
  Scalar<DataVector> g_phi{DataVector{1_st, 0.0}};
  solver.dg_boundary_terms(
      make_not_null(&g_d), make_not_null(&g_ye), make_not_null(&g_tau),
      make_not_null(&g_s), make_not_null(&g_b), make_not_null(&g_phi), l.d,
      l.ye, l.tau, l.s, l.b, l.phi, l.nf_d, l.nf_ye, l.nf_tau, l.nf_s, l.nf_b,
      l.nf_phi, l.largest_out, l.largest_in, l.fast_out, l.fast_in, l.normal,
      l.flatness, l.iface_lapse, l.iface_shift_n, l.iface_metric, l.rho,
      l.velocity, l.pressure, l.lorentz, l.eps, r.d, r.ye, r.tau, r.s, r.b,
      r.phi, r.nf_d, r.nf_ye, r.nf_tau, r.nf_s, r.nf_b, r.nf_phi, r.largest_out,
      r.largest_in, r.fast_out, r.fast_in, r.normal, r.flatness, r.iface_lapse,
      r.iface_shift_n, r.iface_metric, r.rho, r.velocity, r.pressure, r.lorentz,
      r.eps, formulation);
  return {{get(g_d)[0], get(g_ye)[0], get(g_tau)[0], get<0>(g_s)[0],
           get<1>(g_s)[0], get<2>(g_s)[0], get<0>(g_b)[0], get<1>(g_b)[0],
           get<2>(g_b)[0], get(g_phi)[0]}};
}

// The interior's packaged normal fluxes, in Correction order.
Correction physical_flux(const Packaged& side) {
  return {{get(side.nf_d)[0], get(side.nf_ye)[0], get(side.nf_tau)[0],
           get<0>(side.nf_s)[0], get<1>(side.nf_s)[0], get<2>(side.nf_s)[0],
           get<0>(side.nf_b)[0], get<1>(side.nf_b)[0], get<2>(side.nf_b)[0],
           get(side.nf_phi)[0]}};
}

// Interior and exterior packaging of one face; `normal` is the interior's
// outward normal covector.
template <typename Solver>
Correction solve_face(
    const Solver& solver, const Primitive& left, const Primitive& right,
    const Geometry& geometry, const std::array<double, 3>& normal,
    const Eos& eos,
    const dg::Formulation formulation = dg::Formulation::WeakInertial) {
  const std::array<double, 3> minus_normal{
      {-normal[0], -normal[1], -normal[2]}};
  const auto l = package(solver, left, geometry, normal, eos);
  const auto r = package(solver, right, geometry, minus_normal, eos);
  return boundary(solver, l, r, formulation);
}

void check_close(const Correction& a, const Correction& b, const double tol,
                 const std::string& what) {
  CAPTURE(what);
  const double scale = std::max({1.0, norm_inf(a), norm_inf(b)});
  for (size_t k = 0; k < a.size(); ++k) {
    CAPTURE(k);
    CAPTURE(gsl::at(a, k));
    CAPTURE(gsl::at(b, k));
    CHECK(std::abs(gsl::at(a, k) - gsl::at(b, k)) <= tol * scale);
  }
}

// ---------------------------------------------------------------------------
// Random states and geometry.
// ---------------------------------------------------------------------------
Matrix3 random_metric(const gsl::not_null<std::mt19937*> gen,
                      const double spread) {
  std::uniform_real_distribution<double> dist(-spread, spread);
  Matrix3 a{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      gsl::at(gsl::at(a, i), j) = (i == j ? 1.0 : 0.0) + dist(*gen);
    }
  }
  Matrix3 g{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        gsl::at(gsl::at(g, i), j) +=
            gsl::at(gsl::at(a, k), i) * gsl::at(gsl::at(a, k), j);
      }
    }
  }
  return g;
}

// A primitive state with |v| < max_speed in the metric g and a field of the
// given scale.
Primitive random_primitive(const gsl::not_null<std::mt19937*> gen,
                           const Matrix3& g, const double max_speed,
                           const double field_scale) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> sym(-1.0, 1.0);
  Primitive p{};
  p.rho = 0.2 + unit(*gen);
  p.pressure = 0.05 + unit(*gen);
  std::array<double, 3> v{{sym(*gen), sym(*gen), sym(*gen)}};
  double v2 = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v2 += gsl::at(gsl::at(g, i), j) * gsl::at(v, i) * gsl::at(v, j);
    }
  }
  const double target = max_speed * unit(*gen);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(p.velocity, i) = gsl::at(v, i) * target / std::sqrt(v2);
    gsl::at(p.field, i) = field_scale * sym(*gen);
  }
  p.ye = 0.05 + 0.4 * unit(*gen);
  return p;
}

std::array<double, 3> random_direction(const gsl::not_null<std::mt19937*> gen) {
  std::uniform_real_distribution<double> sym(-1.0, 1.0);
  return {{sym(*gen), sym(*gen), sym(*gen)}};
}

// ---------------------------------------------------------------------------
// 1. The frame itself.
// ---------------------------------------------------------------------------
void test_frame(const gsl::not_null<std::mt19937*> gen) {
  for (size_t trial = 0; trial < 200; ++trial) {
    Geometry geometry{};
    geometry.metric = random_metric(gen, 0.4);
    const auto normal = unit_normal(geometry, random_direction(gen));
    std::array<double, 6> g6{{geometry.metric[0][0], geometry.metric[0][1],
                              geometry.metric[0][2], geometry.metric[1][1],
                              geometry.metric[1][2], geometry.metric[2][2]}};
    const std::array<double, 3> minus{{-normal[0], -normal[1], -normal[2]}};
    const auto frame =
        bc::detail::make_face_frame(0.7, 0.7, 0.3, -0.3, g6, g6, normal, minus);
    REQUIRE(frame.has_value());
    CHECK(frame->face_speed == approx(0.3 / 0.7));
    // Orthonormal in gamma, duals lowered, first dual the normal.
    for (size_t a = 0; a < 3; ++a) {
      for (size_t b = 0; b < 3; ++b) {
        double dot = 0.0;
        for (size_t i = 0; i < 3; ++i) {
          dot += frame->duals[a][i] * frame->legs[b][i];
        }
        CHECK(dot == approx(a == b ? 1.0 : 0.0));
      }
      CHECK(frame->duals[0][a] == approx(normal[a]));
    }
    // Swapping the sides mirrors the frame: (-e0, e1, -e2).
    const auto mirror =
        bc::detail::make_face_frame(0.7, 0.7, -0.3, 0.3, g6, g6, minus, normal);
    REQUIRE(mirror.has_value());
    CHECK(mirror->face_speed == approx(-0.3 / 0.7));
    for (size_t i = 0; i < 3; ++i) {
      CHECK(mirror->legs[0][i] == approx(-frame->legs[0][i]));
      CHECK(mirror->legs[1][i] == approx(frame->legs[1][i]));
      CHECK(mirror->legs[2][i] == approx(-frame->legs[2][i]));
    }
    // components / vector / covector are inverse maps.
    const auto x = random_direction(gen);
    const auto back = frame->vector(frame->components(x));
    for (size_t i = 0; i < 3; ++i) {
      CHECK(back[i] == approx(x[i]));
    }
  }
  // Where the frame cannot be built.
  const std::array<double, 6> identity{{1.0, 0.0, 0.0, 1.0, 0.0, 1.0}};
  const std::array<double, 6> singular{{1.0, 1.0, 0.0, 1.0, 0.0, 1.0}};
  const std::array<double, 3> n{{1.0, 0.0, 0.0}};
  const std::array<double, 3> m{{-1.0, 0.0, 0.0}};
  CHECK_FALSE(
      bc::detail::make_face_frame(0.0, 0.0, 0.0, 0.0, identity, identity, n, m)
          .has_value());
  CHECK_FALSE(
      bc::detail::make_face_frame(1.0, 1.0, 0.0, 0.0, singular, singular, n, m)
          .has_value());
  CHECK_FALSE(
      bc::detail::make_face_frame(1.0, 1.0, 0.0, 0.0, identity, identity, n, n)
          .has_value());
}

// ---------------------------------------------------------------------------
// 2. The new shim entry points at w = 0 reproduce the old ones bit for bit
//    (the user's condition), on batches that reach every branch.
// ---------------------------------------------------------------------------
void test_static_entry_points_reproduced(
    const gsl::not_null<std::mt19937*> gen) {
  const ScopedFpeState fpe_off(false);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> sym(-1.0, 1.0);
  // 2500 interfaces so that the shim chunks (capacity 1024); a mix of weak,
  // moderate and strong fields (the B -> 0 branch, HLLD and the seven-wave
  // solve), slow and fast flows (supersonic faces), and large jumps (HLL
  // fallbacks).
  constexpr size_t npts = 2500;
  std::vector<double> vl(8 * npts), vr(8 * npts);
  for (size_t k = 0; k < npts; ++k) {
    const double field = k % 4 == 0 ? 1.0e-4 : (k % 4 == 1 ? 0.3 : 2.0);
    const double speed = k % 3 == 0 ? 0.95 : 0.4;
    for (auto* v : {&vl, &vr}) {
      double* s = v->data() + 8 * k;
      s[0] = 0.05 + unit(*gen);
      std::array<double, 3> u{{sym(*gen), sym(*gen), sym(*gen)}};
      const double norm = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
      const double target = speed * unit(*gen);
      for (size_t i = 0; i < 3; ++i) {
        s[1 + i] = u[i] * target / norm;
        s[4 + i] = field * sym(*gen);
      }
      s[7] = 0.01 + (k % 5 == 0 ? 50.0 : 1.0) * unit(*gen);
    }
    vr[8 * k + 4] = vl[8 * k + 4];  // one normal field, as the adapter hands
  }
  const std::vector<double> zero_speed(npts, 0.0);
  using Static =
      int (*)(int, const double*, const double*, double, double*, double*);
  using Moving = int (*)(int, const double*, const double*, double,
                         const double*, double*, double*);
  const std::array<std::pair<Static, Moving>, 4> solvers{
      {{&pluto_hlld_flux, &pluto_hlld_flux_moving},
       {&pluto_hllc_mb_flux, &pluto_hllc_mb_flux_moving},
       {&pluto_hllc_kb_flux, &pluto_hllc_kb_flux_moving},
       {&pluto_hllx_flux, &pluto_hllx_flux_moving}}};
  for (size_t s = 0; s < solvers.size(); ++s) {
    CAPTURE(s);
    std::vector<double> f_old(8 * npts), f_new(8 * npts), p_old(npts),
        p_new(npts);
    gsl::at(solvers, s)
        .first(static_cast<int>(npts), vl.data(), vr.data(), adiabatic_index,
               f_old.data(), p_old.data());
    gsl::at(solvers, s)
        .second(static_cast<int>(npts), vl.data(), vr.data(), adiabatic_index,
                zero_speed.data(), f_new.data(), p_new.data());
    // memcmp: bit for bit, signed zeros and NaN payloads included.
    CHECK(std::memcmp(f_old.data(), f_new.data(),
                      f_old.size() * sizeof(double)) == 0);
    CHECK(std::memcmp(p_old.data(), p_new.data(),
                      p_old.size() * sizeof(double)) == 0);
  }
}

// ---------------------------------------------------------------------------
// 3. The fan sampled on a moving ray: the flux through the moving face is
//    continuous and piecewise linear in w (Rankine-Hugoniot makes the region
//    fluxes F_k - w U_k agree at w = S_k, so a wrong region test or a wrong
//    state shows as a jump), and beyond the fan it is the upwind physical
//    flux F - w U, with F and U from the special-relativistic formulas.
// ---------------------------------------------------------------------------
std::array<double, 8> sr_conserved(const double* v) {
  const double rho = v[0], p = v[7];
  const double v2 = v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  const double b2 = v[4] * v[4] + v[5] * v[5] + v[6] * v[6];
  const double vb = v[1] * v[4] + v[2] * v[5] + v[3] * v[6];
  const double w2 = 1.0 / (1.0 - v2);
  const double h = 1.0 + adiabatic_index / (adiabatic_index - 1.0) * p / rho;
  const double d = rho * std::sqrt(w2);
  const double wt = rho * h * w2 + b2;
  std::array<double, 8> u{};
  u[0] = d;
  for (size_t i = 0; i < 3; ++i) {
    u[1 + i] = wt * v[1 + i] - vb * v[4 + i];
    u[4 + i] = v[4 + i];
  }
  u[7] = rho * h * w2 - p + 0.5 * (b2 * (1.0 + v2) - vb * vb) - d;
  return u;
}

// PLUTO's flux along x (normal total pressure apart, no normal-field flux).
std::array<double, 9> sr_flux_and_pressure(const double* v) {
  const auto u = sr_conserved(v);
  const double v2 = v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  const double b2 = v[4] * v[4] + v[5] * v[5] + v[6] * v[6];
  const double vb = v[1] * v[4] + v[2] * v[5] + v[3] * v[6];
  std::array<double, 9> f{};
  f[0] = u[0] * v[1];
  for (size_t i = 0; i < 3; ++i) {
    f[1 + i] = u[1 + i] * v[1] - v[4] * (v[4 + i] * (1.0 - v2) + vb * v[1 + i]);
  }
  f[4] = 0.0;
  for (size_t i = 1; i < 3; ++i) {
    f[4 + i] = v[1] * v[4 + i] - v[1 + i] * v[4];
  }
  f[7] = u[1] - u[0] * v[1];
  f[8] = v[7] + 0.5 * (b2 * (1.0 - v2) + vb * vb);
  return f;
}

void test_moving_face_sampling(const gsl::not_null<std::mt19937*> gen) {
  const ScopedFpeState fpe_off(false);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> sym(-1.0, 1.0);
  using Moving = int (*)(int, const double*, const double*, double,
                         const double*, double*, double*);
  const std::array<Moving, 4> solvers{
      {&pluto_hlld_flux_moving, &pluto_hllc_mb_flux_moving,
       &pluto_hllc_kb_flux_moving, &pluto_hllx_flux_moving}};
  constexpr size_t num_rays = 2601;
  constexpr double w_min = -1.3;
  constexpr double step = 2.6 / static_cast<double>(num_rays - 1);
  std::vector<double> rays(num_rays);
  for (size_t j = 0; j < num_rays; ++j) {
    rays[j] = w_min + step * static_cast<double>(j);
  }
  for (size_t trial = 0; trial < 12; ++trial) {
    std::array<double, 8> sl{}, sr{};
    const double field = trial % 3 == 0 ? 1.0e-3 : 1.0;
    for (auto* s : {&sl, &sr}) {
      (*s)[0] = 0.2 + unit(*gen);
      for (size_t i = 0; i < 3; ++i) {
        (*s)[1 + i] = 0.3 * sym(*gen);
        (*s)[4 + i] = field * sym(*gen);
      }
      (*s)[7] = 0.1 + unit(*gen);
    }
    sr[4] = sl[4];
    std::vector<double> vl(8 * num_rays), vr(8 * num_rays);
    for (size_t j = 0; j < num_rays; ++j) {
      std::copy(sl.begin(), sl.end(), vl.begin() + 8 * j);
      std::copy(sr.begin(), sr.end(), vr.begin() + 8 * j);
    }
    const auto ul = sr_conserved(sl.data());
    const auto ur = sr_conserved(sr.data());
    const auto fl = sr_flux_and_pressure(sl.data());
    const auto fr = sr_flux_and_pressure(sr.data());
    double u_scale = 0.0;
    for (size_t nv = 0; nv < 8; ++nv) {
      u_scale = std::max({u_scale, std::abs(ul[nv]), std::abs(ur[nv])});
    }
    for (size_t s = 0; s < solvers.size(); ++s) {
      CAPTURE(trial);
      CAPTURE(s);
      std::vector<double> flux(8 * num_rays), press(num_rays);
      gsl::at(solvers, s)(static_cast<int>(num_rays), vl.data(), vr.data(),
                          adiabatic_index, rays.data(), flux.data(),
                          press.data());
      // Beyond the fan: the upwind physical flux through the moving face.
      for (const size_t j : {size_t{0}, num_rays - 1}) {
        const double w = rays[j];
        const auto& u = j == 0 ? ul : ur;
        const auto& f = j == 0 ? fl : fr;
        for (size_t nv = 0; nv < 8; ++nv) {
          CAPTURE(nv);
          CHECK(flux[8 * j + nv] ==
                approx(f[nv] - w * u[nv]).epsilon(1.0e-11).scale(u_scale));
        }
        CHECK(press[j] == approx(f[8]).epsilon(1.0e-11).scale(u_scale));
      }
      // Continuity: a step of `step` in w moves the flux by at most
      // step |U_sel| (a few times the input states' size; a jump would be
      // O(|w| |Delta U|)), and the total (momentum) flux carries the pressure.
      // One exception, a property of the solver and not of the sampling:
      // hllc_kb.c's star states are not integral-consistent at the contact:
      // the jump there, F*_L - v U*_L - (F*_R - v U*_R), is exactly the
      // residual hllc_kb.c itself computes after its flux (into a `scrh` it
      // then discards), so its moving-face flux may jump once, at the contact.
      size_t jumps = 0;
      for (size_t j = 0; j + 1 < num_rays; ++j) {
        bool jumped = false;
        for (size_t nv = 0; nv < 8; ++nv) {
          const double total_a = flux[8 * j + nv] + (nv == 1 ? press[j] : 0.0);
          const double total_b =
              flux[8 * (j + 1) + nv] + (nv == 1 ? press[j + 1] : 0.0);
          jumped =
              jumped or std::abs(total_b - total_a) > 20.0 * step * u_scale;
        }
        if (jumped) {
          CAPTURE(rays[j]);
          ++jumps;
        }
      }
      CHECK(jumps <= (s == 2 ? 1_st : 0_st));
    }
  }
}

// ---------------------------------------------------------------------------
// 3b. No interface's flux depends on the other interfaces of its batch. PLUTO's
//     MaxSignalSpeed used to return at the first interface whose eigenvalues
//     failed, leaving that interface and every later one with an earlier
//     call's wave speeds (pluto/README.md, modification 6). Physical
//     interfaces solved alone and interleaved with failing ones (|v| >= 1,
//     negative pressure), after a different preceding batch, must agree bit
//     for bit, through every entry point.
// ---------------------------------------------------------------------------
void test_batch_independence(const gsl::not_null<std::mt19937*> gen) {
  const ScopedFpeState fpe_off(false);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> sym(-1.0, 1.0);
  constexpr size_t npts = 300;
  const auto physical = [&](double* s) {
    s[0] = 0.2 + unit(*gen);
    std::array<double, 3> v{{sym(*gen), sym(*gen), sym(*gen)}};
    const double norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    const double target = 0.8 * unit(*gen);
    for (size_t i = 0; i < 3; ++i) {
      s[1 + i] = v[i] * target / norm;
      s[4 + i] = sym(*gen);
    }
    s[7] = 0.05 + unit(*gen);
  };
  std::vector<double> pl(8 * npts), pr(8 * npts), ml(16 * npts), mr(16 * npts),
      speeds(npts), mixed_speeds(2 * npts, 0.0);
  for (size_t k = 0; k < npts; ++k) {
    physical(pl.data() + 8 * k);
    physical(pr.data() + 8 * k);
    pr[8 * k + 4] = pl[8 * k + 4];
    speeds[k] = 0.5 * sym(*gen);
    mixed_speeds[2 * k] = speeds[k];
    std::copy(pl.begin() + 8 * k, pl.begin() + 8 * k + 8, ml.begin() + 16 * k);
    std::copy(pr.begin() + 8 * k, pr.begin() + 8 * k + 8, mr.begin() + 16 * k);
    // The failing partner: superluminal on one side, or a negative pressure.
    physical(ml.data() + 16 * k + 8);
    physical(mr.data() + 16 * k + 8);
    if (k % 2 == 0) {
      ml[16 * k + 9] = 1.1;
    } else {
      mr[16 * k + 15] = -0.3;
    }
    mr[16 * k + 12] = ml[16 * k + 12];
  }
  using Static =
      int (*)(int, const double*, const double*, double, double*, double*);
  using Moving = int (*)(int, const double*, const double*, double,
                         const double*, double*, double*);
  const std::array<std::pair<Static, Moving>, 4> solvers{
      {{&pluto_hlld_flux, &pluto_hlld_flux_moving},
       {&pluto_hllc_mb_flux, &pluto_hllc_mb_flux_moving},
       {&pluto_hllc_kb_flux, &pluto_hllc_kb_flux_moving},
       {&pluto_hllx_flux, &pluto_hllx_flux_moving}}};
  for (size_t s = 0; s < solvers.size(); ++s) {
    CAPTURE(s);
    for (const bool moving : {false, true}) {
      CAPTURE(moving);
      std::vector<double> f_alone(8 * npts), p_alone(npts), f_mixed(16 * npts),
          p_mixed(2 * npts);
      const auto call = [&](const size_t n, const double* l, const double* r,
                            const double* w, double* f, double* p) {
        if (moving) {
          gsl::at(solvers, s)
              .second(static_cast<int>(n), l, r, adiabatic_index, w, f, p);
        } else {
          gsl::at(solvers, s)
              .first(static_cast<int>(n), l, r, adiabatic_index, f, p);
        }
      };
      call(npts, pl.data(), pr.data(), speeds.data(), f_alone.data(),
           p_alone.data());
      // A different batch in between, so that any state left behind differs.
      call(2 * npts, mr.data(), ml.data(), mixed_speeds.data(), f_mixed.data(),
           p_mixed.data());
      call(2 * npts, ml.data(), mr.data(), mixed_speeds.data(), f_mixed.data(),
           p_mixed.data());
      size_t differing = 0;
      for (size_t k = 0; k < npts; ++k) {
        differing +=
            std::memcmp(f_alone.data() + 8 * k, f_mixed.data() + 16 * k,
                        8 * sizeof(double)) != 0 or
            std::memcmp(p_alone.data() + k, p_mixed.data() + 2 * k,
                        sizeof(double)) != 0;
      }
      CHECK(differing == 0);
    }
  }
}

// ---------------------------------------------------------------------------
// Solvers under test.
// ---------------------------------------------------------------------------
const bc::PlutoHlld hlld{1.0e-30, 1.0e-8};
const bc::PlutoHllc hllc_mb{bc::PlutoHllcVariant::MignoneBodo, 1.0e-30, 1.0e-8};
const bc::PlutoHllc hllc_kb{bc::PlutoHllcVariant::KimBalsara, 1.0e-30, 1.0e-8};
const bc::PlutoHllx hllx{1.0e-30, 1.0e-8};

template <typename F>
void for_each_solver(F&& f) {
  f(hlld, std::string{"PlutoHlld"});
  f(hllc_mb, std::string{"PlutoHllc MignoneBodo"});
  f(hllc_kb, std::string{"PlutoHllc KimBalsara"});
  f(hllx, std::string{"PlutoHllx"});
}

// ---------------------------------------------------------------------------
// 4. A uniform state on a curved face (constant non-identity gamma, any lapse
//    and shift, the face inside or outside the light cone): the numerical flux
//    is the physical flux, and the strong-form correction vanishes.
// ---------------------------------------------------------------------------
void test_uniform_state(const gsl::not_null<std::mt19937*> gen,
                        const Eos& eos) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (size_t trial = 0; trial < 40; ++trial) {
    Geometry geometry{};
    geometry.metric = random_metric(gen, 0.35);
    geometry.lapse = 0.3 + unit(*gen);
    const auto direction = random_direction(gen);
    const auto normal = unit_normal(geometry, direction);
    // shift = s n^i, so that w = s / alpha; up to |w| ~ 1.6.
    tnsr::ii<double, 3> g{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        g.get(i, j) = geometry.metric[i][j];
      }
    }
    const auto inv = determinant_and_inverse(g).second;
    const double s = (2.0 * unit(*gen) - 1.0) * 1.6 * geometry.lapse;
    for (size_t i = 0; i < 3; ++i) {
      geometry.shift[i] = 0.0;
      for (size_t j = 0; j < 3; ++j) {
        geometry.shift[i] += s * inv.get(i, j) * normal[j];
      }
    }
    auto prim = random_primitive(gen, geometry.metric, 0.8, 0.8);
    prim.phi = 0.1 * (2.0 * unit(*gen) - 1.0);
    for_each_solver([&](const auto& solver, const std::string& name) {
      CAPTURE(name);
      CAPTURE(s / geometry.lapse);
      const std::array<double, 3> minus{{-normal[0], -normal[1], -normal[2]}};
      const auto l = package(solver, prim, geometry, normal, eos);
      const auto r = package(solver, prim, geometry, minus, eos);
      CHECK(get(l.flatness)[0] > 1.0e-12);
      bc::detail::reset_face_frame_counts();
      const auto weak = boundary(solver, l, r, dg::Formulation::WeakInertial);
      const auto counts = bc::detail::face_frame_counts();
      CHECK(counts.curved_points == 1);
      CHECK(counts.curved_solved == 1);
      // PLUTO's HLLD total-pressure secant (also hllx's HLLD stage) stops at
      // |dp| < 1e-5 p or |f| < 1e-6, so even a uniform state is reproduced
      // only to ~1e-6 (6e-6 seen) when the face lies inside the fan; HLLC has
      // no iteration and is exact to round-off. A mapping error would be O(1).
      const bool iterated = name == "PlutoHlld" or name == "PlutoHllx";
      const double tol = iterated ? 1.0e-4 : 1.0e-11;
      check_close(weak, physical_flux(l), tol, name + " weak");
      const auto strong =
          boundary(solver, l, r, dg::Formulation::StrongInertial);
      CAPTURE(norm_inf(physical_flux(l)));
      const double scale = std::max(1.0, norm_inf(physical_flux(l)));
      CHECK(norm_inf(strong) <= tol * scale);
    });
  }
}

// ---------------------------------------------------------------------------
// 5. A lapse alone (gamma = delta, beta = 0) rescales time: the curved path at
//    lapse alpha is alpha times the flat path, on general two-state faces.
// ---------------------------------------------------------------------------
void test_lapse_rescaling(const gsl::not_null<std::mt19937*> gen,
                          const Eos& eos) {
  for (size_t trial = 0; trial < 40; ++trial) {
    Geometry flat{};
    const auto normal = unit_normal(flat, random_direction(gen));
    const auto left = random_primitive(gen, flat.metric, 0.7, 1.0);
    auto right = random_primitive(gen, flat.metric, 0.7, 1.0);
    right.phi = 0.05;
    Geometry lapsed = flat;
    lapsed.lapse = 0.45;
    for_each_solver([&](const auto& solver, const std::string& name) {
      CAPTURE(name);
      const auto g_flat = solve_face(solver, left, right, flat, normal, eos);
      bc::detail::reset_face_frame_counts();
      const auto g_lapsed =
          solve_face(solver, left, right, lapsed, normal, eos);
      CHECK(bc::detail::face_frame_counts().curved_solved == 1);
      Correction scaled{};
      for (size_t k = 0; k < scaled.size(); ++k) {
        gsl::at(scaled, k) = gsl::at(g_lapsed, k) / lapsed.lapse;
      }
      check_close(scaled, g_flat, 1.0e-10, name + " lapse");
    });
  }
}

// ---------------------------------------------------------------------------
// 6. Minkowski space in linear coordinates x^k = A^k_i (x'^i + beta'^i t),
//    t_inertial = alpha t, with a tangential shift (beta'^i n'_i = 0): the
//    numerical flux maps back to the flat-space flux of the inertial states,
//    G'(D) = alpha |det A| G(D), G'(S_j) = alpha |det A| A^k_j G(S_k),
//    G'(B^j) = alpha |det A| (A^-1)^j_k G(B^k). The HLLD and seven-wave
//    branches are reached here on a curved face.
// ---------------------------------------------------------------------------
void test_minkowski_in_linear_coordinates(
    const gsl::not_null<std::mt19937*> gen, const Eos& eos) {
  std::uniform_real_distribution<double> sym(-1.0, 1.0);
  for (size_t trial = 0; trial < 40; ++trial) {
    Matrix3 a{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        a[i][j] = (i == j ? 1.0 : 0.0) + 0.4 * sym(*gen);
      }
    }
    tnsr::ii<double, 3> a_as_tensor{};  // only for the inverse
    Matrix3 a_inv{};
    {
      const double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                         a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                         a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
      REQUIRE(std::abs(det) > 0.1);
      a_inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
      a_inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
      a_inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
      a_inv[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
      a_inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
      a_inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
      a_inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
      a_inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det;
      a_inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
    }
    (void)a_as_tensor;
    const double abs_det =
        std::abs(a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                 a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                 a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]));
    Geometry inertial{};
    const auto normal = unit_normal(inertial, random_direction(gen));
    const auto left = random_primitive(gen, inertial.metric, 0.6, 1.2);
    const auto right = random_primitive(gen, inertial.metric, 0.6, 1.2);

    // gamma'_ij = A^k_i A^k_j; n'_i = A^k_i n_k; v' = A^-1 v; B' = A^-1 B.
    Geometry primed{};
    primed.lapse = 0.8;
    std::array<double, 3> normal_primed{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        primed.metric[i][j] = 0.0;
        for (size_t k = 0; k < 3; ++k) {
          primed.metric[i][j] += a[k][i] * a[k][j];
        }
      }
      for (size_t k = 0; k < 3; ++k) {
        normal_primed[i] += a[k][i] * normal[k];
      }
    }
    // A tangential shift: beta'^i n'_i = 0.
    const auto raw = random_direction(gen);
    double raw_n = 0.0, nn = 0.0;
    std::array<double, 3> n_up{};  // n'^i = (A^-1 A^-T)^{ij} n'_j = A^-1 n
    for (size_t i = 0; i < 3; ++i) {
      for (size_t k = 0; k < 3; ++k) {
        n_up[i] += a_inv[i][k] * normal[k];
      }
    }
    for (size_t i = 0; i < 3; ++i) {
      raw_n += raw[i] * normal_primed[i];
      nn += n_up[i] * normal_primed[i];
    }
    for (size_t i = 0; i < 3; ++i) {
      primed.shift[i] = 0.3 * (raw[i] - raw_n / nn * n_up[i]);
    }
    const auto map = [&a_inv](const Primitive& p) {
      Primitive q = p;
      for (size_t i = 0; i < 3; ++i) {
        q.velocity[i] = 0.0;
        q.field[i] = 0.0;
        for (size_t k = 0; k < 3; ++k) {
          q.velocity[i] += a_inv[i][k] * p.velocity[k];
          q.field[i] += a_inv[i][k] * p.field[k];
        }
      }
      return q;
    };
    for_each_solver([&](const auto& solver, const std::string& name) {
      CAPTURE(name);
      const auto g_flat =
          solve_face(solver, left, right, inertial, normal, eos);
      const auto g_primed =
          solve_face(solver, map(left), map(right), primed, normal_primed, eos);
      const double scale = primed.lapse * abs_det;
      Correction expected{};
      expected[0] = scale * g_flat[0];
      expected[1] = scale * g_flat[1];
      expected[2] = scale * g_flat[2];
      expected[9] = scale * g_flat[9];
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k < 3; ++k) {
          expected[3 + j] += scale * a[k][j] * g_flat[3 + k];
          expected[6 + j] += scale * a_inv[j][k] * g_flat[6 + k];
        }
      }
      // hllc_kb.c solves for its transverse star velocities by Newton
      // iteration stopped at |dv| <= 1e-7, so rotating the transverse pair
      // (the face frame's differs from the flat triad) moves its flux by up
      // to that much; every other solver agrees to round-off.
      check_close(g_primed, expected,
                  name == "PlutoHllc KimBalsara" ? 1.0e-6 : 1.0e-9,
                  name + " Minkowski");
    });
  }
}

// ---------------------------------------------------------------------------
// 7. A pure shift (alpha = 1, gamma = delta): tangential, it changes nothing;
//    normal, it translates the fan (continuity of the boundary correction in
//    beta^n, the upwind physical flux once the face leaves the fan). The same
//    face on a moving flat mesh (v_g = beta) gives the same correction (Q4).
// ---------------------------------------------------------------------------
void test_shift_and_mesh_velocity(const gsl::not_null<std::mt19937*> gen,
                                  const Eos& eos) {
  for (size_t trial = 0; trial < 10; ++trial) {
    Geometry flat{};
    const auto normal = unit_normal(flat, random_direction(gen));
    const auto left = random_primitive(gen, flat.metric, 0.6, 0.8);
    const auto right = random_primitive(gen, flat.metric, 0.6, 0.8);
    // A tangential direction.
    auto t = random_direction(gen);
    const double tn = t[0] * normal[0] + t[1] * normal[1] + t[2] * normal[2];
    for (size_t i = 0; i < 3; ++i) {
      t[i] -= tn * normal[i];
    }
    for_each_solver([&](const auto& solver, const std::string& name) {
      CAPTURE(name);
      const auto g_flat = solve_face(solver, left, right, flat, normal, eos);
      Geometry tangential = flat;
      for (size_t i = 0; i < 3; ++i) {
        tangential.shift[i] = 0.4 * t[i];
      }
      check_close(solve_face(solver, left, right, tangential, normal, eos),
                  g_flat, 1.0e-10, name + " tangential shift");

      // Normal shift beta = s n: the correction is continuous in s.
      Correction previous{};
      constexpr size_t num_shifts = 801;
      for (size_t j = 0; j < num_shifts; ++j) {
        const double s =
            -1.4 + 2.8 * static_cast<double>(j) / (num_shifts - 1.0);
        Geometry shifted = flat;
        for (size_t i = 0; i < 3; ++i) {
          shifted.shift[i] = s * normal[i];
        }
        const auto g = solve_face(solver, left, right, shifted, normal, eos);
        // Where the face crosses the light cone (|w| = 1) the common normal
        // field handed to the MHD solver switches from the GLM middle state
        // to the upwind side's own: the GLM/MHD splitting is not Rankine-
        // Hugoniot consistent across the GLM wave, so the correction jumps
        // there. Upwind consistency beyond the cone is checked below.
        const double previous_s =
            -1.4 + 2.8 * (static_cast<double>(j) - 1.0) / (num_shifts - 1.0);
        const bool crosses_cone =
            j > 0 and (std::abs(s) - 1.0) * (std::abs(previous_s) - 1.0) <= 0.0;
        if (j > 0 and not crosses_cone) {
          CAPTURE(s);
          double jump = 0.0;
          for (size_t k = 0; k < g.size(); ++k) {
            jump =
                std::max(jump, std::abs(gsl::at(g, k) - gsl::at(previous, k)));
          }
          CHECK(jump <=
                50.0 * 2.8 / (num_shifts - 1.0) * std::max(1.0, norm_inf(g)));
        }
        previous = g;
        // Beyond the light cone the upwind side's physical flux.
        if (std::abs(s) > 1.05) {
          const std::array<double, 3> minus{
              {-normal[0], -normal[1], -normal[2]}};
          const auto l = package(solver, left, shifted, normal, eos);
          const auto r = package(solver, right, shifted, minus, eos);
          // The face moves with w = s: for s < -1 it is left of the fan
          // (the interior's flux), for s > 1 right of it (the exterior's,
          // whose packaged flux is along -n).
          Correction expected = physical_flux(s < 0.0 ? l : r);
          if (s > 0.0) {
            for (double& x : expected) {
              x = -x;
            }
          }
          check_close(g, expected, 1.0e-10, name + " beyond the cone");
        }
        // The same face on a moving flat mesh, v_g = beta.
        if (j % 100 == 0) {
          Geometry moving = flat;
          moving.mesh_velocity = shifted.shift;
          bc::detail::reset_face_frame_counts();
          const auto g_mesh =
              solve_face(solver, left, right, moving, normal, eos);
          if (s != 0.0) {
            CHECK(bc::detail::face_frame_counts().curved_points == 1);
          }
          check_close(g_mesh, g, 1.0e-12, name + " moving mesh");
        }
      }
    });
  }
}

// ---------------------------------------------------------------------------
// 8. Conservation on Kerr-Schild faces (Schwarzschild, M = 1): outside the
//    horizon (r = 6, w ~ 0.3 on a radial face) and inside it (r = 1.5, w > 1),
//    G(int, ext) = -G(ext, int) in the weak form.
// ---------------------------------------------------------------------------
void test_kerr_schild_conservation(const gsl::not_null<std::mt19937*> gen,
                                   const Eos& eos) {
  for (const double radius : {6.0, 1.5}) {
    for (size_t trial = 0; trial < 10; ++trial) {
      auto x = random_direction(gen);
      const double norm = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      for (double& xi : x) {
        xi *= radius / norm;
      }
      const double h = 2.0 / radius;  // 2M/r
      Geometry geometry{};
      geometry.lapse = 1.0 / std::sqrt(1.0 + h);
      for (size_t i = 0; i < 3; ++i) {
        geometry.shift[i] = h / (1.0 + h) * x[i] / radius;
        for (size_t j = 0; j < 3; ++j) {
          geometry.metric[i][j] =
              (i == j ? 1.0 : 0.0) + h * x[i] * x[j] / (radius * radius);
        }
      }
      // Radial faces and random ones.
      const auto direction = trial % 2 == 0 ? x : random_direction(gen);
      const auto normal = unit_normal(geometry, direction);
      const std::array<double, 3> minus{{-normal[0], -normal[1], -normal[2]}};
      const auto left = random_primitive(gen, geometry.metric, 0.7, 0.6);
      const auto right = random_primitive(gen, geometry.metric, 0.7, 0.6);
      for_each_solver([&](const auto& solver, const std::string& name) {
        CAPTURE(name);
        CAPTURE(radius);
        const auto l = package(solver, left, geometry, normal, eos);
        const auto r = package(solver, right, geometry, minus, eos);
        bc::detail::reset_face_frame_counts();
        const auto g_int =
            boundary(solver, l, r, dg::Formulation::WeakInertial);
        const auto counts = bc::detail::face_frame_counts();
        CHECK(counts.curved_points == 1);
        CHECK(counts.fallback_frame == 0);
        if (trial % 2 == 0 and radius < 2.0) {
          CHECK(counts.outside_light_cone == counts.curved_solved);
        }
        const auto g_ext =
            boundary(solver, r, l, dg::Formulation::WeakInertial);
        Correction minus_ext{};
        for (size_t k = 0; k < minus_ext.size(); ++k) {
          gsl::at(minus_ext, k) = -gsl::at(g_ext, k);
        }
        check_close(g_int, minus_ext, 1.0e-10, name + " conservation");
      });
    }
  }
}

// ---------------------------------------------------------------------------
// 9. D2-R on a curved face whose face does not move (a lapse alone): solved
//    by its own kernel at ray 0 in the face frame, alpha times the flat flux.
// ---------------------------------------------------------------------------
void test_d2r_lapse_only(const Eos& eos) {
  const ScopedFpeState fpe_off(false);
  const bc::PlutoD2R d2r{1.0e-30, 1.0e-12};
  Primitive left{};
  left.rho = 1.0;
  left.pressure = 1.0;
  left.field = {{0.5, 1.0, 0.0}};
  Primitive right{};
  right.rho = 0.125;
  right.pressure = 0.1;
  right.field = {{0.5, -1.0, 0.0}};
  Geometry flat{};
  const std::array<double, 3> normal{{1.0, 0.0, 0.0}};
  const auto g_flat = solve_face(d2r, left, right, flat, normal, eos);
  Geometry lapsed = flat;
  lapsed.lapse = 0.6;
  bc::detail::reset_face_frame_counts();
  const auto g_lapsed = solve_face(d2r, left, right, lapsed, normal, eos);
  CHECK(bc::detail::face_frame_counts().curved_solved == 1);
  CHECK(bc::detail::face_frame_counts().fallback_ray == 0);
  Correction scaled{};
  for (size_t k = 0; k < scaled.size(); ++k) {
    gsl::at(scaled, k) = gsl::at(g_lapsed, k) / lapsed.lapse;
  }
  check_close(scaled, g_flat, 1.0e-9, "D2-R lapse");
}
}  // namespace

// [[TimeOut, 60]]
SPECTRE_TEST_CASE(
    "Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.PlutoFaceFrame",
    "[Unit][GrMhd]") {
  MAKE_GENERATOR(gen);
  const auto eos =
      EquationsOfState::IdealFluid<true>{adiabatic_index}.promote_to_3d_eos();
  test_frame(make_not_null(&gen));
  test_static_entry_points_reproduced(make_not_null(&gen));
  test_moving_face_sampling(make_not_null(&gen));
  test_batch_independence(make_not_null(&gen));
  test_uniform_state(make_not_null(&gen), *eos);
  test_lapse_rescaling(make_not_null(&gen), *eos);
  test_minkowski_in_linear_coordinates(make_not_null(&gen), *eos);
  test_shift_and_mesh_velocity(make_not_null(&gen), *eos);
  test_kerr_schild_conservation(make_not_null(&gen), *eos);
  test_d2r_lapse_only(*eos);
}
