// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/PerturbedTovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated3d.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::AnalyticData {
namespace {

using TovCoordinates = RelativisticEuler::Solutions::TovCoordinates;
using TovStar = RelativisticEuler::Solutions::TovStar;

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<evolution::initial_data::InitialData,
                             tmpl::list<PerturbedTovStar>>>;
  };
};

static_assert(not is_analytic_solution_v<PerturbedTovStar>,
              "PerturbedTovStar should be analytic data");
static_assert(is_analytic_data_v<PerturbedTovStar>,
              "PerturbedTovStar should be analytic data");

// A tiny npe-gas table made by GenerateNpeGasEosTable (24 x 2 x 30 nodes,
// source tarball stripped). It supplies a beta-equilibrium Y_e(rho) with a
// real gradient, which is what the composition displacement needs.
std::string npe_table_path() {
  return unit_test_src_path() +
         "PointwiseFunctions/AnalyticData/GrMhd/npe_unit_test.h5";
}

std::unique_ptr<EquationsOfState::EquationOfState<true, 1>> make_polytrope() {
  return std::make_unique<EquationsOfState::PolytropicFluid<true>>(100.0, 2.0);
}

std::unique_ptr<EquationsOfState::EquationOfState<true, 3>> make_yeq() {
  return std::make_unique<EquationsOfState::Tabulated3D<true>>(
      npe_table_path(), "npe_unit_test.eos", 1);
}

constexpr double rho_c = 1.28e-3;
constexpr double omega = 5.0e-3;

// An analytic stand-in for a mode: xi_r ~ r near the centre, vanishing at the
// surface; xi_perp half of it. Tabulated on [0, R].
// An analytic stand-in for a mode: xi_r ~ r near the centre, vanishing at the
// surface; xi_perp half of it. Tabulated on [0, R].
constexpr size_t profile_points = 201;

double analytic_xi_r(const double r, const double outer_radius) {
  return 0.02 * r * (1.0 - r / outer_radius);
}

ModeProfile make_profile(const double outer_radius) {
  const size_t n = profile_points;
  std::vector<double> r(n);
  std::vector<double> xi_r(n);
  std::vector<double> xi_perp(n);
  for (size_t i = 0; i < n; ++i) {
    r[i] = outer_radius * static_cast<double>(i) / static_cast<double>(n - 1);
    xi_r[i] = analytic_xi_r(r[i], outer_radius);
    xi_perp[i] = 0.5 * xi_r[i];
  }
  return ModeProfile{2, omega, std::move(r), std::move(xi_r),
                     std::move(xi_perp)};
}

std::string write_profile_file(const ModeProfile& mode,
                               const std::string& filename) {
  // Write the same analytic values make_profile tabulated, not spline samples:
  // GSL evaluates the last knot through the previous interval's cubic, so a
  // sampled file would differ from the in-memory profile at round-off and the
  // exact operator== below would fail.
  std::ofstream file(filename);
  file.precision(17);
  file << "# test profile\n";
  const size_t n = profile_points;
  const double R = mode.outer_radius();
  file << mode.l() << " " << mode.omega() << " " << n << "\n";
  for (size_t i = 0; i < n; ++i) {
    const double r = R * static_cast<double>(i) / static_cast<double>(n - 1);
    const double xr = analytic_xi_r(r, R);
    file << r << " " << xr << " " << 0.5 * xr << "\n";
  }
  return filename;
}

void test_mode_profile() {
  ModeProfile mode = make_profile(10.0);
  CHECK(mode.l() == 2);
  CHECK(mode.omega() == omega);
  CHECK(mode.outer_radius() == 10.0);
  CHECK(mode.xi_r(0.0) == 0.0);
  CHECK(mode.xi_r(5.0) == approx(0.02 * 5.0 * 0.5));
  CHECK(mode.xi_perp(5.0) == approx(0.5 * 0.02 * 5.0 * 0.5));
  // zero outside the tabulated range, never an out-of-range spline lookup
  CHECK(mode.xi_r(-1.0) == 0.0);
  CHECK(mode.xi_r(10.5) == 0.0);
  CHECK(mode.xi_perp(10.5) == 0.0);
  CHECK(serialize_and_deserialize(mode) == mode);
  CHECK(mode != make_profile(11.0));

  const std::string filename = "Unit.PerturbedTovStar.profile.txt";
  write_profile_file(mode, filename);
  const ModeProfile from_file{filename};
  CHECK(from_file.l() == 2);
  CHECK(from_file.omega() == omega);
  CHECK(from_file.outer_radius() == 10.0);
  CHECK(from_file.xi_r(3.3) == approx(mode.xi_r(3.3)).epsilon(1.0e-10));
  CHECK(from_file == mode);
  file_system::rm(filename, false);

  CHECK_THROWS_WITH(
      (ModeProfile{3,
                   omega,
                   {0.0, 1.0, 2.0, 3.0},
                   {0.0, 1.0, 1.0, 0.0},
                   {0.0, 1.0, 1.0, 0.0}}),
      Catch::Matchers::ContainsSubstring("only l = 2 is implemented"));
  CHECK_THROWS_WITH((ModeProfile{2,
                                 omega,
                                 {0.1, 1.0, 2.0, 3.0},
                                 {0.0, 1.0, 1.0, 0.0},
                                 {0.0, 1.0, 1.0, 0.0}}),
                    Catch::Matchers::ContainsSubstring("must start at r = 0"));
  CHECK_THROWS_WITH((ModeProfile{2,
                                 omega,
                                 {0.0, 2.0, 1.0, 3.0},
                                 {0.0, 1.0, 1.0, 0.0},
                                 {0.0, 1.0, 1.0, 0.0}}),
                    Catch::Matchers::ContainsSubstring("increase strictly"));
  CHECK_THROWS_WITH((ModeProfile{"Unit.PerturbedTovStar.does_not_exist.txt"}),
                    Catch::Matchers::ContainsSubstring("cannot open"));
}

// P_2 pattern of the displacement at a point x for unit polar axis n
std::array<double, 3> expected_displacement(const ModeProfile& mode,
                                            const std::array<double, 3>& x,
                                            const std::array<double, 3>& n) {
  const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
  double mu = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    mu += gsl::at(x, i) / r * gsl::at(n, i);
  }
  const double p2 = 0.5 * (3.0 * mu * mu - 1.0);
  const double xr = mode.xi_r(r);
  const double xp = mode.xi_perp(r);
  std::array<double, 3> xi{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(xi, i) = (xr * p2 - 3.0 * mu * mu * xp) * gsl::at(x, i) / r +
                     3.0 * mu * xp * gsl::at(n, i);
  }
  return xi;
}

template <typename Tag>
typename Tag::type tov_var(const TovStar& tov,
                           const tnsr::I<DataVector, 3>& x) {
  return get<Tag>(tov.variables(x, 0.0, tmpl::list<Tag>{}));
}

template <typename Tag>
typename Tag::type pert_var(const PerturbedTovStar& star,
                            const tnsr::I<DataVector, 3>& x) {
  return get<Tag>(star.variables(x, tmpl::list<Tag>{}));
}

void test_physics(const TovCoordinates coord_system) {
  const TovStar tov{rho_c, make_polytrope(), coord_system, make_yeq()};
  const double R = tov.radial_solution().outer_radius();
  const ModeProfile mode = make_profile(R);
  const std::array<double, 3> axis{{0.0, 1.0, 0.0}};  // CartoonCylinder's axis

  // Points: on the axis, on the equator, at mu = 1/sqrt(3) (P_2 = 0), and one
  // outside the star. Radii well inside so every source radius stays inside.
  const double r1 = 0.3 * R;
  const double r2 = 0.6 * R;
  const double s3 = 1.0 / std::sqrt(3.0);
  const tnsr::I<DataVector, 3> x{{DataVector{0.0, r2, r1 * s3, 0.0},
                                  DataVector{r1, 0.0, r1 * s3, 1.5 * R},
                                  DataVector{0.0, 0.0, r1 * s3, 0.0}}};
  const tnsr::I<DataVector, 3> x_interior{{DataVector{0.0, r2, r1 * s3},
                                           DataVector{r1, 0.0, r1 * s3},
                                           DataVector{0.0, 0.0, r1 * s3}}};

  // --- zero amplitudes: identical to TovStar everywhere --------------------
  {
    const PerturbedTovStar star{
        rho_c, make_polytrope(), coord_system, make_yeq(), mode, 0.0, 0.0,
        axis};
    CHECK_ITERABLE_APPROX(
        (pert_var<hydro::Tags::ElectronFraction<DataVector>>(star, x)),
        (tov_var<hydro::Tags::ElectronFraction<DataVector>>(tov, x)));
    CHECK_ITERABLE_APPROX(
        (pert_var<hydro::Tags::RestMassDensity<DataVector>>(star, x)),
        (tov_var<hydro::Tags::RestMassDensity<DataVector>>(tov, x)));
    CHECK_ITERABLE_APPROX(
        (pert_var<hydro::Tags::Pressure<DataVector>>(star, x)),
        (tov_var<hydro::Tags::Pressure<DataVector>>(tov, x)));
    CHECK_ITERABLE_APPROX(
        (pert_var<hydro::Tags::SpatialVelocity<DataVector, 3>>(star, x)),
        (tov_var<hydro::Tags::SpatialVelocity<DataVector, 3>>(tov, x)));
    CHECK_ITERABLE_APPROX(
        (pert_var<hydro::Tags::LorentzFactor<DataVector>>(star, x)),
        (tov_var<hydro::Tags::LorentzFactor<DataVector>>(tov, x)));
    CHECK_ITERABLE_APPROX((pert_var<gr::Tags::Lapse<DataVector>>(star, x)),
                          (tov_var<gr::Tags::Lapse<DataVector>>(tov, x)));
    CHECK_ITERABLE_APPROX(
        (pert_var<gr::Tags::SpatialMetric<DataVector, 3>>(star, x)),
        (tov_var<gr::Tags::SpatialMetric<DataVector, 3>>(tov, x)));
  }

  // --- composition displacement only ---------------------------------------
  {
    const double amplitude = 1.5;
    const PerturbedTovStar star{
        rho_c, make_polytrope(), coord_system, make_yeq(),
        mode,  amplitude,        0.0,          axis};
    // Y_e(x) must be TovStar's Y_e at the source radius r - A xi_r P_2(mu).
    const auto ye =
        pert_var<hydro::Tags::ElectronFraction<DataVector>>(star, x);
    for (size_t p = 0; p < 4; ++p) {
      const std::array<double, 3> xp{
          {get<0>(x)[p], get<1>(x)[p], get<2>(x)[p]}};
      const double r = std::sqrt(xp[0] * xp[0] + xp[1] * xp[1] + xp[2] * xp[2]);
      double mu = 0.0;
      for (size_t i = 0; i < 3; ++i) {
        mu += gsl::at(xp, i) / r * gsl::at(axis, i);
      }
      const double p2 = 0.5 * (3.0 * mu * mu - 1.0);
      const double r_source = r - amplitude * mode.xi_r(r) * p2;
      tnsr::I<DataVector, 3> x_source{};
      for (size_t i = 0; i < 3; ++i) {
        x_source.get(i) = DataVector{gsl::at(xp, i) * r_source / r};
      }
      const double expected = get(
          tov_var<hydro::Tags::ElectronFraction<DataVector>>(tov, x_source))[0];
      CHECK(get(ye)[p] == approx(expected));
      if (p < 2) {
        // the displacement really moved something (point 2 sits at P_2 = 0,
        // where zero radial displacement is the correct answer)
        CHECK(std::abs(r_source - r) > 1.0e-4 * R);
        CHECK(
            get(ye)[p] !=
            get(tov_var<hydro::Tags::ElectronFraction<DataVector>>(tov, x))[p]);
      }
    }
    // everything else stays on the background
    CHECK_ITERABLE_APPROX(
        (pert_var<hydro::Tags::RestMassDensity<DataVector>>(star, x)),
        (tov_var<hydro::Tags::RestMassDensity<DataVector>>(tov, x)));
    CHECK_ITERABLE_APPROX(
        (pert_var<hydro::Tags::Pressure<DataVector>>(star, x)),
        (tov_var<hydro::Tags::Pressure<DataVector>>(tov, x)));
    CHECK_ITERABLE_APPROX(
        (pert_var<hydro::Tags::SpatialVelocity<DataVector, 3>>(star, x)),
        (tov_var<hydro::Tags::SpatialVelocity<DataVector, 3>>(tov, x)));
    CHECK_ITERABLE_APPROX(
        (pert_var<hydro::Tags::LorentzFactor<DataVector>>(star, x)),
        (tov_var<hydro::Tags::LorentzFactor<DataVector>>(tov, x)));
  }

  // --- velocity kick only ---------------------------------------------------
  {
    const double amplitude = 2.0;
    const PerturbedTovStar star{
        rho_c, make_polytrope(), coord_system, make_yeq(), mode,
        0.0,   amplitude,        axis};
    const auto v =
        pert_var<hydro::Tags::SpatialVelocity<DataVector, 3>>(star, x);
    for (size_t p = 0; p < 4; ++p) {
      const std::array<double, 3> xp{
          {get<0>(x)[p], get<1>(x)[p], get<2>(x)[p]}};
      const auto xi = expected_displacement(mode, xp, axis);
      for (size_t i = 0; i < 3; ++i) {
        CHECK(v.get(i)[p] == approx(amplitude * omega * gsl::at(xi, i)));
      }
    }
    // the exterior point is untouched
    for (size_t i = 0; i < 3; ++i) {
      CHECK(v.get(i)[3] == 0.0);
    }
    // on the axis the kick is purely radial; on the equator it is -1/2 xi_r
    // radial; at P_2 = 0 it comes from xi_perp alone
    CHECK(v.get(0)[0] == approx(0.0));
    CHECK(v.get(2)[0] == approx(0.0));
    CHECK(v.get(1)[0] == approx(amplitude * omega * mode.xi_r(r1)));
    CHECK(v.get(0)[1] == approx(-0.5 * amplitude * omega * mode.xi_r(r2)));
    CHECK(v.get(1)[1] == approx(0.0));
    // Lorentz factor from the TOV spatial metric
    const auto W = pert_var<hydro::Tags::LorentzFactor<DataVector>>(star, x);
    const auto gamma = tov_var<gr::Tags::SpatialMetric<DataVector, 3>>(tov, x);
    const auto v_sq = dot_product(v, v, gamma);
    for (size_t p = 0; p < 4; ++p) {
      CHECK(get(W)[p] == approx(1.0 / std::sqrt(1.0 - get(v_sq)[p])));
    }
    CHECK(get(W)[0] > 1.0);
    CHECK(get(W)[3] == 1.0);
    // composition stays on the background
    CHECK_ITERABLE_APPROX(
        (pert_var<hydro::Tags::ElectronFraction<DataVector>>(star, x)),
        (tov_var<hydro::Tags::ElectronFraction<DataVector>>(tov, x)));
    // the all-interior DataVector code path agrees with the mixed one
    const auto v_int =
        pert_var<hydro::Tags::SpatialVelocity<DataVector, 3>>(star, x_interior);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t p = 0; p < 3; ++p) {
        CHECK(v_int.get(i)[p] == approx(v.get(i)[p]));
      }
    }
  }

  // --- a profile in the wrong coordinates is refused -----------------------
  {
    const TovStar other{rho_c, make_polytrope(),
                        coord_system == TovCoordinates::Isotropic
                            ? TovCoordinates::Schwarzschild
                            : TovCoordinates::Isotropic,
                        make_yeq()};
    const ModeProfile wrong =
        make_profile(other.radial_solution().outer_radius());
    CHECK_THROWS_WITH(
        (PerturbedTovStar{rho_c, make_polytrope(), coord_system, make_yeq(),
                          wrong, 1.0, 0.0, axis}),
        Catch::Matchers::ContainsSubstring("outer radius in the selected"));
  }
  CHECK_THROWS_WITH(
      (PerturbedTovStar{rho_c, make_polytrope(), coord_system, make_yeq(), mode,
                        1.0, 0.0, std::array<double, 3>{{0.0, 0.0, 0.0}}}),
      Catch::Matchers::ContainsSubstring("nonzero vector"));
}

void test_options_and_serialization(const TovCoordinates coord_system) {
  register_factory_classes_with_charm<Metavariables>();
  register_classes_with_charm<PerturbedTovStar>();
  register_classes_with_charm<EquationsOfState::PolytropicFluid<true>>();
  register_classes_with_charm<EquationsOfState::Tabulated3D<true>>();

  const TovStar tov{rho_c, make_polytrope(), coord_system, make_yeq()};
  const ModeProfile mode = make_profile(tov.radial_solution().outer_radius());
  const std::string filename =
      "Unit.PerturbedTovStar." + get_output(coord_system) + ".txt";
  write_profile_file(mode, filename);

  const std::unique_ptr<evolution::initial_data::InitialData> option_data =
      TestHelpers::test_option_tag<
          evolution::initial_data::OptionTags::InitialData, Metavariables>(
          "PerturbedTovStar:\n"
          "  CentralDensity: 1.28e-3\n"
          "  EquationOfState:\n"
          "    PolytropicFluid:\n"
          "      PolytropicConstant: 100.0\n"
          "      PolytropicExponent: 2.0\n"
          "  Coordinates: " +
          get_output(coord_system) +
          "\n"
          "  Yeq:\n"
          "    Tabulated3D:\n"
          "      TableFilename: " +
          npe_table_path() +
          "\n"
          "      TableSubFilename: npe_unit_test.eos\n"
          "      InterpolationOrder: 1\n"
          "  ModeProfileFile: " +
          filename +
          "\n"
          "  DisplacementAmplitude: 1.0\n"
          "  VelocityAmplitude: 0.0\n"
          "  PolarAxis: [0.0, 2.0, 0.0]\n")
          ->get_clone();
  const auto deserialized = serialize_and_deserialize(option_data);
  const auto& star = dynamic_cast<const PerturbedTovStar&>(*deserialized);
  CHECK(star.mode() == mode);
  CHECK(star.displacement_amplitude() == 1.0);
  CHECK(star.velocity_amplitude() == 0.0);
  // the axis was normalised
  CHECK(star.polar_axis() == std::array<double, 3>{{0.0, 1.0, 0.0}});

  const PerturbedTovStar direct{
      rho_c, make_polytrope(), coord_system, make_yeq(), mode, 1.0,
      0.0,   {{0.0, 1.0, 0.0}}};
  CHECK(star == direct);
  CHECK(star != PerturbedTovStar{rho_c,
                                 make_polytrope(),
                                 coord_system,
                                 make_yeq(),
                                 mode,
                                 2.0,
                                 0.0,
                                 {{0.0, 1.0, 0.0}}});
  CHECK(star != PerturbedTovStar{rho_c,
                                 make_polytrope(),
                                 coord_system,
                                 make_yeq(),
                                 mode,
                                 1.0,
                                 0.5,
                                 {{0.0, 1.0, 0.0}}});
  CHECK(star != PerturbedTovStar{rho_c,
                                 make_polytrope(),
                                 coord_system,
                                 make_yeq(),
                                 mode,
                                 1.0,
                                 0.0,
                                 {{0.0, 0.0, 1.0}}});
  test_copy_semantics(direct);
  auto copy = direct;
  test_move_semantics(std::move(copy), direct);

  file_system::rm(filename, false);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.GrMhd.PerturbedTovStar",
                  "[Unit][PointwiseFunctions]") {
  test_mode_profile();
  for (const auto coord_system :
       {TovCoordinates::Schwarzschild, TovCoordinates::Isotropic}) {
    test_physics(coord_system);
    test_options_and_serialization(coord_system);
  }
}
}  // namespace grmhd::AnalyticData
