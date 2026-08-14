// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <algorithm>
#include <array>
#include <boost/program_options.hpp>
#include <cmath>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "IO/ComposeTableDerivatives.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/EosTable.hpp"
#include "IO/H5/File.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {

// Parse an ASCII CompOSE eos.beta file. The CompOSE format is four
// whitespace-separated columns (verified against DD2 HS by cross-check
// against the 3D eos.h5's "pressure" and "specific internal energy"
// datasets at matching (n_b, T_min, Y_e_beta) points):
//   1. baryon number density n_b               [fm^-3]
//   2. electron fraction Y_e_beta              [dimensionless]
//   3. total energy density e = n_b m_n (1+eps)  [MeV/fm^3]
//   4. pressure p                              [MeV/fm^3]
// Column 3 is converted at read time to the SpECTRE dimensionless
// specific-internal-energy convention via
//   epsilon = col3 / (n_b * m_n_MeV) - 1
// so that the h5 "specific internal energy" dataset matches what
// Tabulated3D and Tabulated1D expect (E/(N m_n) - 1, dimensionless).
// epsilon can be negative (bound cold nuclear matter near saturation).
// The n_b grid is expected to be log-spaced.
struct BetaSlice {
  std::vector<double> n_b;
  std::vector<double> electron_fraction;
  std::vector<double> pressure;
  std::vector<double> specific_internal_energy;
};

BetaSlice read_compose_beta(const std::string& path) {
  std::ifstream in(path);
  if (not in.is_open()) {
    ERROR("Could not open CompOSE eos.beta file: " << path);
  }
  BetaSlice slice;
  double nb = 0.0;
  double ye = 0.0;
  double energy_density = 0.0;
  double p = 0.0;
  while (in >> nb >> ye >> energy_density >> p) {
    slice.n_b.push_back(nb);
    slice.electron_fraction.push_back(ye);
    slice.pressure.push_back(p);
    // Convert CompOSE total energy density e [MeV/fm^3] to the SpECTRE
    // dimensionless specific internal energy eps = e/(n_b m_n) - 1.
    const double eps =
        energy_density / (nb * hydro::units::nuclear::neutron_mass) - 1.0;
    slice.specific_internal_energy.push_back(eps);
  }
  if (slice.n_b.size() < 2) {
    ERROR("CompOSE eos.beta at " << path << " yielded fewer than 2 rows.");
  }
  return slice;
}

// Verify n_b is uniformly log-spaced; return (log_min, log_max).
std::pair<double, double> check_log_spacing(const std::vector<double>& n_b) {
  const size_t n = n_b.size();
  if (n_b.front() <= 0.0) {
    ERROR("First n_b value must be positive for log spacing check; got "
          << n_b.front());
  }
  const double log_lo = std::log(n_b.front());
  const double log_hi = std::log(n_b.back());
  const double dlog_expected = (log_hi - log_lo) / static_cast<double>(n - 1);
  double max_abs_dev = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double log_expected = log_lo + dlog_expected * static_cast<double>(i);
    max_abs_dev =
        std::max(max_abs_dev, std::abs(std::log(n_b[i]) - log_expected));
  }
  const double tol = 1.0e-6;
  if (max_abs_dev > tol) {
    ERROR(
        "CompOSE eos.beta n_b grid is not uniformly log-spaced. Maximum "
        "absolute deviation in log(n_b) from a uniform grid: "
        << max_abs_dev << " (tolerance " << tol << ").");
  }
  return {log_lo, log_hi};
}

std::vector<double> compute_adiabatic_index(
    const std::vector<double>& n_b, const std::vector<double>& pressure) {
  const size_t n = n_b.size();
  std::vector<double> log_n_b(n);
  for (size_t i = 0; i < n; ++i) {
    log_n_b[i] = std::log(n_b[i]);
  }
  std::vector<double> slope(n);
  for (size_t i = 0; i < n; ++i) {
    const bool at_lo = (i == 0);
    const bool at_hi = (i + 1 == n);
    const double p_minus = at_lo ? 0.0 : pressure[i - 1];
    const double p_plus = at_hi ? 0.0 : pressure[i + 1];
    const double log_x_minus = at_lo ? 0.0 : log_n_b[i - 1];
    const double log_x_center = log_n_b[i];
    const double log_x_plus = at_hi ? 0.0 : log_n_b[i + 1];
    slope[i] = io::log_log_derivative(p_minus, pressure[i], p_plus, log_x_minus,
                                      log_x_center, log_x_plus, at_lo, at_hi);
  }
  return slope;
}

// Compute h_geom = 1 + eps + p_geom / rho_geom at each grid point and check
// that both pressure and h are strictly monotonic in n_b. Runs BEFORE writing
// the h5 so the user gets a clean error at conversion time rather than at
// runtime EOS load time.
void verify_monotonicity(const std::vector<double>& n_b,
                         const std::vector<double>& pressure,
                         const std::vector<double>& eps) {
  constexpr double nb_fm3_to_geom = hydro::units::nuclear::neutron_mass /
                                    hydro::units::nuclear::pressure_unit;
  constexpr double press_MeV_to_geom =
      1.0 / hydro::units::nuclear::pressure_unit;
  const size_t n = n_b.size();
  double h_prev = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double rho_geom = nb_fm3_to_geom * n_b[i];
    const double p_geom = press_MeV_to_geom * pressure[i];
    const double h = 1.0 + eps[i] + p_geom / rho_geom;
    if (i > 0) {
      if (not(pressure[i] > pressure[i - 1])) {
        ERROR(
            "CompOSE eos.beta pressure not strictly monotonic in n_b at "
            "row "
            << i << ": p[" << i - 1 << "] = " << pressure[i - 1] << ", p[" << i
            << "] = " << pressure[i] << ".");
      }
      if (not(h > h_prev)) {
        ERROR(
            "Computed specific enthalpy h = 1 + eps + p/rho not strictly "
            "monotonic in n_b at row "
            << i << ": h[" << i - 1 << "] = " << h_prev << ", h[" << i
            << "] = " << h << " at n_b = " << n_b[i] << " fm^-3.");
      }
    }
    h_prev = h;
  }
}

DataVector to_datavector(const std::vector<double>& v) {
  DataVector out(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    out[i] = v[i];
  }
  return out;
}

void convert_file(const std::string& compose_beta_path,
                  const std::string& spectre_eos_filename,
                  const std::string& spectre_eos_subfile) {
  const BetaSlice slice = read_compose_beta(compose_beta_path);
  const auto log_bounds = check_log_spacing(slice.n_b);
  verify_monotonicity(slice.n_b, slice.pressure,
                      slice.specific_internal_energy);
  const std::vector<double> adiabatic_index =
      compute_adiabatic_index(slice.n_b, slice.pressure);

  h5::H5File<h5::AccessType::ReadWrite> spectre_file(spectre_eos_filename,
                                                     true);
  auto& spectre_eos = spectre_file.insert<h5::EosTable>(
      spectre_eos_subfile, std::vector<std::string>{"number density"},
      std::vector<std::array<double, 2>>{{slice.n_b.front(), slice.n_b.back()}},
      std::vector<size_t>{slice.n_b.size()}, std::vector<bool>{true},
      /*beta_equilibrium=*/true);

  spectre_eos.write_quantity("pressure", to_datavector(slice.pressure));
  spectre_eos.write_quantity("specific internal energy",
                             to_datavector(slice.specific_internal_energy));
  spectre_eos.write_quantity("adiabatic index", to_datavector(adiabatic_index));
  spectre_eos.write_quantity("electron fraction",
                             to_datavector(slice.electron_fraction));

  Parallel::printf(
      "Wrote 1D beta-equilibrium EosTable to %s:/%s\n"
      "  %zu grid points, n_b in [%.4e, %.4e] fm^-3 (log-spaced)\n"
      "  Quantities: pressure, specific internal energy, adiabatic index, "
      "electron fraction\n"
      "  log(n_b) bounds: [%.6f, %.6f]\n",
      spectre_eos_filename, spectre_eos_subfile, slice.n_b.size(),
      slice.n_b.front(), slice.n_b.back(), log_bounds.first, log_bounds.second);
}

}  // namespace

int main(int argc, char** argv) {
  namespace bpo = boost::program_options;
  try {
    bpo::options_description command_line_options(
        "Convert a CompOSE eos.beta 1D beta-equilibrium ASCII slice to a "
        "SpECTRE-formatted 1D HDF5 EosTable readable by "
        "EquationsOfState::Tabulated1D.\n\n"
        "Input: a CompOSE eos.beta ASCII file (four whitespace-separated "
        "columns:\n"
        "  1. baryon number density n_b [fm^-3]\n"
        "  2. electron fraction Y_e_beta [dimensionless]\n"
        "  3. total energy density e = n_b m_n (1 + eps) [MeV/fm^3]\n"
        "  4. pressure p [MeV/fm^3]).\n"
        "Column 3 is converted to the SpECTRE dimensionless specific "
        "internal energy epsilon = e/(n_b m_n) - 1 at read time (this "
        "can be negative for bound cold nuclear matter near saturation).\n\n"
        "Output: an HDF5 file containing a single EosTable subfile with\n"
        "  independent variable: 'number density' [fm^-3, log-spaced]\n"
        "  quantities:\n"
        "    'pressure' [MeV/fm^3, linear]\n"
        "    'specific internal energy' [dimensionless, linear]\n"
        "    'adiabatic index' [dimensionless Gamma_eff = d(ln p)/d(ln n_b), "
        "linear]\n"
        "    'electron fraction' [dimensionless, linear — provenance only]\n\n"
        "adiabatic_index is computed by a 3-point centered log-log finite "
        "difference (io::log_log_derivative) with 2-point one-sided stencils "
        "at the endpoints. The runtime Tabulated1D converts n_b -> rho and "
        "p -> geometric units, computes h = 1 + eps + p/rho, and reconstructs "
        "chi = (p/rho) * Gamma_eff.\n\n"
        "Both p and the computed h are checked for strict monotonicity "
        "before writing; a non-monotonic input file fails the conversion.\n\n"
        "Available options are");

    // clang-format off
    command_line_options.add_options()
      ("help,h", "Describe program options.\n")
      ("compose-beta", bpo::value<std::string>(),
       "Path to the CompOSE eos.beta ASCII file.")
      ("eos-subfile", bpo::value<std::string>(),
       "Path of the EosTable subfile inside the output HDF5 file "
       "(e.g. dd2.eos_beta).")
      ("output,o", bpo::value<std::string>(),
       "Output HDF5 file path (including the .h5 extension).");
    // clang-format on

    bpo::command_line_parser command_line_parser(argc, argv);
    command_line_parser.options(command_line_options);

    bpo::variables_map parsed_command_line_options;
    bpo::store(command_line_parser.run(), parsed_command_line_options);
    bpo::notify(parsed_command_line_options);

    if (parsed_command_line_options.count("help") != 0 or
        parsed_command_line_options.count("compose-beta") == 0 or
        parsed_command_line_options.count("output") == 0 or
        parsed_command_line_options.count("eos-subfile") == 0) {
      Parallel::printf("%s\n", command_line_options);
      return 1;
    }
    convert_file(
        parsed_command_line_options.at("compose-beta").as<std::string>(),
        parsed_command_line_options.at("output").as<std::string>(),
        parsed_command_line_options.at("eos-subfile").as<std::string>());
  } catch (const bpo::error& e) {
    ERROR(e.what());
  }
  return 0;
}
