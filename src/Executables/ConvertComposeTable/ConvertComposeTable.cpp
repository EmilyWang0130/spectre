// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "IO/ComposeTable.hpp"
#include "IO/ComposeTableDerivatives.hpp"
#include "IO/H5/EosTable.hpp"
#include "IO/H5/File.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
std::vector<double> make_grid_1d(const std::array<double, 2>& bounds,
                                 const size_t npts, const bool log_spacing) {
  std::vector<double> grid(npts);
  if (npts == 0) {
    ERROR("Number of points is zero.");
  }
  if (npts == 1) {
    grid[0] = bounds[0];
    return grid;
  }
  if (log_spacing) {
    const double log_lo = std::log(bounds[0]);
    const double log_hi = std::log(bounds[1]);
    const double dlog = (log_hi - log_lo) / static_cast<double>(npts - 1);
    for (size_t i = 0; i < npts; ++i) {
      grid[i] = std::exp(log_lo + dlog * static_cast<double>(i));
    }
  } else {
    const double d = (bounds[1] - bounds[0]) / static_cast<double>(npts - 1);
    for (size_t i = 0; i < npts; ++i) {
      grid[i] = bounds[0] + d * static_cast<double>(i);
    }
  }
  return grid;
}

void convert_file(const std::string& compose_directory,
                  const std::string& spectre_eos_filename,
                  const std::string& spectre_eos_subfile) {
  const io::ComposeTable compose_table(compose_directory);
  h5::H5File<h5::AccessType::ReadWrite> spectre_file(spectre_eos_filename,
                                                     true);
  auto& spectre_eos = spectre_file.insert<h5::EosTable>(
      spectre_eos_subfile,
      std::vector<std::string>{"number density", "temperature",
                               "electron fraction"},
      std::vector{compose_table.number_density_bounds(),
                  compose_table.temperature_bounds(),
                  compose_table.electron_fraction_bounds()},
      std::vector{compose_table.number_density_number_of_points(),
                  compose_table.temperature_number_of_points(),
                  compose_table.electron_fraction_number_of_points()},
      std::vector{compose_table.number_density_log_spacing(),
                  compose_table.temperature_log_spacing(),
                  compose_table.electron_fraction_log_spacing()},
      compose_table.beta_equilibrium());

  const auto& data = compose_table.data();

  // Required base quantities
  const DataVector& pressure = data.at("pressure");
  const DataVector& eps = data.at("specific internal energy");

  const size_t nN = compose_table.number_density_number_of_points();
  const size_t nT = compose_table.temperature_number_of_points();
  const size_t nYe = compose_table.electron_fraction_number_of_points();
  const size_t ntot = nN * nT * nYe;

  ASSERT(pressure.size() == ntot,
         "Pressure size does not match table dimensions.");
  ASSERT(eps.size() == pressure.size(),
         "Epsilon size does not match pressure size.");

  // Free-energy derivatives used to compute zeta analytically (CompOSE Table
  // 7.3 indices 3,4,5,8,9). They are inputs only; their CompOSE names contain
  // '/', which HDF5 would treat as group separators, so they are consumed here
  // and not written to the output table.
  const std::array<std::string, 5> zeta_derivative_keys{
      {"d2 F / d T2", "d2 F / d T d n_b", "d2 F / d T d Y_e",
       "d2 F / d n_b d Y_e", "d F / d Y_e"}};
  const bool have_derivatives = std::all_of(
      zeta_derivative_keys.begin(), zeta_derivative_keys.end(),
      [&data](const std::string& key) { return data.count(key) == 1; });

  // Write everything available from the CompOSE table except those
  // derivatives, the sound speed squared, and kappa. The latter two are
  // reconstructed from p, s, and ε below; CompOSE's tabulated Q12 and Q11 are
  // unreliable at table corners.
  for (const auto& [quantity_name, quantity_data] : data) {
    if (std::find(zeta_derivative_keys.begin(), zeta_derivative_keys.end(),
                  quantity_name) != zeta_derivative_keys.end()) {
      continue;
    }
    if (quantity_name == "sound speed squared" or quantity_name == "kappa") {
      continue;
    }
    spectre_eos.write_quantity(quantity_name, quantity_data);
  }

  // Replace CompOSE's tabulated Q12 (sound speed squared) with one
  // reconstructed by finite-differencing pressure and entropy on the table
  // grid. CompOSE's Q12 is computed internally from high-order interpolation
  // of the free energy and is known to go unphysical (negative, or > 1) at
  // dilute-table corners (a small percent of the DD2 table, for example).
  // The reconstruction follows the recipe used by PyCompOSE
  // (https://github.com/computationalrelativity/PyCompOSE) — see
  // compute_cs2_from_pressure_and_entropy in ComposeTableDerivatives. For
  // β-equilibrium tables Y_e is slaved and the table is 2D, so the
  // independent (∂s/∂T)/(∂s/∂n_b) derivatives that the recipe needs do not
  // mean the same thing; in that case we leave the original Q12 in place and
  // warn.
  //
  // The cs² floor matches PyCompOSE's DD2 example (1e-6).
  constexpr double cs2_floor = 1.0e-6;
  const bool have_inputs_for_cs2_reconstruction =
      data.count("pressure") == 1 and data.count("specific entropy") == 1 and
      data.count("specific internal energy") == 1;
  if (compose_table.beta_equilibrium() and
      data.count("sound speed squared") == 1) {
    Parallel::printf(
        "WARNING: source CompOSE table is flagged beta-equilibrium; leaving "
        "sound speed squared (Q12) as-is. The finite-difference reconstruction "
        "from p and s assumes T and Y_e are independent table axes.\n");
    spectre_eos.write_quantity("sound speed squared",
                               data.at("sound speed squared"));
  } else if (not have_inputs_for_cs2_reconstruction) {
    if (data.count("sound speed squared") == 1) {
      Parallel::printf(
          "WARNING: source CompOSE table does not contain all of pressure "
          "(Q1), specific entropy (Q2), and specific internal energy (Q7), so "
          "the cs² reconstruction is skipped; writing CompOSE's Q12 verbatim. "
          "Regenerate with indices 1 2 7 12 in eos.quantities to enable the "
          "reconstruction.\n");
      spectre_eos.write_quantity("sound speed squared",
                                 data.at("sound speed squared"));
    } else {
      ERROR(
          "Source CompOSE table is missing sound speed squared (Q12) and the "
          "quantities needed to reconstruct it (pressure Q1, specific entropy "
          "Q2, specific internal energy Q7). Regenerate the CompOSE table "
          "with all of indices 1 2 7 12 in eos.quantities.");
    }
  } else {
    const auto T_grid = make_grid_1d(compose_table.temperature_bounds(), nT,
                                     compose_table.temperature_log_spacing());
    const auto nb_grid =
        make_grid_1d(compose_table.number_density_bounds(), nN,
                     compose_table.number_density_log_spacing());
    const DataVector reconstructed_cs2 =
        io::compute_cs2_from_pressure_and_entropy(
            data.at("pressure"), data.at("specific entropy"),
            data.at("specific internal energy"), nb_grid, T_grid,
            hydro::units::nuclear::neutron_mass, nN, nT, nYe, cs2_floor);

    if (data.count("sound speed squared") == 1) {
      const DataVector& original_cs2 = data.at("sound speed squared");
      size_t n_negative_original = 0;
      size_t n_above_one_original = 0;
      for (size_t i = 0; i < original_cs2.size(); ++i) {
        if (original_cs2[i] < 0.0) {
          ++n_negative_original;
        }
        if (original_cs2[i] > 1.0) {
          ++n_above_one_original;
        }
      }
      Parallel::printf(
          "Replacing CompOSE sound speed squared (Q12) with finite-difference "
          "reconstruction from p and s; original had %zu/%zu points with "
          "cs² < 0 and %zu with cs² > 1. Reconstructed cs² floored at "
          "%.1e.\n",
          n_negative_original, original_cs2.size(), n_above_one_original,
          cs2_floor);
    } else {
      Parallel::printf(
          "Computing sound speed squared from finite-difference of p and s "
          "(CompOSE Q12 was not requested in eos.quantities); floored at "
          "%.1e.\n",
          cs2_floor);
    }
    spectre_eos.write_quantity("sound speed squared", reconstructed_cs2);
  }

  // Replace CompOSE's tabulated Q11 (kappa = ∂p/∂ε) with a finite-difference
  // reconstruction from p and ε on the T axis at fixed (n_b, Y_e):
  //   κ = (∂p/∂T) / (∂ε/∂T)
  // Same motivation as for cs²: CompOSE's tabulated Q11 has interpolation
  // noise (negative κ values are common at table corners; in the DD2 HS table
  // ~1% of points have κ < 0, which is unphysical for any stable EoS). For
  // β-equilibrium tables we keep the CompOSE Q11 because Y_e is slaved (T is
  // also reduced by the β-eq condition, but the formula still nominally
  // applies; we err on the side of preserving the original behaviour for
  // β-eq inputs and re-evaluate later if needed). For single-T (cold)
  // tables we cannot FD on T at all and fall back to zeros.
  //
  // Floor matches the cs² treatment: 1e-6 in input units (MeV/fm^3 when called
  // on CompOSE data).
  constexpr double kappa_floor = 1.0e-6;
  // The reconstructed kappa is also what repairs zeta at low temperature
  // below, so keep a copy of it (and of whether we actually reconstructed it).
  DataVector kappa_for_zeta_repair(ntot, 0.0);
  bool have_reconstructed_kappa = false;
  const bool have_inputs_for_kappa_reconstruction =
      nT >= 2 and data.count("pressure") == 1 and
      data.count("specific internal energy") == 1;
  if (compose_table.beta_equilibrium() and data.count("kappa") == 1) {
    Parallel::printf(
        "WARNING: source CompOSE table is flagged beta-equilibrium; leaving "
        "kappa (Q11) as-is. The finite-difference reconstruction from p and "
        "ε assumes T is an independent table axis.\n");
    spectre_eos.write_quantity("kappa", data.at("kappa"));
  } else if (not have_inputs_for_kappa_reconstruction) {
    if (data.count("kappa") == 1) {
      Parallel::printf(
          "WARNING: cannot reconstruct kappa from p and ε (either only one T "
          "slice or missing pressure / specific internal energy); writing "
          "CompOSE's Q11 verbatim.\n");
      spectre_eos.write_quantity("kappa", data.at("kappa"));
    } else {
      Parallel::printf(
          "WARNING: source CompOSE table does not contain kappa (Q11 dp/dε) "
          "and cannot be reconstructed (either only one T slice or missing "
          "pressure / specific internal energy); writing kappa = 0. "
          "Regenerate with indices 1 7 11 in eos.quantities and a multi-T "
          "grid to enable the reconstruction.\n");
      spectre_eos.write_quantity("kappa", DataVector(ntot, 0.0));
    }
  } else {
    const auto T_grid_kappa =
        make_grid_1d(compose_table.temperature_bounds(), nT,
                     compose_table.temperature_log_spacing());
    const DataVector reconstructed_kappa =
        io::compute_kappa_from_pressure_and_energy(
            data.at("pressure"), data.at("specific internal energy"),
            T_grid_kappa, hydro::units::nuclear::neutron_mass, nN, nT, nYe,
            kappa_floor);

    if (data.count("kappa") == 1) {
      const DataVector& original_kappa = data.at("kappa");
      size_t n_negative_original = 0;
      for (size_t i = 0; i < original_kappa.size(); ++i) {
        if (original_kappa[i] < 0.0) {
          ++n_negative_original;
        }
      }
      Parallel::printf(
          "Replacing CompOSE kappa (Q11) with finite-difference reconstruction "
          "from p and ε; original had %zu/%zu points with κ < 0. Reconstructed "
          "κ floored at %.1e.\n",
          n_negative_original, original_kappa.size(), kappa_floor);
    } else {
      Parallel::printf(
          "Computing kappa from finite-difference of p and ε (CompOSE Q11 was "
          "not requested in eos.quantities); floored at %.1e.\n",
          kappa_floor);
    }
    spectre_eos.write_quantity("kappa", reconstructed_kappa);
    kappa_for_zeta_repair = reconstructed_kappa;
    have_reconstructed_kappa = true;
  }

  // zeta = ∂P/∂Ye is computed analytically from the CompOSE free-energy
  // derivatives (Table 7.3 indices 3,4,5,8,9). It is ill-defined for
  // beta-equilibrium tables (Ye is fixed by beta equilibrium rather than free).
  // In either the beta-equilibrium case or when those derivatives weren't
  // tabulated we cannot compute it, so — as with kappa — emit a warning and
  // write zeros to keep the H5 format uniform.
  if (compose_table.beta_equilibrium()) {
    Parallel::printf(
        "WARNING: source CompOSE table is flagged beta-equilibrium; writing "
        "zeta = 0 (∂P/∂Ye is undefined when Ye is fixed by beta "
        "equilibrium).\n");
    spectre_eos.write_quantity("zeta", DataVector(ntot, 0.0));
  } else if (not have_derivatives) {
    Parallel::printf(
        "WARNING: source CompOSE table does not contain the free-energy "
        "derivatives (Table 7.3 indices 3,4,5,8,9) needed for zeta = "
        "(∂P/∂Ye)_{rho,eps}; writing zeta = 0. Regenerate with derivative "
        "indices 3 4 5 8 9 in eos.quantities to get the true value.\n");
    spectre_eos.write_quantity("zeta", DataVector(ntot, 0.0));
  } else {
    const auto T_grid = make_grid_1d(compose_table.temperature_bounds(), nT,
                                     compose_table.temperature_log_spacing());
    const auto nb_grid =
        make_grid_1d(compose_table.number_density_bounds(), nN,
                     compose_table.number_density_log_spacing());

    // The analytic zeta divides by eps_T = -T d²F/dT², which vanishes together
    // with its numerator p_T in degenerate matter: at low T the formula is a
    // 0/0 evaluated on two extrapolated derivative columns, and it fails
    // loudly. On the Togashi table at T = 0.1 MeV, 26.4% of nodes in the
    // stellar band n_b ∈ [0.05, 0.62] fm^-3 have d²F/dT² > 0 — a negative
    // specific heat c_V = -T d²F/dT², which no stable EoS can have — and 39.5%
    // of the resulting zeta values come out with the wrong sign. Below
    // zeta_cold_repair_temperature we therefore replace the ratio p_T/eps_T
    // (which is exactly kappa, in fm^-3) by the reconstructed kappa frozen at
    // the lowest trustworthy row; kappa is genuinely T-independent there.
    // Rows above the switch are untouched, so the warm table — where the
    // stored zeta is independently good to 1-3% — is bit-for-bit unchanged.
    // This mirrors what we already do for CompOSE's Q11 and Q12; zeta was the
    // one derived quantity still taken on trust. See
    // compute_zeta_from_free_energy_derivatives for the full justification.
    //
    // 0.4 MeV selects T = 0.4365 MeV on the Togashi grid, the lowest row at
    // which the stored zeta is still good to a few percent.
    constexpr double zeta_cold_repair_temperature = 0.4;
    // Below this density the finite-difference kappa has collapsed onto its
    // own floor and carries no information, so leave zeta alone there.
    constexpr double zeta_cold_repair_minimum_number_density = 1.0e-6;

    if (not have_reconstructed_kappa) {
      Parallel::printf(
          "WARNING: kappa was not reconstructed from p and ε, so the "
          "low-temperature repair of zeta is disabled. zeta below %.2f MeV is "
          "formed from the 0/0 ratio (∂p/∂T)/(∂ε/∂T) of two CompOSE "
          "free-energy derivative columns and is unreliable there.\n",
          zeta_cold_repair_temperature);
    } else {
      // Count the points the repair covers, and how many of them are provably
      // unphysical in the source table (c_V = -T d²F/dT² <= 0).
      const DataVector& d2f_dt2 = data.at("d2 F / d T2");
      size_t n_repaired = 0;
      size_t n_negative_heat_capacity = 0;
      for (size_t iT = 0; iT < nT; ++iT) {
        if (T_grid[iT] >= zeta_cold_repair_temperature) {
          continue;
        }
        for (size_t in = 0; in < nN; ++in) {
          if (nb_grid[in] < zeta_cold_repair_minimum_number_density) {
            continue;
          }
          for (size_t iYe = 0; iYe < nYe; ++iYe) {
            ++n_repaired;
            if (d2f_dt2[(iT * nN + in) * nYe + iYe] >= 0.0) {
              ++n_negative_heat_capacity;
            }
          }
        }
      }
      Parallel::printf(
          "Repairing zeta below T = %.2f MeV by freezing the reconstructed "
          "kappa at the lowest row at or above it: %zu/%zu points rebuilt, of "
          "which %zu had d²F/dT² >= 0 (negative specific heat) in the source "
          "table. Warmer rows keep the analytic p_Ye - p_T eps_Ye / eps_T.\n",
          zeta_cold_repair_temperature, n_repaired, ntot,
          n_negative_heat_capacity);
    }

    spectre_eos.write_quantity(
        "zeta",
        io::compute_zeta_from_free_energy_derivatives(
            data.at("d2 F / d T2"), data.at("d2 F / d T d n_b"),
            data.at("d2 F / d T d Y_e"), data.at("d2 F / d n_b d Y_e"),
            data.at("d F / d Y_e"), kappa_for_zeta_repair, nb_grid, T_grid, nN,
            nT, nYe,
            have_reconstructed_kappa ? zeta_cold_repair_temperature : -1.0,
            zeta_cold_repair_minimum_number_density));
  }
}
}  // namespace

int main(int argc, char** argv) {
  namespace bpo = boost::program_options;
  try {
    bpo::options_description command_line_options(
        "This executable converts an ASCII formatted CompOSE 3d equation of "
        "state table into a SpECTRE-formatted HDF5 table. This reduces the "
        "file size by about a factor of 4. We don't use the CompOSE HDF5 "
        "tables since that requires an HDF5 that works with Fortran.\n"
        "Note: support for 1d and 2d tables can be added if the CompOSE ASCII "
        "reader is generalized to support them.\n\n"
        "Generating the ASCII table using compose:\n"
        "1.\n"
        "Download from: https://compose.obspm.fr/software (there's a "
        "GitLab link)\n\n"
        "2.\n"
        "Build by running 'make' in the directory. This process will create "
        "the 'compose' executable.\n\n"
        "3.\n"
        "Download EOS from https://compose.obspm.fr/table We will use "
        "https://compose.obspm.fr/eos/34 as an example. To download, use wget "
        "on the link from the 'eos.zip' file, or download the 'eos.zip' file "
        "directly.\n\n"
        "4.\n"
        "Unzip the eos.zip file. This will create multiple 'eos.*' files in "
        "the current directory.\n\n"
        "5.\n"
        "Run the 'compose' executable in the directory with all the eos "
        "files. There are 3 main options and you will run the executable "
        "3 times. Each main option or 'task' has a bunch of numerical value "
        "inputs."
        "\n"
        "Task 1\n"
        "How many regular thermodynamic quantities...\n"
        "8\n"
        "Please select the indices of the thermodynamic quantities...\n"
        " Index #           1 ?"
        "1\n"
        " Index #           2 ?"
        "2\n"
        "The remaining are: 3 4 5 7 11 12\n"
        "(Index 11 is dp/dε, stored as 'kappa' in SpECTRE and required by "
        "Tabulated3D.)\n"
        "The following function values and derivatives of the free energy...\n"
        "5\n"
        "Please select the indices of the thermodynamic...\n"
        "3 4 5 8 9\n"
        "(These are d2F/dT2, d2F/dTdn_b, d2F/dTdY_e, d2F/dn_bdY_e, and "
        "dF/dY_e. SpECTRE uses them to compute the bulk-viscosity-like "
        "quantity zeta = (dp/dY_e)_{rho,eps} analytically. If they are "
        "omitted, zeta is written as zeros, with a warning, like kappa.)\n"
        "How many particles do you want to select for the file eos.table?\n"
        "0\n"
        "There are average mass, charge and neutron numbers...\n"
        "0\n"
        "There are microscopic data available of the following type:...\n"
        "0\n"
        "There are error estimates available of the following type:...\n"
        "0\n"
        "If successful, you should see new file 'eos.quantities' generated. "
        "Now rerun compose for Task2.\n\n"
        "Task 2\n"
        "Temperature interpolation order:\n"
        "3\n"
        "Baryon density interpolation order:\n"
        "3\n"
        "Hadronic charge fraction interpolation order:\n"
        "3\n"
        "beta-equilibrium\n"
        "0\n"
        "entropy per baryon\n"
        "0\n"
        "Please select the tabulation scheme for the parameters from\n"
        "1\n"
        "Get the lower and upper bounds as well as the grid points from the "
        "compose website for your EOS. Spacing should be\n"
        "T: log\n"
        "n_b: log\n"
        "Y_q: linear\n"
        "You must enter the bounds as:\n"
        "lower upper\n"
        "If successful, you should see new file 'eos.parameters' generated. "
        "Now rerun compose for Task3.\n\n"
        "Task 3\n"
        "This will just run, no options needed, but it can take quite a long "
        "time.\n"
        "If successful, it will list 'file eos.table written', along with the "
        "respective labels [and units] of the columns, for example, '1 "
        "temperature T [MeV]'.\n\n"
        "Available options are");

    // clang-format off
    command_line_options.add_options()
        ("help,h", "Describe program options.\n")
        ("compose-directory", bpo::value<std::string>(),
         "The directory in which the CompOSE eos.quantities, eos.parameters, "
         "and eos.table files are.")
        ("eos-subfile", bpo::value<std::string>(),
         "Path of where to write the subfile SpECTRE EOS Table inside the "
         "HDF5 file.")
        ("output,o", bpo::value<std::string>(),
         "Path of the output HDF5 file to which the EOS subfile will be "
         "written, including the .h5 extension.")
        ;
    // clang-format on

    bpo::command_line_parser command_line_parser(argc, argv);
    command_line_parser.options(command_line_options);

    bpo::variables_map parsed_command_line_options;
    bpo::store(command_line_parser.run(), parsed_command_line_options);
    bpo::notify(parsed_command_line_options);

    if (parsed_command_line_options.count("help") != 0 or
        parsed_command_line_options.count("compose-directory") == 0 or
        parsed_command_line_options.count("output") == 0 or
        parsed_command_line_options.count("eos-subfile") == 0) {
      Parallel::printf("%s\n", command_line_options);
      return 1;
    }
    convert_file(
        parsed_command_line_options.at("compose-directory").as<std::string>(),
        parsed_command_line_options.at("output").as<std::string>(),
        parsed_command_line_options.at("eos-subfile").as<std::string>());
  } catch (const bpo::error& e) {
    ERROR(e.what());
  }
  return 0;
}
