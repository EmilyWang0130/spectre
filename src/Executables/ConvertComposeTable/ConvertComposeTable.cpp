// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>

#include <array>
#include <cmath>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "IO/ComposeTable.hpp"
#include "IO/ComposeTableDerivatives.hpp"
#include "IO/H5/EosTable.hpp"
#include "IO/H5/File.hpp"
#include "Parallel/Printf/Printf.hpp"
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

  ASSERT(pressure.size() == nN * nT * nYe,
         "Pressure size does not match table dimensions.");
  ASSERT(eps.size() == pressure.size(),
         "Epsilon size does not match pressure size.");

  // Reconstruct coordinate grids (needed for nonuniform spacing, e.g. log T)
  const auto T_grid = make_grid_1d(compose_table.temperature_bounds(), nT,
                                   compose_table.temperature_log_spacing());
  const auto Ye_grid =
      make_grid_1d(compose_table.electron_fraction_bounds(), nYe,
                   compose_table.electron_fraction_log_spacing());

  const DataVector zeta = io::compute_zeta_from_pressure_and_eps(
      pressure, eps, T_grid, Ye_grid, nN, nT, nYe);
  for (const auto& [quantity_name, quantity_data] : data) {
    spectre_eos.write_quantity(quantity_name, quantity_data);
  }
  spectre_eos.write_quantity("zeta", zeta);
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
        "7\n"
        "Please select the indices of the thermodynamic quantities...\n"
        " Index #           1 ?"
        "1\n"
        " Index #           2 ?"
        "2\n"
        "The remaining are: 3 4 5 7 12\n"
        "The following function values and derivatives of the free energy...\n"
        "1\n"
        "Please select the indices of the thermodynamic...\n"
        "1\n"
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
