// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryDiagnostics.hpp"

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> fields{};
  size_t begin = 0;
  while (true) {
    const size_t comma = line.find(',', begin);
    fields.push_back(line.substr(begin, comma - begin));
    if (comma == std::string::npos) {
      return fields;
    }
    begin = comma + 1;
  }
}

std::unordered_map<std::string, std::string> read_only_data_row(
    const std::filesystem::path& output_directory) {
  std::vector<std::filesystem::path> csv_files{};
  for (const auto& entry :
       std::filesystem::directory_iterator{output_directory}) {
    if (entry.path().extension() == ".csv") {
      csv_files.push_back(entry.path());
    }
  }
  REQUIRE(csv_files.size() == 1);
  std::ifstream input{csv_files.front()};
  REQUIRE(input.good());
  std::string header_line{};
  std::string data_line{};
  REQUIRE(static_cast<bool>(std::getline(input, header_line)));
  REQUIRE(static_cast<bool>(std::getline(input, data_line)));
  std::string unexpected_line{};
  CHECK_FALSE(static_cast<bool>(std::getline(input, unexpected_line)));

  const auto headers = split_csv_line(header_line);
  const auto values = split_csv_line(data_line);
  REQUIRE(headers.size() == values.size());
  std::unordered_map<std::string, std::string> result{};
  for (size_t i = 0; i < headers.size(); ++i) {
    result.emplace(headers[i], values[i]);
  }
  return result;
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.PrimitiveRecoveryDiagnostics",
    "[Unit][Evolution]") {
  namespace diagnostics =
      grmhd::ValenciaDivClean::primitive_recovery_diagnostics;
  const std::filesystem::path output_directory =
      std::filesystem::temp_directory_path() /
      ("spectre-recovery-diagnostics-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(output_directory);
  std::filesystem::create_directories(output_directory);
  const std::filesystem::path detail_directory = output_directory / "details";
  std::filesystem::create_directories(detail_directory);
  REQUIRE(setenv("SPECTRE_RECOVERY_STATS_DIR", output_directory.c_str(), 1) ==
          0);
  REQUIRE(setenv("SPECTRE_RECOVERY_FAILURE_DIR", detail_directory.c_str(), 1) ==
          0);
  REQUIRE(setenv("SPECTRE_RECOVERY_FAILURE_LIMIT", "2", 1) == 0);

  diagnostics::set_time(4.25);
  diagnostics::record_point(false);
  diagnostics::record_point(true);
  diagnostics::record_scheme_result(diagnostics::Scheme::Kastaun, false, false,
                                    1.0e-9);
  diagnostics::record_failure_reason(
      diagnostics::Scheme::Kastaun,
      diagnostics::FailureReason::UnbracketedRoot);
  diagnostics::record_scheme_result(diagnostics::Scheme::Palenzuela, true, true,
                                    1.0e-5);
  diagnostics::record_all_schemes_failed();
  diagnostics::flush();

  const auto row = read_only_data_row(output_directory);
  CHECK(row.at("time_bin_start") == "4");
  CHECK(row.at("time_bin_end") == "5");
  CHECK(row.at("points") == "2");
  CHECK(row.at("atmosphere_skips") == "1");
  CHECK(row.at("kastaun_attempts") == "1");
  CHECK(row.at("kastaun_failures") == "1");
  CHECK(row.at("kastaun_unbracketed") == "1");
  CHECK(row.at("kastaun_failed_density_1e-10_to_1e-8") == "1");
  CHECK(row.at("palenzuela_attempts") == "1");
  CHECK(row.at("palenzuela_successes") == "1");
  CHECK(row.at("palenzuela_fallback_successes") == "1");
  CHECK(row.at("all_schemes_failed") == "1");

  // A repeated explicit flush replaces the snapshot with all counts seen so
  // far. This is the behavior used by periodic and final evolution events.
  diagnostics::record_point(false);
  diagnostics::flush();
  CHECK(read_only_data_row(output_directory).at("points") == "3");

  tnsr::I<DataVector, 3, Frame::Inertial> coordinates{size_t{1}, 0.0};
  get<0>(coordinates)[0] = 1.0;
  get<1>(coordinates)[0] = 2.0;
  get<2>(coordinates)[0] = 2.0;
  diagnostics::set_grid_context("[B0,(L0I0,L0I0,L0I0)]", coordinates, true);
  diagnostics::set_point_context(0, 2.0, 0.0, 4.0, 0.0, 0.0, 0.1);
  diagnostics::record_kastaun_root_failure(
      diagnostics::KastaunRootSolve::AuxiliaryBracket, 0.0, 1.0, -1.0, -0.5);
  diagnostics::record_scheme_result(diagnostics::Scheme::NewmanHamlin, true,
                                    true, 2.0);
  diagnostics::flush();

  const auto detail_row = read_only_data_row(detail_directory);
  CHECK(detail_row.at("element_id") == "[B0;(L0I0;L0I0;L0I0)]");
  CHECK(detail_row.at("grid_index") == "0");
  CHECK(detail_row.at("r") == "3");
  CHECK(detail_row.at("element_needed_fixing") == "1");
  CHECK(detail_row.at("tau_is_zero") == "1");
  CHECK(detail_row.at("B_squared_is_zero") == "1");
  CHECK(detail_row.at("root_solve") == "auxiliary_bracket");
  CHECK(detail_row.at("endpoint_signs_bracket_root") == "0");
  CHECK(detail_row.at("fallback_outcome") == "newman_hamlin");

  std::filesystem::remove_all(output_directory);
}
