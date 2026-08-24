// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryDiagnostics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/System/ParallelInfo.hpp"

namespace grmhd::ValenciaDivClean::primitive_recovery_diagnostics {
namespace {

constexpr size_t number_of_schemes = 4;
constexpr size_t number_of_reasons = 4;
constexpr size_t number_of_density_bins = 6;

struct BinCounts {
  uint64_t points = 0;
  uint64_t atmosphere_skips = 0;
  std::array<uint64_t, number_of_schemes> attempts{};
  std::array<uint64_t, number_of_schemes> successes{};
  std::array<uint64_t, number_of_schemes> failures{};
  std::array<uint64_t, number_of_schemes> fallback_successes{};
  std::array<std::array<uint64_t, number_of_reasons>, number_of_schemes>
      failure_reasons{};
  std::array<std::array<uint64_t, number_of_density_bins>, number_of_schemes>
      failure_density_bins{};
  uint64_t all_schemes_failed = 0;
};

struct PointContext {
  size_t grid_index = std::numeric_limits<size_t>::max();
  double conserved_density = std::numeric_limits<double>::quiet_NaN();
  double tau = std::numeric_limits<double>::quiet_NaN();
  double momentum_density_squared = std::numeric_limits<double>::quiet_NaN();
  double momentum_density_dot_magnetic_field =
      std::numeric_limits<double>::quiet_NaN();
  double magnetic_field_squared = std::numeric_limits<double>::quiet_NaN();
  double electron_fraction = std::numeric_limits<double>::quiet_NaN();
};

struct PendingFailure {
  KastaunRootSolve root_solve{};
  double lower_bound = std::numeric_limits<double>::quiet_NaN();
  double upper_bound = std::numeric_limits<double>::quiet_NaN();
  double function_at_lower_bound = std::numeric_limits<double>::quiet_NaN();
  double function_at_upper_bound = std::numeric_limits<double>::quiet_NaN();
};

const char* scheme_name(const Scheme scheme) {
  constexpr std::array<const char*, number_of_schemes> names{
      "kastaun", "newman_hamlin", "palenzuela", "kastaun_hydro"};
  return gsl::at(names, static_cast<size_t>(scheme));
}

const char* root_solve_name(const KastaunRootSolve root_solve) {
  constexpr std::array<const char*, 4> names{"auxiliary_bracket",
                                             "lower_density_corner",
                                             "upper_density_corner", "master"};
  return gsl::at(names, static_cast<size_t>(root_solve));
}

bool environment_flag_is_set(const char* const name) {
  const char* const value = std::getenv(name);
  return value != nullptr and value[0] != '\0' and std::string{value} != "0" and
         std::string{value} != "false" and std::string{value} != "False";
}

size_t density_bin(const double conserved_density) {
  if (not std::isfinite(conserved_density) or conserved_density < 1.0e-12) {
    return 0;
  }
  if (conserved_density < 1.0e-10) {
    return 1;
  }
  if (conserved_density < 1.0e-8) {
    return 2;
  }
  if (conserved_density < 1.0e-6) {
    return 3;
  }
  if (conserved_density < 1.0e-4) {
    return 4;
  }
  return 5;
}

class Recorder {
 public:
  Recorder() {
    const char* const output_directory =
        std::getenv("SPECTRE_RECOVERY_STATS_DIR");
    if (output_directory != nullptr and output_directory[0] != '\0') {
      enabled_ = true;
      rank_ = sys::my_proc();
      path_ = std::string{output_directory} + "/recovery-health-rank-" +
              std::to_string(rank_) + "-pid-" +
              std::to_string(static_cast<long long>(getpid())) + ".csv";
    }

    const char* const failure_directory =
        std::getenv("SPECTRE_RECOVERY_FAILURE_DIR");
    if (failure_directory != nullptr and failure_directory[0] != '\0') {
      detailed_enabled_ = true;
      if (rank_ < 0) {
        rank_ = sys::my_proc();
      }
      detail_path_ = std::string{failure_directory} +
                     "/kastaun-failures-rank-" + std::to_string(rank_) +
                     "-pid-" +
                     std::to_string(static_cast<long long>(getpid())) + ".csv";
      const char* const requested_limit =
          std::getenv("SPECTRE_RECOVERY_FAILURE_LIMIT");
      if (requested_limit != nullptr and requested_limit[0] != '\0') {
        char* end = nullptr;
        const auto parsed_limit = std::strtoull(requested_limit, &end, 10);
        if (end != requested_limit and *end == '\0') {
          detail_limit_ = static_cast<size_t>(parsed_limit);
        }
      }
    }
    force_hydro_for_zero_magnetic_field_ =
        environment_flag_is_set("SPECTRE_RECOVERY_FORCE_HYDRO_FOR_ZERO_B");
  }

  ~Recorder() { flush(); }

  bool enabled() const { return enabled_; }

  bool detailed_enabled() const {
    return detailed_enabled_ and detail_records_written_ < detail_limit_;
  }

  bool force_hydro_for_zero_magnetic_field() const {
    return force_hydro_for_zero_magnetic_field_;
  }

  void flush() {
    emit_pending_failure("outcome_not_recorded");
    write();
  }

  void set_time(const double time) {
    current_time_ = time;
    if (not enabled_) {
      return;
    }
    const double nonnegative_time = std::max(0.0, time);
    current_bin_ = static_cast<size_t>(std::floor(nonnegative_time));
    if (bins_.size() <= current_bin_) {
      bins_.resize(current_bin_ + 1);
    }
  }

  BinCounts* current() {
    if (not enabled_) {
      return nullptr;
    }
    if (bins_.size() <= current_bin_) {
      bins_.resize(current_bin_ + 1);
    }
    return &bins_.at(current_bin_);
  }

  void set_grid_context(
      std::string element_id,
      const tnsr::I<DataVector, 3, Frame::Inertial>& coordinates,
      const bool element_needed_fixing) {
    element_id_ = std::move(element_id);
    coordinates_ = &coordinates;
    element_needed_fixing_ = element_needed_fixing;
  }

  void set_point_context(const size_t grid_index,
                         const double conserved_density, const double tau,
                         const double momentum_density_squared,
                         const double momentum_density_dot_magnetic_field,
                         const double magnetic_field_squared,
                         const double electron_fraction) {
    emit_pending_failure("no_fallback_success");
    point_context_ = PointContext{grid_index,
                                  conserved_density,
                                  tau,
                                  momentum_density_squared,
                                  momentum_density_dot_magnetic_field,
                                  magnetic_field_squared,
                                  electron_fraction};
  }

  void record_kastaun_root_failure(const KastaunRootSolve root_solve,
                                   const double lower_bound,
                                   const double upper_bound,
                                   const double function_at_lower_bound,
                                   const double function_at_upper_bound) {
    if (not detailed_enabled()) {
      return;
    }
    pending_failure_ =
        PendingFailure{root_solve, lower_bound, upper_bound,
                       function_at_lower_bound, function_at_upper_bound};
  }

  void record_fallback_success(const Scheme scheme) {
    if (pending_failure_.has_value()) {
      emit_pending_failure(scheme_name(scheme));
    }
  }

  void record_all_schemes_failed() {
    emit_pending_failure("all_schemes_failed");
  }

 private:
  void emit_pending_failure(const char* const fallback_outcome) {
    if (not pending_failure_.has_value()) {
      return;
    }
    if (detail_records_written_ >= detail_limit_) {
      pending_failure_.reset();
      return;
    }
    std::ofstream output{detail_path_, std::ios::app};
    if (not output) {
      pending_failure_.reset();
      return;
    }
    if (not detail_header_written_) {
      output
          << "time,rank,element_id,grid_index,x,y,z,r,element_needed_fixing,"
             "D,tau,S_squared,S_dot_B,B_squared,Ye,q,r_squared_over_D2,"
             "b_squared_over_D,r_dot_b_squared_over_D3,tau_is_zero,"
             "B_squared_is_zero,root_solve,lower_bound,upper_bound,"
             "f_lower,f_upper,endpoint_signs_bracket_root,fallback_outcome\n";
      detail_header_written_ = true;
    }
    double x = std::numeric_limits<double>::quiet_NaN();
    double y = std::numeric_limits<double>::quiet_NaN();
    double z = std::numeric_limits<double>::quiet_NaN();
    if (coordinates_ != nullptr and
        point_context_.grid_index < get<0>(*coordinates_).size()) {
      x = get<0>(*coordinates_)[point_context_.grid_index];
      y = get<1>(*coordinates_)[point_context_.grid_index];
      z = get<2>(*coordinates_)[point_context_.grid_index];
    }
    const double radius = sqrt(square(x) + square(y) + square(z));
    const double d = point_context_.conserved_density;
    const double q = point_context_.tau / d;
    const double r_squared_over_d2 =
        point_context_.momentum_density_squared / square(d);
    const double b_squared_over_d = point_context_.magnetic_field_squared / d;
    const double r_dot_b_squared_over_d3 =
        square(point_context_.momentum_density_dot_magnetic_field) / cube(d);
    const auto& failure = *pending_failure_;
    const bool endpoints_bracket_root =
        failure.function_at_lower_bound == 0.0 or
        failure.function_at_upper_bound == 0.0 or
        std::signbit(failure.function_at_lower_bound) !=
            std::signbit(failure.function_at_upper_bound);
    std::string safe_element_id = element_id_;
    std::replace(safe_element_id.begin(), safe_element_id.end(), ',', ';');
    output << std::setprecision(17) << current_time_ << ',' << rank_ << ','
           << safe_element_id << ',' << point_context_.grid_index << ',' << x
           << ',' << y << ',' << z << ',' << radius << ','
           << element_needed_fixing_ << ',' << d << ',' << point_context_.tau
           << ',' << point_context_.momentum_density_squared << ','
           << point_context_.momentum_density_dot_magnetic_field << ','
           << point_context_.magnetic_field_squared << ','
           << point_context_.electron_fraction << ',' << q << ','
           << r_squared_over_d2 << ',' << b_squared_over_d << ','
           << r_dot_b_squared_over_d3 << ',' << (point_context_.tau == 0.0)
           << ',' << (point_context_.magnetic_field_squared == 0.0) << ','
           << root_solve_name(failure.root_solve) << ',' << failure.lower_bound
           << ',' << failure.upper_bound << ','
           << failure.function_at_lower_bound << ','
           << failure.function_at_upper_bound << ',' << endpoints_bracket_root
           << ',' << fallback_outcome << '\n';
    ++detail_records_written_;
    pending_failure_.reset();
  }

  void write() const {
    if (not enabled_ or bins_.empty()) {
      return;
    }
    std::ofstream output{path_};
    if (not output) {
      return;
    }
    output << "time_bin_start,time_bin_end,rank,points,atmosphere_skips";
    constexpr std::array<const char*, number_of_schemes> scheme_names{
        "kastaun", "newman_hamlin", "palenzuela", "kastaun_hydro"};
    constexpr std::array<const char*, number_of_reasons> reason_names{
        "unbracketed", "nonconvergence", "rejected_state", "other_exception"};
    constexpr std::array<const char*, number_of_density_bins> density_bin_names{
        "lt_1e-12",     "1e-12_to_1e-10", "1e-10_to_1e-8",
        "1e-8_to_1e-6", "1e-6_to_1e-4",   "ge_1e-4"};
    for (const char* const scheme : scheme_names) {
      output << ',' << scheme << "_attempts," << scheme << "_successes,"
             << scheme << "_failures," << scheme << "_fallback_successes";
      for (const char* const reason : reason_names) {
        output << ',' << scheme << '_' << reason;
      }
      for (const char* const density : density_bin_names) {
        output << ',' << scheme << "_failed_density_" << density;
      }
    }
    output << ",all_schemes_failed\n";

    for (size_t bin = 0; bin < bins_.size(); ++bin) {
      const auto& counts = gsl::at(bins_, bin);
      if (counts.points == 0) {
        continue;
      }
      output << std::setprecision(17) << static_cast<double>(bin) << ','
             << static_cast<double>(bin + 1) << ',' << rank_ << ','
             << counts.points << ',' << counts.atmosphere_skips;
      for (size_t scheme = 0; scheme < number_of_schemes; ++scheme) {
        output << ',' << gsl::at(counts.attempts, scheme) << ','
               << gsl::at(counts.successes, scheme) << ','
               << gsl::at(counts.failures, scheme) << ','
               << gsl::at(counts.fallback_successes, scheme);
        for (size_t reason = 0; reason < number_of_reasons; ++reason) {
          output << ','
                 << gsl::at(gsl::at(counts.failure_reasons, scheme), reason);
        }
        for (size_t density = 0; density < number_of_density_bins; ++density) {
          output << ','
                 << gsl::at(gsl::at(counts.failure_density_bins, scheme),
                            density);
        }
      }
      output << ',' << counts.all_schemes_failed << '\n';
    }
  }

  bool enabled_{false};
  bool detailed_enabled_{false};
  bool force_hydro_for_zero_magnetic_field_{false};
  int rank_{-1};
  size_t current_bin_{0};
  double current_time_{0.0};
  std::string path_{};
  std::string detail_path_{};
  std::vector<BinCounts> bins_{};
  size_t detail_limit_{20};
  size_t detail_records_written_{0};
  bool detail_header_written_{false};
  std::string element_id_{"unknown"};
  const tnsr::I<DataVector, 3, Frame::Inertial>* coordinates_{nullptr};
  bool element_needed_fixing_{false};
  PointContext point_context_{};
  std::optional<PendingFailure> pending_failure_{};
};

Recorder& recorder() {
  static thread_local Recorder instance{};
  return instance;
}

size_t index(const Scheme scheme) { return static_cast<size_t>(scheme); }
size_t index(const FailureReason reason) { return static_cast<size_t>(reason); }

}  // namespace

void set_time(const double time) { recorder().set_time(time); }

void set_grid_context(
    std::string element_id,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coordinates,
    const bool element_needed_fixing) {
  recorder().set_grid_context(std::move(element_id), coordinates,
                              element_needed_fixing);
}

void set_point_context(const size_t grid_index, const double conserved_density,
                       const double tau, const double momentum_density_squared,
                       const double momentum_density_dot_magnetic_field,
                       const double magnetic_field_squared,
                       const double electron_fraction) {
  recorder().set_point_context(grid_index, conserved_density, tau,
                               momentum_density_squared,
                               momentum_density_dot_magnetic_field,
                               magnetic_field_squared, electron_fraction);
}

bool detailed_failure_logging_enabled() {
  return recorder().detailed_enabled();
}

void record_kastaun_root_failure(const KastaunRootSolve root_solve,
                                 const double lower_bound,
                                 const double upper_bound,
                                 const double function_at_lower_bound,
                                 const double function_at_upper_bound) {
  recorder().record_kastaun_root_failure(root_solve, lower_bound, upper_bound,
                                         function_at_lower_bound,
                                         function_at_upper_bound);
}

bool force_hydro_recovery_for_zero_magnetic_field() {
  return recorder().force_hydro_for_zero_magnetic_field();
}

void record_point(const bool skipped_as_atmosphere) {
  if (auto* const counts = recorder().current(); counts != nullptr) {
    ++counts->points;
    if (skipped_as_atmosphere) {
      ++counts->atmosphere_skips;
    }
  }
}

void record_scheme_result(const Scheme scheme, const bool succeeded,
                          const bool is_fallback,
                          const double conserved_density) {
  if (auto* const counts = recorder().current(); counts != nullptr) {
    const size_t scheme_index = index(scheme);
    ++gsl::at(counts->attempts, scheme_index);
    if (succeeded) {
      ++gsl::at(counts->successes, scheme_index);
      if (is_fallback) {
        ++gsl::at(counts->fallback_successes, scheme_index);
      }
    } else {
      ++gsl::at(counts->failures, scheme_index);
      ++gsl::at(gsl::at(counts->failure_density_bins, scheme_index),
                density_bin(conserved_density));
    }
  }
  if (succeeded and is_fallback) {
    recorder().record_fallback_success(scheme);
  }
}

void record_failure_reason(const Scheme scheme, const FailureReason reason) {
  if (auto* const counts = recorder().current(); counts != nullptr) {
    ++gsl::at(gsl::at(counts->failure_reasons, index(scheme)), index(reason));
  }
}

void record_all_schemes_failed() {
  if (auto* const counts = recorder().current(); counts != nullptr) {
    ++counts->all_schemes_failed;
  }
  recorder().record_all_schemes_failed();
}

void flush() { recorder().flush(); }

}  // namespace grmhd::ValenciaDivClean::primitive_recovery_diagnostics
