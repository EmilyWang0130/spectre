// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"

class DataVector;

namespace grmhd::ValenciaDivClean::primitive_recovery_diagnostics {

enum class Scheme : uint8_t {
  Kastaun,
  NewmanHamlin,
  Palenzuela,
  KastaunHydro,
};

enum class FailureReason : uint8_t {
  UnbracketedRoot,
  Nonconvergence,
  RejectedState,
  OtherException,
};

enum class KastaunRootSolve : uint8_t {
  AuxiliaryBracket,
  LowerDensityCorner,
  UpperDensityCorner,
  Master,
};

/// Select the one-code-unit time bin for subsequent counters on this worker.
/// Diagnostics are enabled only when SPECTRE_RECOVERY_STATS_DIR is set.
void set_time(double time);

/// Attach spatial metadata to detailed failure records made during the next
/// primitive-recovery call. The coordinates are not copied and must remain
/// alive until that call returns.
void set_grid_context(
    std::string element_id,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coordinates,
    bool element_needed_fixing);

/// Attach the conservative state for the grid point currently being inverted.
void set_point_context(size_t grid_index, double conserved_density, double tau,
                       double momentum_density_squared,
                       double momentum_density_dot_magnetic_field,
                       double magnetic_field_squared, double electron_fraction);

/// True only when SPECTRE_RECOVERY_FAILURE_DIR requests bounded point records.
bool detailed_failure_logging_enabled();

/// Record which of Kastaun's internal scalar root solves threw. The record is
/// held until the fallback outcome for the same point is known.
void record_kastaun_root_failure(KastaunRootSolve root_solve,
                                 double lower_bound, double upper_bound,
                                 double function_at_lower_bound,
                                 double function_at_upper_bound);

/// Diagnostic-only A/B switch. The default is false, preserving the original
/// primitive-recovery selection exactly.
bool force_hydro_recovery_for_zero_magnetic_field();

void record_point(bool skipped_as_atmosphere);

void record_scheme_result(Scheme scheme, bool succeeded, bool is_fallback,
                          double conserved_density);

void record_failure_reason(Scheme scheme, FailureReason reason);

void record_all_schemes_failed();

/// Write the counters accumulated on this worker to its CSV file.
///
/// Charm++ shutdown does not guarantee that thread-local destructors run, so
/// production evolutions must call this function from an event before setting
/// their termination flag. Repeated calls safely replace the CSV with a
/// complete snapshot of all counters accumulated so far.
void flush();

}  // namespace grmhd::ValenciaDivClean::primitive_recovery_diagnostics
