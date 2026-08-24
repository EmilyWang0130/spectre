// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryDiagnostics.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::Events {

/// Write the primitive-recovery health counters accumulated on this worker.
///
/// This event is intentionally separate from `Events::Completion` so input
/// files can flush periodically and immediately before terminating. It has no
/// effect unless `SPECTRE_RECOVERY_STATS_DIR` is set.
class FlushPrimitiveRecoveryDiagnostics : public ::Event {
 public:
  /// \cond
  explicit FlushPrimitiveRecoveryDiagnostics(CkMigrateMessage* msg)
      : Event(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(FlushPrimitiveRecoveryDiagnostics);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Write primitive-recovery health counters to CSV.";

  FlushPrimitiveRecoveryDiagnostics() = default;

  using compute_tags_for_observation_box = tmpl::list<>;
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  void operator()(Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ArrayIndex& /*array_index*/,
                  const Component* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {
    primitive_recovery_diagnostics::flush();
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return false; }
};

}  // namespace grmhd::ValenciaDivClean::Events
