// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Utilities/PrettyType.hpp"
#include "Utilities/System/ParallelInfo.hpp"

namespace Parallel::phase3_action_timing {

enum class RecordKind : uint8_t { Action = 1, PerformAlgorithm = 2 };

enum class ExecutionResult : uint8_t {
  Continue = 0,
  Retry = 1,
  Pause = 2,
  Halt = 3
};

// This fixed-size format is intentionally simple so the diagnostic can be
// decoded without linking against SpECTRE. Timestamps use CLOCK_MONOTONIC via
// std::chrono::steady_clock and are comparable across processes on one node.
struct Record {
  uint64_t start_ns;
  uint64_t end_ns;
  uint64_t component_id;
  uint64_t chare_id;
  uint64_t action_id;
  uint32_t sequence;
  uint16_t phase;
  uint16_t phase_index;
  uint16_t action_index;
  uint16_t next_action_index;
  uint8_t kind;
  uint8_t result;
  uint16_t reserved;
  int32_t pe;
  uint32_t pid;
};
static_assert(sizeof(Record) == 64);

struct FileHeader {
  char magic[8];
  uint32_t version;
  uint32_t header_size;
  uint32_t record_size;
  uint32_t pe;
  uint32_t pid;
  uint32_t reserved;
  uint64_t record_count;
  uint64_t dropped_count;
  uint64_t clock_period_numerator;
  uint64_t clock_period_denominator;
};
static_assert(sizeof(FileHeader) == 64);

inline uint64_t timestamp_ns() noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

inline uint64_t fnv1a(const std::string& text) noexcept {
  uint64_t result = 14695981039346656037ULL;
  for (const unsigned char character : text) {
    result ^= static_cast<uint64_t>(character);
    result *= 1099511628211ULL;
  }
  return result;
}

template <typename ArrayIndex>
uint64_t chare_id(const ArrayIndex& array_index) noexcept {
  if constexpr (std::is_integral_v<ArrayIndex> or std::is_enum_v<ArrayIndex>) {
    return static_cast<uint64_t>(array_index);
  } else if constexpr (std::is_trivially_copyable_v<ArrayIndex> and
                       sizeof(ArrayIndex) <= sizeof(uint64_t)) {
    uint64_t result = 0;
    std::memcpy(&result, &array_index, sizeof(ArrayIndex));
    return result;
  } else {
    return 0;
  }
}

template <typename ArrayIndex>
std::string chare_name(const ArrayIndex& array_index) {
  if constexpr (requires(std::ostream& stream) { stream << array_index; }) {
    std::ostringstream stream{};
    stream << array_index;
    return stream.str();
  } else {
    return "unprintable-array-index";
  }
}

class Recorder {
 public:
  Recorder()
      : pe_(sys::my_proc()),
        pid_(static_cast<uint32_t>(getpid())),
        output_directory_([]() {
          const char* const value = std::getenv("SPECTRE_PHASE3_TIMING_DIR");
          return value == nullptr ? std::string{"."} : std::string{value};
        }()),
        maximum_records_([]() {
          const char* const value = std::getenv("SPECTRE_PHASE3_MAX_RECORDS");
          if (value == nullptr) {
            return size_t{2000000};
          }
          char* end = nullptr;
          const unsigned long long parsed = std::strtoull(value, &end, 10);
          if (end == value or *end != '\0' or parsed == 0 or
              parsed > std::numeric_limits<size_t>::max()) {
            return size_t{2000000};
          }
          return static_cast<size_t>(parsed);
        }()) {
    records_.reserve(std::min(maximum_records_, size_t{262144}));
  }

  Recorder(const Recorder&) = delete;
  Recorder& operator=(const Recorder&) = delete;
  Recorder(Recorder&&) = delete;
  Recorder& operator=(Recorder&&) = delete;

  ~Recorder() { flush(); }

  void add_record(const uint64_t start_ns, const uint64_t end_ns,
                  const uint64_t component_id, const uint64_t chare_id,
                  const uint64_t action_id, const uint16_t phase,
                  const uint16_t phase_index, const uint16_t action_index,
                  const uint16_t next_action_index, const RecordKind kind,
                  const ExecutionResult result) noexcept {
    if (records_.size() >= maximum_records_) {
      ++dropped_count_;
      return;
    }
    try {
      records_.push_back(Record{start_ns, end_ns, component_id, chare_id,
                                action_id, sequence_++, phase, phase_index,
                                action_index, next_action_index,
                                static_cast<uint8_t>(kind),
                                static_cast<uint8_t>(result), 0, pe_, pid_});
    } catch (...) {
      ++dropped_count_;
    }
  }

  void register_component(const uint64_t id, std::string name) {
    component_names_.try_emplace(id, std::move(name));
  }

  void register_action(const uint64_t id, std::string name) {
    action_names_.try_emplace(id, std::move(name));
  }

  template <typename ArrayIndex>
  void register_chare(const uint64_t component_id_value,
                      const ArrayIndex& array_index) {
    const uint64_t id = chare_id(array_index);
    const uint64_t key = component_id_value ^ (id + 0x9e3779b97f4a7c15ULL +
                                               (component_id_value << 6U) +
                                               (component_id_value >> 2U));
    if (registered_chares_.insert(key).second) {
      chare_names_.push_back({component_id_value, id, chare_name(array_index)});
    }
  }

 private:
  struct ChareName {
    uint64_t component_id;
    uint64_t chare_id;
    std::string name;
  };

  std::string file_stem() const {
    std::ostringstream result{};
    result << output_directory_ << "/phase3_action_timing.pe" << std::setw(6)
           << std::setfill('0') << pe_ << ".pid" << pid_;
    return result.str();
  }

  void flush() noexcept {
    try {
      const std::string stem = file_stem();
      std::ofstream binary{stem + ".bin", std::ios::binary};
      const FileHeader header{{'S', 'P', '3', 'T', 'I', 'M', 'E', '1'},
                              1,
                              sizeof(FileHeader),
                              sizeof(Record),
                              static_cast<uint32_t>(pe_),
                              pid_,
                              0,
                              records_.size(),
                              dropped_count_,
                              std::chrono::steady_clock::period::num,
                              std::chrono::steady_clock::period::den};
      binary.write(reinterpret_cast<const char*>(&header), sizeof(header));
      if (not records_.empty()) {
        binary.write(
            reinterpret_cast<const char*>(records_.data()),
            static_cast<std::streamsize>(records_.size() * sizeof(Record)));
      }
      binary.close();

      std::ofstream names{stem + ".names.tsv"};
      names << "kind\tid\tname\n";
      for (const auto& [id, name] : component_names_) {
        names << "component\t" << id << '\t' << name << '\n';
      }
      for (const auto& [id, name] : action_names_) {
        names << "action\t" << id << '\t' << name << '\n';
      }
      names.close();

      std::ofstream chares{stem + ".chares.tsv"};
      chares << "component_id\tchare_id\tchare_name\n";
      for (const auto& chare : chare_names_) {
        chares << chare.component_id << '\t' << chare.chare_id << '\t'
               << chare.name << '\n';
      }
    } catch (...) {
      // This is diagnostic-only shutdown code. Never interfere with the
      // executable's normal exit if writing the profiling records fails.
    }
  }

  int32_t pe_;
  uint32_t pid_;
  std::string output_directory_;
  size_t maximum_records_;
  std::vector<Record> records_{};
  uint64_t dropped_count_{0};
  uint32_t sequence_{0};
  std::unordered_map<uint64_t, std::string> component_names_{};
  std::unordered_map<uint64_t, std::string> action_names_{};
  std::unordered_set<uint64_t> registered_chares_{};
  std::vector<ChareName> chare_names_{};
};

inline Recorder& recorder() {
  static Recorder instance{};
  return instance;
}

template <typename ParallelComponent>
uint64_t component_id() {
  static const uint64_t id = []() {
    std::string name = pretty_type::name<ParallelComponent>();
    const uint64_t result = fnv1a(name);
    recorder().register_component(result, std::move(name));
    return result;
  }();
  return id;
}

template <typename Action>
uint64_t action_id() {
  static const uint64_t id = []() {
    std::string name = pretty_type::name<Action>();
    const uint64_t result = fnv1a(name);
    recorder().register_action(result, std::move(name));
    return result;
  }();
  return id;
}

}  // namespace Parallel::phase3_action_timing
