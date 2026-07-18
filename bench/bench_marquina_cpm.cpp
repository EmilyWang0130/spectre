// Distributed under the MIT License.
// See LICENSE.txt for details.
//
// Throughput benchmark of the SpECTRE Complementary Projection Method flux
// kernel (grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCpm) over
// N = 10^6 random hydro-only interface state pairs. See
// bench_marquina_common.hpp for the shared harness and configuration.
//
// MarquinaCpm is the SpECTRE analog of WHISKY-Marquina's FAST=TRUE path (the
// Aloy et al. 1999 CPC shortcut), so this is the apples-to-apples comparison
// against bench_whisky.
//
// Two correctness checks run every invocation:
//   1. Equivalence envelope: MarquinaCpm is compared against full Marquina on
//      the same states; abs and rel diffs must stay within (1e-12, 1e-12), the
//      documented CPM-vs-full envelope. On violation the program exits
//      non-zero, gating every CPM optimization.
//   2. Regression reference: the MarquinaCpm output is written to
//      <outdir>/cpm_output.dat and, on first run, cpm_reference.dat. Later
//      runs compare against it with a loose tolerance (the CPM math may be
//      reformulated between iterations, so this is informational, not a hard
//      bit-exact gate — the envelope check is the hard gate).

#include <charm++.h>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/MarquinaCpm.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"

#include "bench_marquina_common.hpp"

// Charm looks for this symbol; we build without a main module.
extern "C" void CkRegisterMainModule(void) {}

int main(int argc, char** argv) {
  namespace bm = bench_marquina;
  using MarquinaCpm = grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCpm;
  const std::string outdir = (argc > 1) ? std::string(argv[1]) : "bench";
  const std::string cpm_out = outdir + "/cpm_output.dat";
  const std::string cpm_reference = outdir + "/cpm_reference.dat";

  const EquationsOfState::IdealFluid<true> base_eos{bm::kGamma, 0.0};
  const std::unique_ptr<EquationsOfState::EquationOfState<true, 3>> eos =
      base_eos.promote_to_3d_eos();

  std::mt19937 gen(20240716);
  bm::PrimitiveStore interior_prim;
  bm::PrimitiveStore exterior_prim;
  bm::generate(&interior_prim, &gen, bm::kNumInterfaces);
  bm::generate(&exterior_prim, &gen, bm::kNumInterfaces);

  // Timed: the CPM kernel.
  std::vector<double> cpm_results;
  const double ns_per_interface = bm::run_benchmark<MarquinaCpm>(
      interior_prim, exterior_prim, *eos, &cpm_results);
  const double gflops = bm::kApproxFlopsPerInterface / ns_per_interface;

  std::printf("CPM N            = %zu\n", bm::kNumInterfaces);
  std::printf("CPM total time   = %.4f s\n",
              ns_per_interface * bm::kNumInterfaces * 1.0e-9);
  std::printf("CPM ns/interface = %.4f\n", ns_per_interface);
  std::printf("CPM GFlop/s (approx) = %.4f\n", gflops);
  std::printf("CPM_NS_PER_CALL %.4f\n", ns_per_interface);

  // Untimed: full Marquina on the same states, for the equivalence envelope.
  std::vector<double> full_results;
  bm::run_benchmark<bm::Marquina>(interior_prim, exterior_prim, *eos,
                                  &full_results);
  const bool envelope_ok = bm::check_envelope(cpm_results, full_results,
                                              1.0e-12, 1.0e-12, "CPM-vs-full");

  bm::write_binary(cpm_results, cpm_out);
  std::FILE* ref = std::fopen(cpm_reference.c_str(), "rb");
  if (ref == nullptr) {
    bm::write_binary(cpm_results, cpm_reference);
    std::printf("CPM: wrote cpm_reference.dat (first run).\n");
  } else {
    std::fclose(ref);
    // Informational drift check against the previous CPM output (loose: the
    // CPM math may be reformulated between iterations).
    bm::compare_with_reference(cpm_results, cpm_reference, 1.0e-9, 1.0e-11,
                               "CPM-drift");
  }

  if (not envelope_ok) {
    std::fprintf(stderr, "CPM: EQUIVALENCE ENVELOPE VIOLATED.\n");
    return 2;
  }
  std::printf("CPM: within equivalence envelope of full Marquina.\n");
  return 0;
}
