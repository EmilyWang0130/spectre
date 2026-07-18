// Distributed under the MIT License.
// See LICENSE.txt for details.
//
// Throughput benchmark of the full-decomposition SpECTRE Marquina flux kernel
// (grmhd::ValenciaDivClean::BoundaryCorrections::Marquina) over N = 10^6
// random hydro-only interface state pairs. See bench_marquina_common.hpp for
// the shared harness and configuration.
//
// The 6 boundary-correction components per interface are written to
// <outdir>/spectre_output.dat and, on first run, to
// <outdir>/reference_output.dat. On later runs the fresh output is compared
// against the reference (rel 1e-12, abs 1e-14); on divergence the program
// prints the worst offender and exits non-zero. This is the bit-exactness
// regression guard for the full Marquina path.

#include <charm++.h>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"

#include "bench_marquina_common.hpp"

// Charm looks for this symbol; we build without a main module.
extern "C" void CkRegisterMainModule(void) {}

int main(int argc, char** argv) {
  namespace bm = bench_marquina;
  const std::string outdir = (argc > 1) ? std::string(argv[1]) : "bench";
  const std::string spectre_out = outdir + "/spectre_output.dat";
  const std::string reference = outdir + "/reference_output.dat";

  const EquationsOfState::IdealFluid<true> base_eos{bm::kGamma, 0.0};
  const std::unique_ptr<EquationsOfState::EquationOfState<true, 3>> eos =
      base_eos.promote_to_3d_eos();

  std::mt19937 gen(20240716);
  bm::PrimitiveStore interior_prim;
  bm::PrimitiveStore exterior_prim;
  bm::generate(&interior_prim, &gen, bm::kNumInterfaces);
  bm::generate(&exterior_prim, &gen, bm::kNumInterfaces);

  std::vector<double> results;
  const double ns_per_interface = bm::run_benchmark<bm::Marquina>(
      interior_prim, exterior_prim, *eos, &results);
  const double gflops = bm::kApproxFlopsPerInterface / ns_per_interface;

  std::printf("SPECTRE N            = %zu\n", bm::kNumInterfaces);
  std::printf("SPECTRE total time   = %.4f s\n",
              ns_per_interface * bm::kNumInterfaces * 1.0e-9);
  std::printf("SPECTRE ns/interface = %.4f\n", ns_per_interface);
  std::printf("SPECTRE GFlop/s (approx) = %.4f\n", gflops);
  std::printf("SPECTRE_NS_PER_CALL %.4f\n", ns_per_interface);

  bm::write_binary(results, spectre_out);

  std::FILE* ref = std::fopen(reference.c_str(), "rb");
  if (ref == nullptr) {
    bm::write_binary(results, reference);
    std::printf("SPECTRE: wrote reference_output.dat (first run).\n");
  } else {
    std::fclose(ref);
    if (not bm::compare_with_reference(results, reference, 1.0e-12, 1.0e-14,
                                       "SPECTRE")) {
      std::fprintf(stderr, "SPECTRE: OUTPUT VERIFICATION FAILED.\n");
      return 2;
    }
    std::printf("SPECTRE: output matches reference.\n");
  }
  return 0;
}
