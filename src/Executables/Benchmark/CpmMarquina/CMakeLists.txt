# Distributed under the MIT License.
# See LICENSE.txt for details.

# Microbenchmark comparing the full-decomposition Marquina flux against the
# Complementary Projection Method (`MarquinaCpm`) for the GRMHD
# ValenciaDivClean system. Uses Google Benchmark for the timing loop, so
# only builds in Release when GoogleBenchmark is available.
if("${GoogleBenchmark_FOUND}" AND NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(executable BenchmarkCpmMarquina)

  add_spectre_executable(
    ${executable}
    EXCLUDE_FROM_ALL
    CpmMarquina.cpp
    )

  target_link_libraries(
    ${executable}
    PRIVATE
    DataStructures
    GeneralRelativity
    GoogleBenchmark
    Hydro
    Informer
    Spectral
    Utilities
    ValenciaDivClean
    )
endif()
