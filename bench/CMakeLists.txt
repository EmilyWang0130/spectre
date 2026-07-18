# Distributed under the MIT License.
# See LICENSE.txt for details.

# Standalone Marquina flux-kernel benchmarks. Wired into the SpECTRE build by
# an add_subdirectory() from src/Executables/Benchmark/CMakeLists.txt so the
# C++ targets can reuse add_spectre_executable and link the ValenciaDivClean
# libraries directly. All targets are EXCLUDE_FROM_ALL; build them explicitly
# with
# `cmake --build build --target bench_marquina bench_marquina_cpm bench_whisky`.

# Only meaningful in an optimized build.
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message(STATUS "bench/: skipping Marquina benchmarks in Debug build.")
  return()
endif()

# --- SpECTRE C++ kernel benchmarks --------------------------------------
foreach(executable bench_marquina bench_marquina_cpm)
  add_spectre_executable(
    ${executable}
    EXCLUDE_FROM_ALL
    ${executable}.cpp
    )
  target_link_libraries(
    ${executable}
    PRIVATE
    DataStructures
    GeneralRelativity
    Hydro
    Informer
    Spectral
    Utilities
    ValenciaDivClean
    )
  target_compile_options(${executable} PRIVATE -O3 -march=native -DNDEBUG)
endforeach()

# --- WHISKY reference kernel benchmark (self-contained Fortran) ----------
add_executable(bench_whisky EXCLUDE_FROM_ALL bench_whisky.F90)
set_target_properties(
  bench_whisky
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LINKER_LANGUAGE Fortran
  )
target_compile_options(bench_whisky PRIVATE -O3 -march=native -DNDEBUG)
