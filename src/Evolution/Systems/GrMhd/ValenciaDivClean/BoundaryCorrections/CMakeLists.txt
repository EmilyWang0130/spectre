#
# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  D2R/Flux.cpp
  D2R/FluxMp.cpp
  D2R/SlowModes.cpp
  D2R/SlowModesMp.cpp
  D2R/State.cpp
  D2R/StateMp.cpp
  Hll.cpp
  Hllc.cpp
  HllcGr.cpp
  Hlld.cpp
  Hllem.cpp
  HllemHydroYe.cpp
  Marquina.cpp
  PlutoD2R.cpp
  PlutoFaceFrame.cpp
  PlutoHllc.cpp
  PlutoHlld.cpp
  PlutoHllx.cpp
  Rusanov.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  D2R/Flux.hpp
  D2R/SlowModes.hpp
  D2R/State.hpp
  D2R/Types.hpp
  Factory.hpp
  HatTransform.hpp
  Hll.hpp
  Hllc.hpp
  HllcGr.hpp
  Hlld.hpp
  HlldImpl.hpp
  Hllem.hpp
  HllemHydroYe.hpp
  Marquina.hpp
  PlutoD2R.hpp
  PlutoFaceFrame.hpp
  PlutoHllc.hpp
  PlutoHlld.hpp
  PlutoHllx.hpp
  NamespaceDocs.hpp
  Rusanov.hpp
  )

# --- Vendored PLUTO v4.4 HLLD and HLLC (see pluto/README.md) ----------------
# PlutoHlld and PlutoHllc call PLUTO's own compiled solvers rather than
# re-implementations of them.
#
# These are C sources, and they must live in their OWN target rather than being
# added to ${LIBRARY}: SpECTRE precompiles a C++ header for that library, and
# CMake insists on a per-language PCH for every language a target uses, so
# adding C files makes it demand a C PCH ("cmake_pch.h") that does not exist.
# Per-source SKIP_PRECOMPILE_HEADERS does not avoid this. A separate static
# library with PCH disabled keeps the C and C++ toolchains cleanly apart.
set(PLUTO_HLLD_SOURCES
  pluto/arrays.c pluto/debug_tools.c pluto/eigenv.c pluto/eos.c
  pluto/fluxes.c pluto/hll_speed.c pluto/hllc_kb.c pluto/hllc_mb.c
  pluto/hlld.c pluto/hllx.c pluto/mappers.c
  pluto/math_misc.c pluto/math_qr_decomp.c pluto/math_root_finders.c
  pluto/output_log.c pluto/rmhd_energy_solve.c pluto/rmhd_pressure_fix.c
  pluto/set_indexes.c pluto/tools.c pluto/pluto_hlld_shim.c
  pluto/d2r_hlld_fixed.c pluto/d2r_sr_context.c pluto/spectre_d2r_core.c
  )

add_library(PlutoHlldC STATIC ${PLUTO_HLLD_SOURCES})

set_target_properties(
  PlutoHlldC PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  DISABLE_PRECOMPILE_HEADERS ON
  LINKER_LANGUAGE C
  )

# PLUTO resolves its headers relative to its own directory ("pluto.h",
# "States/plm_coeffs.h", ...). PUBLIC so PlutoHlld.cpp picks up the shim header.
target_include_directories(
  PlutoHlldC PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/pluto
  )

# Upstream PLUTO is not warning-clean under SpECTRE's flags. We do not patch
# third-party code to silence warnings; our own shim is small and reviewed.
target_compile_options(PlutoHlldC PRIVATE -w -ffp-contract=off)

# The D2-R correctness backend relies on represented affine flux arithmetic.
# Do not inherit reciprocal approximations from the default optimized flags.
# TARGET_DIRECTORY: the target is created in the parent directory, so without it
# these properties never applied (compile_commands.json showed the PCH and no
# -ffp-contract=off on the D2R sources; found in review, 2026-10-15).
set_source_files_properties(
  D2R/Flux.cpp D2R/FluxMp.cpp D2R/SlowModes.cpp D2R/SlowModesMp.cpp D2R/State.cpp
  D2R/StateMp.cpp
  TARGET_DIRECTORY ${LIBRARY}
  PROPERTIES
  COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off"
  SKIP_PRECOMPILE_HEADERS ON
  )

target_link_libraries(${LIBRARY} PRIVATE PlutoHlldC)
