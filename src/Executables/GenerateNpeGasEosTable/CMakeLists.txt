# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE GenerateNpeGasEosTable)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  GenerateNpeGasEosTable.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Boost::boost
  Boost::program_options
  DataStructures
  H5
  Informer
  Printf
  )
