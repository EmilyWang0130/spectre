#
# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Hll.cpp
  Marquina.cpp
  MarquinaCpm.cpp
  Rusanov.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  Hll.hpp
  Marquina.hpp
  MarquinaCpm.hpp
  NamespaceDocs.hpp
  Rusanov.hpp
  )
