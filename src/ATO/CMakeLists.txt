##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

SET(SOURCES
  ${CMAKE_SOURCE_DIR}/src/ATO/ATO_Solver.cpp
  ${CMAKE_SOURCE_DIR}/src/ATO/evaluators/ATO_ComputeBasisFunctions.cpp
  ${CMAKE_SOURCE_DIR}/src/ATO/problems/ATO_EvaluatorUtils.cpp
  ${CMAKE_SOURCE_DIR}/src/ATO/problems/LinearElasticityProblem.cpp
)

SET(HEADERS
  ${CMAKE_SOURCE_DIR}/src/ATO/ATO_Solver.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/evaluators/ATO_ComputeBasisFunctions_Def.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/evaluators/ATO_ComputeBasisFunctions.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/problems/ATO_EvaluatorUtils_Def.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/problems/ATO_EvaluatorUtils.hpp
  ${CMAKE_SOURCE_DIR}/src/ATO/problems/LinearElasticityProblem.hpp
)

include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src 
  ${CMAKE_SOURCE_DIR}/src/evaluators
  ${CMAKE_SOURCE_DIR}/src/problems 
  ${CMAKE_SOURCE_DIR}/src/responses
  ${CMAKE_SOURCE_DIR}/src/LCM/
  ${CMAKE_SOURCE_DIR}/src/LCM/evaluators 
  ${CMAKE_SOURCE_DIR}/src/LCM/utils
  ${CMAKE_SOURCE_DIR}/src/ATO/
  ${CMAKE_SOURCE_DIR}/src/ATO/evaluators 
  ${CMAKE_SOURCE_DIR}/src/ATO/problems
)

add_library(ATO ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

IF ( INSTALL_ALBANY )
    install(TARGETS ATO EXPORT Albany
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()

