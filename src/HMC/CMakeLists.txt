##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
# HYDRIDE

# problems
SET(SOURCES
  evaluators/HMC_Stresses.cpp
  evaluators/HMC_StrainDifference.cpp
  evaluators/HMC_TotalStress.cpp
  evaluators/HMC_Residual.cpp
  evaluators/PHAL_DOFTensorGradInterpolation.cpp
  evaluators/PHAL_DOFTensorInterpolation.cpp
  evaluators/HMC_ScatterResidual.cpp
  problems/HMCProblem.cpp
  problems/HMC_EvaluatorUtils.cpp
  )
SET(HEADERS
  evaluators/HMC_Stresses.hpp
  evaluators/HMC_Stresses_Def.hpp
  evaluators/HMC_StrainDifference.hpp
  evaluators/HMC_StrainDifference_Def.hpp
  evaluators/HMC_TotalStress.hpp
  evaluators/HMC_TotalStress_Def.hpp
  evaluators/HMC_Residual.hpp
  evaluators/HMC_Residual_Def.hpp
  evaluators/PHAL_DOFTensorGradInterpolation_Def.hpp
  evaluators/PHAL_DOFTensorGradInterpolation.hpp
  evaluators/PHAL_DOFTensorInterpolation_Def.hpp
  evaluators/PHAL_DOFTensorInterpolation.hpp
  evaluators/HMC_ScatterResidual.hpp
  evaluators/HMC_ScatterResidual_Def.hpp
  problems/HMCProblem.hpp
  problems/HMC_EvaluatorUtils.hpp
  problems/HMC_EvaluatorUtils_Def.hpp
  )


# evaluators
#SET(SOURCES ${SOURCES}
#   )

#SET(HEADERS ${HEADERS}
#   )


include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators
  ${Albany_SOURCE_DIR}/src/problems ${Albany_SOURCE_DIR}/src/responses
  ${Albany_SOURCE_DIR}/src/LCM/
  ${Albany_SOURCE_DIR}/src/LCM/evaluators 
  ${Albany_SOURCE_DIR}/src/HMC/evaluators 
  ${Albany_SOURCE_DIR}/src/LCM/utils
  ./problems
  )

add_library(HMC ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

IF ( INSTALL_ALBANY )
    install(TARGETS HMC EXPORT Albany
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
