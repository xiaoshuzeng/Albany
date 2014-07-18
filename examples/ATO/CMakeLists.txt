##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# ATO ##################
# file(COPY input_exodus DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/default_piro_params.xml
#                ${CMAKE_CURRENT_BINARY_DIR}/default_piro_params.xml COPYONLY)

add_subdirectory(Elasticity)
add_subdirectory(ATOLinearElasticity)

