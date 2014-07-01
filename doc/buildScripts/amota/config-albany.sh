#!/bin/bash

if [ -f ./CMakeCache.txt ]; then
    rm CMakeCache.txt
fi

# The Trilinos Dir is the same as the PREFIX entry from the
# Trilinos configuration script

cmake \
 -D CMAKE_CXX_FLAGS:STRING="cmake_cxx_flags" \
 -D CMAKE_BUILD_TYPE:STRING="build_type" \
 -D ALBANY_TRILINOS_DIR:FILEPATH=install_dir \
 -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
 -D ENABLE_LCM:BOOL=ON \
 -D ENABLE_QCAD:BOOL=ON \
 -D ENABLE_MOR:BOOL=ON \
 -D ENABLE_SG_MP:BOOL=ON \
 -D ENABLE_FELIX:BOOL=ON \
 -D ENABLE_AERAS:BOOL=ON \
 -D ENABLE_HYDRIDE:BOOL=ON \
 -D ENABLE_ASCR:BOOL=ON \
 -D ENABLE_LAME:BOOL=OFF \
 -D ENABLE_LAMENT:BOOL=OFF \
 -D ENABLE_CHECK_FPE:BOOL=ON \
..
