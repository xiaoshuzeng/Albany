#!/bin/bash

module unload cmake netcdf-hdf5parallel python
module load PrgEnv-gnu; module load cmake/2.8.9 python netcdf-hdf5parallel/4.3.0 usg-default-modules/1.1
module load boost
module list
