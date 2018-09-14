#!/bin/bash

# Fail at first error & echo commands before executing
set -ev
# Do not use "python run_tests.py" to run all tests, so that:
# - Each set/directory of tests are timed separately
# - Script fails after first broken set of tests
# (Could alternatively group sets of tests with && operator)
cd regression/

if [ "$MPI_CHOICE" == "openmpi" ]; then
    PATH=$TRAVIS_BUILD_DIR/openmpi/bin/:$PATH
    MPI_OPTS=--oversubscribe
    # Disable OpenMPI 3.1 vader CMA due to namespace permission issues on Travis CI / Docker containers
    export OMPI_MCA_btl_vader_single_copy_mechanism=none
else
    PATH=$TRAVIS_BUILD_DIR/mpich/bin/:$PATH
fi

# --silent option refers only to stdout of Makefile calls for condensed build logs
python3 run_tests.py pgen --config=--cxx=$TEMP_CXX --config=--cflag="$(./ci/set_warning_cflag.sh $TEMP_CXX)"
python3 run_tests.py mpi --config=--cxx=$TEMP_CXX --mpirun_opts=$MPI_OPTS --silent
python3 run_tests.py grav --config=--cxx=$TEMP_CXX --mpirun_opts=$MPI_OPTS --silent # requires FFTW library
python3 run_tests.py amr --config=--cxx=$TEMP_CXX --silent
python3 run_tests.py hydro --config=--cxx=$TEMP_CXX --silent
python3 run_tests.py outputs --config=--cxx=$TEMP_CXX --silent
python3 run_tests.py curvilinear --config=--cxx=$TEMP_CXX --silent
python3 run_tests.py gr --config=--cxx=$TEMP_CXX --silent
python3 run_tests.py sr --config=--cxx=$TEMP_CXX --silent
python3 run_tests.py shearingbox --config=--cxx=$TEMP_CXX --silent
python3 run_tests.py diffusion --config=--cxx=$TEMP_CXX --silent
python3 run_tests.py hydro4 --config=--cxx=$TEMP_CXX --silent
python3 run_tests.py symmetry --config=--cxx=$TEMP_CXX --silent

# mhd/ currently contains the longest set of tests. Command times-out after 10 m on Travis CI
# python3 run_tests.py mhd --config=--cxx=$TEMP_CXX --silent
