#!/bin/bash                                           

#rm -rf repos
rm -rf build
rm -rf ctest_nightly.cmake.work
rm -rf nightly_log*
rm -rf results*
rm -rf slurm* 
rm -rf modules*out 

ulimit -c 0

cat trilinos ctest_nightly.cmake.frag >& ctest_nightly.cmake  
bash -c -l "source ride_modules_cuda.sh >& modules_trilinos.out; bash nightly_cron_script_trilinos_ride.sh"
cat albany ctest_nightly.cmake.frag >& ctest_nightly.cmake
bash -c -l "source ride_modules_cuda.sh >& modules_albany.out; bash nightly_cron_script_albany_ride.sh"
bash process_results_ctest.sh 
bash send_email_ctest.sh  

