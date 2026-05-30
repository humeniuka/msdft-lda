#!/bin/bash

# include clean up and comparison scripts in PATH.
export PATH=$(pwd):$PATH

set -e

for subtest in test-*/
do
    pushd ${subtest}
    clean.sh

    echo "Running msdft test ${subtest} ..."
    ./run.sh
    compare_output_with_reference.py
    
    clean.sh
    popd
done
