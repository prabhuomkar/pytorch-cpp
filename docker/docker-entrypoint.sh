#!/usr/bin/env bash
set -Eeo pipefail

cmake -B build -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_PREFIX_PATH=/opt/conda/lib/python${PYTHON_VERSION}/site-packages/torch/share/cmake/Torch/ \
    -D CREATE_SCRIPTMODULES=ON
cmake --build build
cd build/tutorials

tutorial_path=$(find . -maxdepth 2 -mindepth 2 -type d -name $(echo $1 | tr - _))

if [ ! -z "$tutorial_path" ]
then
    cd $tutorial_path
    ./$(echo $(basename $tutorial_path) | tr _ -) ${@:2}
else
    exec $@
fi
