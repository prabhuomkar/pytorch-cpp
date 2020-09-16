#!/usr/bin/env bash
set -Eeo pipefail

cmake -B build -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_PREFIX_PATH=/opt/conda/lib/python${PYTHON_VERSION}/site-packages/torch/share/cmake/Torch/ \
    -D CREATE_SCRIPTMODULES=ON

case $1 in
    basics|intermediate|advanced|popular)
        cmake --build build --target $1
        cd build/tutorials/$1
        exec bash
        ;;
    "")
        cmake --build build
        cd build/tutorials
        exec bash
        ;;
    *)
        tutorial_build_dir=$(find build/tutorials -maxdepth 3 -mindepth 2 -type d -name $(echo $1 | tr - _))

        if [ ! -z "$tutorial_build_dir" ]
        then
            cmake --build build --target $1
            cd $tutorial_build_dir
            ./$1 ${@:2}
        else
            cmake --build build
            cd build/tutorials
            exec bash
        fi
        ;;
esac
