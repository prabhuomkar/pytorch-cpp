#!/bin/bash

function install() {
	wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
	unzip libtorch-shared-with-deps-latest.zip
  rm -rf libtorch-shared-with-deps-latest.zip
}

function build() {
	rm -rf build
	mkdir build
	cd build
	cmake -DCMAKE_PREFIX_PATH=$(dirname $(pwd))/libtorch ..
	make
}

if [ $1 = "install" ]
then
	install
else
	build
fi