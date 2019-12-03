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

function lint() {
	cpplint --linelength=120 --recursive \
		--filter=-build/include_subdir,-build/include_what_you_use main.cpp tutorials/*/*/**
}

function lintci() {
	python cpplint.py --linelength=120 --recursive \
		--filter=-build/include_subdir,-build/include_what_you_use main.cpp tutorials/*/*/**
}

if [ $1 = "install" ]
then
	install
elif [ $1 = "build" ]
then
	build
elif [ $1 = "lint" ]
then
	lint
elif [ $1 = "lintci" ]
then
	lintci
fi