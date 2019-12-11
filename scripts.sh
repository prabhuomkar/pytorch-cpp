#!/bin/bash

function install() {
    PYTORCH_VERSION="1.3.1"

    case $1 in
        --cuda=*)
            cuda="${1#*=}"
            ;;
        *)
    esac
    
    cuda="${cuda:-None}"
    
    case $cuda in
        9.2|10.1|None)
        ;;
        *)
        echo "Invalid cuda version, expected \"9.2\", \"10.1\" or \"None\"(cpu) but got: \"${cuda}\""
        return -1
        ;;
    esac
    
    if [ $cuda = "None" ]; then
        echo "Downloading libtorch for cpu, build-version ${PYTORCH_VERSION}..."
        wget "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip"
        unzip "libtorch-shared-with-deps-${PYTORCH_VERSION}+cpu.zip"
        rm -rf "libtorch-shared-with-deps-${PYTORCH_VERSION}+cpu.zip"
    elif [ $cuda = "10.1" ]; then
        echo "Downloading libtorch for gpu (CUDA ${cuda}), build-version ${PYTORCH_VERSION}..."
        wget "https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-${PYTORCH_VERSION}.zip"
        unzip "libtorch-shared-with-deps-${PYTORCH_VERSION}.zip"
        rm -rf "libtorch-shared-with-deps-${PYTORCH_VERSION}.zip"
    elif [ $cuda = "9.2" ]; then
        echo "Downloading libtorch for gpu (CUDA ${cuda}), build-version ${PYTORCH_VERSION}..."
        wget "https://download.pytorch.org/libtorch/cu92/libtorch-shared-with-deps-${PYTORCH_VERSION}%2Bcu92.zip"
        unzip "libtorch-shared-with-deps-${PYTORCH_VERSION}+cu92.zip"
        rm -rf "libtorch-shared-with-deps-${PYTORCH_VERSION}+cu92.zip"
    fi      
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
        --filter=-build/include_subdir,-build/include_what_you_use \
        --exclude=tutorials/advanced/utils/include/external/* main.cpp tutorials/*/*/**
}

function lintci() {
    python cpplint.py --linelength=120 --recursive \
        --filter=-build/include_subdir,-build/include_what_you_use \
        --exclude=tutorials/advanced/utils/include/external/* main.cpp tutorials/*/*/**
}

function download_mnist() {
    echo "---MNIST---"
    mnist_dir="data/mnist"
    mnist_base_url="http://yann.lecun.com/exdb/mnist"
    mkdir -p $mnist_dir
    
    declare -a files
    files=( "train-images-idx3-ubyte" "train-labels-idx1-ubyte" \
    "t10k-images-idx3-ubyte" "t10k-labels-idx1-ubyte" )
    
    for file in "${files[@]}"
    do
        if [ -f "${mnist_dir}/${file}" ]; then
            echo "${mnist_dir}/${file} already exists, skipping..."
        else
            wget "${mnist_base_url}/${file}.gz" -P $mnist_dir 
            gunzip "${mnist_dir}/${file}.gz"
        fi
    done
}

function download_cifar10() {
    echo "---CIFAR10---"
    cifar_dir="data/cifar10"
    cifar_url="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    mkdir -p $cifar_dir
    
    declare -a files
    files=( "data_batch_1.bin" "data_batch_2.bin" "data_batch_3.bin" \
    "data_batch_4.bin" "data_batch_5.bin" "test_batch.bin" )
    
    files_already_exist=true
    
    for file in "${files[@]}"
    do
        if [ ! -f "${cifar_dir}/${file}" ]; then
            files_already_exist=false
            break
        fi
    done
    
    if [ "$files_already_exist" = true ]; then
        echo "Files already exist, skipping..."
    else
        wget $cifar_url -P $cifar_dir
        tar xf "${cifar_dir}/cifar-10-binary.tar.gz" -C $cifar_dir
        rm "${cifar_dir}/cifar-10-binary.tar.gz"
        mv "${cifar_dir}/cifar-10-batches-bin/"* "${cifar_dir}"
        rm -rf "${cifar_dir}/cifar-10-batches-bin"
    fi
}

function download_penntreebank() {
    echo "---Penn Treebank---"
    penntreebank_dir="data/penntreebank"
    mkdir -p $penntreebank_dir
    penntreebank_url="https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt"
    
    train_filename="train.txt"
    
    if [ -f "${penntreebank_dir}/${train_filename}" ]; then
        echo "${penntreebank_dir}/${train_filename} already exists, skipping..."
    else
        wget $penntreebank_url -O "${penntreebank_dir}/${train_filename}"
    fi
}

function download_datasets() {
    download_mnist
    echo
    download_cifar10
    echo
    download_penntreebank
}

if [ $1 = "install" ]
then
    install $2
elif [ $1 = "build" ]
then
    build
elif [ $1 = "lint" ]
then
    lint
elif [ $1 = "lintci" ]
then
    lintci
elif [ $1 = "download_datasets" ]
then
    download_datasets
fi