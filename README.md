<h1 align="center">
    <img src="images/pytorch_cpp.png" width="50%">
</h1>
<p align="center">
    C++ Implementation of PyTorch Tutorials for Deep Learning Researchers
    <br />
<img src="https://img.shields.io/travis/prabhuomkar/pytorch-cpp">
<img src="https://img.shields.io/github/license/prabhuomkar/pytorch-cpp">
<img src="https://img.shields.io/badge/libtorch-1.6.0-ee4c2c">
<img src="https://img.shields.io/badge/cmake-3.14-064f8d">
</p>


| OS (Compiler)\\LibTorch |                                                  1.6.0                                                  |  nightly |
| :---------------------: | :---------------------------------------------------------------------------------------------------: |  :-----: |
|    macOS (clang 9.1)    | ![Status](https://travis-matrix-badges.herokuapp.com/repos/prabhuomkar/pytorch-cpp/branches/master/1) |          |
|    macOS (clang 10.0)   | ![Status](https://travis-matrix-badges.herokuapp.com/repos/prabhuomkar/pytorch-cpp/branches/master/2) |          |
|    macOS (clang 11.0)   | ![Status](https://travis-matrix-badges.herokuapp.com/repos/prabhuomkar/pytorch-cpp/branches/master/3) |          |
|      Linux (gcc 5)      | ![Status](https://travis-matrix-badges.herokuapp.com/repos/prabhuomkar/pytorch-cpp/branches/master/4) |          |
|      Linux (gcc 6)      | ![Status](https://travis-matrix-badges.herokuapp.com/repos/prabhuomkar/pytorch-cpp/branches/master/5) |          |
|      Linux (gcc 7)      | ![Status](https://travis-matrix-badges.herokuapp.com/repos/prabhuomkar/pytorch-cpp/branches/master/6) |          |
|      Linux (gcc 8)      | ![Status](https://travis-matrix-badges.herokuapp.com/repos/prabhuomkar/pytorch-cpp/branches/master/7) |          |
|    Windows (msvc 2017)  | ![Status](https://travis-matrix-badges.herokuapp.com/repos/prabhuomkar/pytorch-cpp/branches/master/8) |          |

This repository provides tutorial code in C++ for deep learning researchers to learn PyTorch.  
**Python Tutorial**: [https://github.com/yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)

## Table of Contents

### 1. Basics
* [PyTorch Basics](tutorials/basics/pytorch_basics/main.cpp)
* [Linear Regression](tutorials/basics/linear_regression/main.cpp)
* [Logistic Regression](tutorials/basics/logistic_regression/main.cpp)
* [Feedforward Neural Network](tutorials/basics/feedforward_neural_network/src/main.cpp)

### 2. Intermediate
* [Convolutional Neural Network](tutorials/intermediate/convolutional_neural_network/src/main.cpp)
* [Deep Residual Network](tutorials/intermediate/deep_residual_network/src/main.cpp)
* [Recurrent Neural Network](tutorials/intermediate/recurrent_neural_network/src/main.cpp)
* [Bidirectional Recurrent Neural Network](tutorials/intermediate/bidirectional_recurrent_neural_network/src/main.cpp)
* [Language Model (RNN-LM)](tutorials/intermediate/language_model/src/main.cpp)

### 3. Advanced
* [Generative Adversarial Networks](tutorials/advanced/generative_adversarial_network/main.cpp)
* [Variational Auto-Encoder](tutorials/advanced/variational_autoencoder/src/main.cpp)
* [Neural Style Transfer](tutorials/advanced/neural_style_transfer/src/main.cpp)
* [Image Captioning (CNN-AttentionRNN)](tutorials/advanced/image_captioning/src/main.cpp)

### 4. Interactive Tutorials
* [Tensor Slicing](notebooks/tensor_slicing.ipynb)

### 5. Other Popular Tutorials
* [Deep Learning with PyTorch: A 60 Minute Blitz](tutorials/popular/blitz)

# Getting Started

## Requirements

1. [C++](http://www.cplusplus.com/doc/tutorial/introduction/)
2. [CMake](https://cmake.org/download/)
3. [LibTorch v1.6.0](https://pytorch.org/cppdocs/installing.html)
4. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)


## For Interactive Tutorials

**Note**: Interactive Tutorials are currently running on **LibTorch Nightly Version**.  
So there are some tutorials which can break when working with _nightly version_.

```bash
conda create --name pytorch-cpp
conda activate pytorch-cpp
conda install xeus-cling notebook -c conda-forge
```
## Clone, build and run tutorials
### In Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prabhuomkar/pytorch-cpp/blob/master/notebooks/pytorch_cpp_colab_notebook.ipynb)

### On Local Machine

```bash
git clone https://github.com/prabhuomkar/pytorch-cpp.git
cd pytorch-cpp
```

#### Generate build system

```bash
cmake -B build #<options>
```
> **_Note for Windows users:_**<br> 
> Libtorch only supports 64bit Windows and an x64 generator needs to be specified. For Visual Studio this can be done by appending `-A x64` to the above command.

Some useful options:

| Option       | Default           | Description  |
| :------------- |:------------|-----:|
| `-D CUDA_V=(9.2 [Linux only]\|10.1\|10.2\|none)`     | `none` | Download LibTorch for a CUDA version (`none` = download CPU version). |
| `-D DOWNLOAD_DATASETS=(OFF\|ON)`     | `ON`      |   Download all datasets used in the tutorials. |
| `-D CMAKE_PREFIX_PATH=path/to/libtorch/share/cmake/Torch` |   `<empty>`    |    Skip the downloading of LibTorch and use your own local version (see [Requirements](#requirements)) instead. |
| `-D CMAKE_BUILD_TYPE=(Release\|Debug)` | `<empty>` (`Release` when downloading LibTorch on Windows) | Set the build type (`Release` = compile with optimizations).|
|`-D CREATE_SCRIPTMODULES=(OFF\|ON)` | `OFF` | Create all needed scriptmodule files for prelearned models / weights. Requires installed  python3 with  pytorch and torchvision. |

<details>
<summary><b>Example Linux</b></summary>

##### Aim
* Use existing Python, PyTorch (see [Requirements](#requirements)) and torchvision installation.
* Download all datasets and create all necessary scriptmodule files.

##### Command
```bash
cmake -B build \
-D CMAKE_BUILD_TYPE=Release \
-D CMAKE_PREFIX_PATH=/path/to/libtorch/share/cmake/Torch \
-D CREATE_SCRIPTMODULES=ON 
```
</details>

<details>
<summary><b>Example Windows</b></summary>

##### Aim
* Automatically download LibTorch for CUDA 10.2 and all necessary datasets.
* Do not create scriptmodule files.

##### Command
```bash
cmake -B build \
-A x64 \
-D CUDA_V=10.2
```
</details>

#### Build

```bash
cmake --build build
```
>**_Note for Windows users:_** <br>
>The CMake script downloads the *Release* version of LibTorch, so `--config Release` has to be appended to the build command.
>
>**_General Note:_** <br>
>By default all tutorials will be built. If you only want to build  one specific tutorial, specify the `target` parameter for the build command. For example to only build the language model tutorial, append `--target language-model` (target name = tutorial foldername with all underscores replaced with hyphens).

#### Run Tutorials
1. (**IMPORTANT!**) First change into the tutorial's directory within `build/tutorials`. For example, assuming you are in the `pytorch-cpp` directory and want to change to the pytorch basics tutorial folder:
     ```bash
     cd build/tutorials/basics/pytorch_basics
     # In general: cd build/tutorials/{basics|intermediate|advanced}/{tutorial_name}
     ```
2. Run the executable. Note that the executable's name is the tutorial's foldername with all underscores replaced with hyphens (e.g. for tutorial folder: `pytorch_basics` -> executable name: `pytorch-basics` (or `pytorch-basics.exe` on Windows)). For example, to run the pytorch basics tutorial:<br><br>
     **Linux/Mac**
     ```bash
     ./pytorch-basics
     # In general: ./{tutorial-name}
     ```
     **Windows**
     ```powershell
     .\pytorch-basics.exe
     # In general: .\{tutorial-name}.exe
     ```

### Using Docker

Find the latest and previous version images on [Docker Hub](https://hub.docker.com/repository/docker/prabhuomkar/pytorch-cpp).

You can build and run the tutorials (on CPU) in a Docker container using the provided `Dockerfile` and `docker-compose.yml` files:  
1. From the root directory of the cloned repo build the image:
    ```bash
    docker-compose build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
    ```
    > **_Note_**:  
    > When you run the Docker container, the host repo directory is mounted as a volume in the Docker container in order to cache build and downloaded dependency files so that it is not necessary to rebuild or redownload everything when a container is restarted. In order to have correct file permissions it is necessary to provide your user and group ids as build arguments when building the image on Linux.
2. Now start the container and build the tutorials using:
    ```bash
    docker-compose run --rm pytorch-cpp
    ```
    This fetches all necessary dependencies and builds the tutorials. After the build is done, by default the container starts `bash` in interactive mode in the `build/tutorials` folder. 
    As an alternative, you can also directly run a tutorial by instead invoking the above command with the tutorial as additional argument, for example:
    ```bash
    docker-compose run --rm pytorch-cpp pytorch-basics
    # In general: docker-compose run --rm pytorch-cpp {tutorial-name} 
    ```
    This will - if necessary - build all tutorials and then start the provided tutorial in a container.

## License
This repository is licensed under MIT as given in [LICENSE](LICENSE).
