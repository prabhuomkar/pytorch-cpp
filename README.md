<h1 align="center">
    <img src="images/pytorch_cpp.jpg" width="50%">
</h1>
<p align="center">
    C++ Implementation of PyTorch Tutorials for Everyone
    <br />
<img src="https://img.shields.io/github/license/prabhuomkar/pytorch-cpp">
<img src="https://img.shields.io/badge/libtorch-2.8.0-ee4c2c">
<img src="https://img.shields.io/badge/cmake-3.28.6-064f8d">
</p>


| OS (Compiler)\\LibTorch |                                                  2.8.0                                                |
| :--------------------- | :--------------------------------------------------------------------------------------------------- |
|    macOS (clang 15, 16)    | [![Status](https://github.com/prabhuomkar/pytorch-cpp/actions/workflows/build_macos.yml/badge.svg?branch=master)](https://github.com/prabhuomkar/pytorch-cpp/actions?query=workflow%3Aci-build-macos) |
|      Linux (gcc 13, 14)      | [![Status](https://github.com/prabhuomkar/pytorch-cpp/actions/workflows/build_ubuntu.yml/badge.svg?branch=master)](https://github.com/prabhuomkar/pytorch-cpp/actions?query=workflow%3Aci-build-ubuntu) |
|    Windows (msvc 2022, 2025)  | [![Status](https://github.com/prabhuomkar/pytorch-cpp/actions/workflows/build_windows.yml/badge.svg?branch=master)](https://github.com/prabhuomkar/pytorch-cpp/actions?query=workflow%3Aci-build-windows) |

## Table of Contents

This repository provides tutorial code in C++ for deep learning researchers to learn PyTorch _(i.e. Section 1 to 3)_  
**Python Tutorial**: [https://github.com/yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)

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

1. [C++-17](http://www.cplusplus.com/doc/tutorial/introduction/) compatible compiler
2. [CMake](https://cmake.org/download/) (minimum version 3.28.6)
3. [LibTorch version >= 1.12.0 and <= 2.8.0](https://pytorch.org/cppdocs/installing.html)
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
| `-D CUDA_V=(11.8\|12.4\|12.6\|12.8\|12.9\|none)`     | `none` | Download LibTorch for a CUDA version (`none` = download CPU version). |
| `-D LIBTORCH_DOWNLOAD_BUILD_TYPE=(Release\|Debug)` | `Release` | Determines which libtorch build type version to download (only relevant on **Windows**).|
| `-D DOWNLOAD_DATASETS=(OFF\|ON)`     | `ON`      |   Download required datasets during build (only if they do not already exist in `pytorch-cpp/data`). |
|`-D CREATE_SCRIPTMODULES=(OFF\|ON)` | `OFF` | Create all required scriptmodule files for prelearned models / weights during build. Requires installed  python3 with  pytorch and torchvision. |
| `-D CMAKE_PREFIX_PATH=path/to/libtorch/share/cmake/Torch` |   `<empty>`    |    Skip the downloading of LibTorch and use your own local version (see [Requirements](#requirements)) instead. |
| `-D CMAKE_BUILD_TYPE=(Release\|Debug\|...)` | `<empty>` | Determines the CMake build-type for single-configuration generators (see [CMake docs](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)).|

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
* Automatically download LibTorch for CUDA 11.8 (Release version) and all necessary datasets.
* Do not create scriptmodule files.

##### Command
```bash
cmake -B build \
-A x64 \
-D CUDA_V=11.8
```
</details>

#### Build
>**_Note for Windows (Visual Studio) users:_** <br>
>The CMake script downloads the *Release* version of LibTorch, so `--config Release` has to be appended to the build command.

**How dataset download and scriptmodule creation work:**
* If `DOWNLOAD_DATASETS` is `ON`, the datasets required by the tutorials you choose to build will be downloaded to `pytorch-cpp/data` (if they do not already exist there).
* If `CREATE_SCRIPTMODULES` is `ON`, the scriptmodule files for the prelearned models / weights required by the tutorials you choose to build will be created in the `model` folder of the respective tutorial's source folder (if they do not already exist).
#### All tutorials
To build all tutorials use
```bash
cmake --build build
```

#### All tutorials in a category
You can choose to only build tutorials in one of the categories `basics`, `intermediate`, `advanced` or `popular`. For example, if you are only interested in the `basics` tutorials:
```bash
cmake --build build --target basics
# In general: cmake --build build --target {category}
```

#### Single tutorial
You can also choose to only build a single tutorial. For example to build the language model tutorial only: 
```bash
cmake --build build --target language-model
# In general: cmake --build build --target {tutorial-name}
```
>**_Note_**:  
> The target argument is the tutorial's foldername with all underscores replaced by hyphens.

>**_Tip for users of CMake version >= 3.15_**:  
> You can specify several targets separated by spaces, for example:  
> ```bash 
> cmake --build build --target language-model image-captioning
> ```

#### Run Tutorials
1. (**IMPORTANT!**) First change into the tutorial's directory within `build/tutorials`. For example, assuming you are in the `pytorch-cpp` directory and want to change to the pytorch basics tutorial folder:
     ```bash
     cd build/tutorials/basics/pytorch_basics
     # In general: cd build/tutorials/{basics|intermediate|advanced|popular/blitz}/{tutorial_name}
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
    This fetches all necessary dependencies and builds all tutorials.
    After the build is done, by default the container starts `bash` in interactive mode in the `build/tutorials` folder.  
    As with the local build, you can choose to only build tutorials of a category (`basics`, `intermediate`, `advanced`, `popular`):
    ```bash
    docker-compose run --rm pytorch-cpp {category}
    ```
    In this case the container is started in the chosen category's base build directory.  
    Alternatively, you can also directly run a tutorial by instead invoking the run command with a tutorial name as additional argument, for example:
    ```bash
    docker-compose run --rm pytorch-cpp pytorch-basics
    # In general: docker-compose run --rm pytorch-cpp {tutorial-name} 
    ```
    This will - if necessary - build the pytorch-basics tutorial and then start the executable in a container.

## License
This repository is licensed under MIT as given in [LICENSE](LICENSE).
