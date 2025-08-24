cmake_minimum_required(VERSION 3.28.6 FATAL_ERROR)

include(FetchContent)

set(CUDA_V "none" CACHE STRING "Determines libtorch CUDA version to download (11.8, 12.4, 12.6, 12.8, 12.9 or none).")

if(${CUDA_V} STREQUAL "none")
    set(LIBTORCH_DEVICE "cpu")
elseif(${CUDA_V} STREQUAL "11.8")
    set(LIBTORCH_DEVICE "cu118")
elseif(${CUDA_V} STREQUAL "12.4")
    set(LIBTORCH_DEVICE "cu124")
elseif(${CUDA_V} STREQUAL "12.6")
    set(LIBTORCH_DEVICE "cu126")
elseif(${CUDA_V} STREQUAL "12.8")
    set(LIBTORCH_DEVICE "cu126")
elseif(${CUDA_V} STREQUAL "12.9")
    set(LIBTORCH_DEVICE "cu126")
else() 
    message(FATAL_ERROR "Invalid CUDA version specified, must be 11.8, 12.4, 12.6, 12.8, 12.9 or none!")
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(LIBTORCH_DOWNLOAD_BUILD_TYPE "Release" CACHE STRING "Determines whether to download Release (default) or Debug libtorch version.")

    if(${LIBTORCH_DOWNLOAD_BUILD_TYPE} STREQUAL "Debug")
        set(LIBTORCH_DOWNLOAD_BUILD_TYPE_TAG "debug-")
    elseif(${LIBTORCH_DOWNLOAD_BUILD_TYPE} STREQUAL "Release")
        set(LIBTORCH_DOWNLOAD_BUILD_TYPE_TAG "")
    else()
        message(FATAL_ERROR "Invalid libtorch build type, must be either Release or Debug.")
    endif()

    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/${LIBTORCH_DEVICE}/libtorch-win-shared-with-deps-${LIBTORCH_DOWNLOAD_BUILD_TYPE_TAG}${PYTORCH_VERSION}%2B${LIBTORCH_DEVICE}.zip")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/${LIBTORCH_DEVICE}/libtorch-shared-with-deps-${PYTORCH_VERSION}%2B${LIBTORCH_DEVICE}.zip")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    if(NOT ${LIBTORCH_DEVICE} STREQUAL "cpu")
        message(WARNING "MacOS binaries do not support CUDA, will download CPU version instead.")
        set(LIBTORCH_DEVICE "cpu")
    endif()
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${PYTORCH_VERSION}.zip")
else()
    message(FATAL_ERROR "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Windows', 'Linux' or 'Darwin')")
endif()

if(${LIBTORCH_DEVICE} STREQUAL "cpu")
    message(STATUS "Downloading libtorch version ${PYTORCH_VERSION} for CPU on ${CMAKE_SYSTEM_NAME} from ${LIBTORCH_URL}...")
else()
    message(STATUS "Downloading libtorch version ${PYTORCH_VERSION} for CUDA ${CUDA_V} on ${CMAKE_SYSTEM_NAME} from ${LIBTORCH_URL}...")
endif()

FetchContent_Declare(
    libtorch
    PREFIX libtorch
    DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/libtorch
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/libtorch
    URL ${LIBTORCH_URL}
)

FetchContent_MakeAvailable(libtorch)

message(STATUS "Downloading libtorch - done")

find_package(Torch REQUIRED PATHS "${CMAKE_SOURCE_DIR}/libtorch")
