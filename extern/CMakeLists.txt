cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(extern VERSION 1.0.0 LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
    extern_cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v2.2.0
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cxxopts
)

set(CXXOPTS_CXX_STANDARD 14)
set(CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "Set to ON to build examples" FORCE)
set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "Set to ON to build tests" FORCE)
set(CXXOPTS_ENABLE_INSTALL OFF CACHE BOOL "Generate the install target" FORCE)

FetchContent_MakeAvailable(extern_cxxopts)

foreach(STB_IMAGE_LIB "stb_image" "stb_image_write" "stb_image_resize")
    FetchContent_Declare(
        extern_${STB_IMAGE_LIB}
        URL https://raw.githubusercontent.com/nothings/stb/master/${STB_IMAGE_LIB}.h
        DOWNLOAD_NO_EXTRACT ON  
        DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${STB_IMAGE_LIB}
    )
    FetchContent_MakeAvailable(extern_${STB_IMAGE_LIB})
    
    string(REPLACE "_" "-" LIB_NAME ${STB_IMAGE_LIB})

    add_library(${LIB_NAME} INTERFACE)
    target_sources(${LIB_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/${STB_IMAGE_LIB}/${STB_IMAGE_LIB}.h)
    target_include_directories(${LIB_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/${STB_IMAGE_LIB})
endforeach()