cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)    

set(FLICKR8K_DIR "${CMAKE_SOURCE_DIR}/data/flickr_8k")

set(FLICKR8K_DATA_URL "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip")
set(FLICKR8K_TEXT_URL "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip")

set(FLICKR8K_DATA_SOURCE_DIR "${FLICKR8K_DIR}/Flickr8k_Dataset")
set(FLICKR8K_TEXT_SOURCE_DIR "${FLICKR8K_DIR}/Flickr8k_text")

if(NOT EXISTS ${FLICKR8K_TEXT_SOURCE_DIR} OR NOT EXISTS ${FLICKR8K_DATA_SOURCE_DIR})
    message(STATUS "Fetching Flickr8k dataset...")
    
    FetchContent_Declare(
        flickr_8k_text
        DOWNLOAD_DIR ${FLICKR8K_DIR}/download
        SOURCE_DIR ${FLICKR8K_TEXT_SOURCE_DIR}
        URL ${FLICKR8K_TEXT_URL}
    )

    FetchContent_MakeAvailable(flickr_8k_text)
    
    set(FETCHCONTENT_QUIET OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        flickr_8k_data
        DOWNLOAD_DIR ${FLICKR8K_DIR}/download
        SOURCE_DIR ${FLICKR8K_DATA_SOURCE_DIR}
        URL ${FLICKR8K_DATA_URL}
    )

    FetchContent_MakeAvailable(flickr_8k_data)

    set(FETCHCONTENT_QUIET ON CACHE BOOL "" FORCE)

    file(REMOVE_RECURSE "${FLICKR8K_DIR}/download")
    
    message(STATUS "Fetching Flickr8k dataset - done")
else()
    message(STATUS "Flickr8k dataset already present, skipping...")
endif()
