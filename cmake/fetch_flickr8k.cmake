cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

function(fetch_flickr8k DATA_DIR)
    set(FLICKR8K_DIR "${DATA_DIR}/flickr_8k")
    set(FLICKR8K_DOWNLOAD_DIR "${FLICKR8K_DIR}/download")

    set(FLICKR8K_DATA_URL
        "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    )
    set(FLICKR8K_TEXT_URL
        "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    )

    set(FLICKR8K_DATA_SOURCE_DIR "${FLICKR8K_DIR}/Flickr8k_Dataset")
    set(FLICKR8K_TEXT_SOURCE_DIR "${FLICKR8K_DIR}/Flickr8k_text")

    if(NOT EXISTS ${FLICKR8K_TEXT_SOURCE_DIR} OR NOT EXISTS
                                                 ${FLICKR8K_DATA_SOURCE_DIR})
        message(STATUS "Fetching Flickr8k dataset...")

        file(
            DOWNLOAD ${FLICKR8K_TEXT_URL}
            "${FLICKR8K_DOWNLOAD_DIR}/Flickr8k_text.zip"
            EXPECTED_MD5 "bf6c1abcb8e4a833b7f922104de18627")

        file(MAKE_DIRECTORY ${FLICKR8K_TEXT_SOURCE_DIR})

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xf
                    "${FLICKR8K_DOWNLOAD_DIR}/Flickr8k_text.zip"
            WORKING_DIRECTORY ${FLICKR8K_TEXT_SOURCE_DIR})

        file(
            DOWNLOAD ${FLICKR8K_DATA_URL}
            "${FLICKR8K_DOWNLOAD_DIR}/Flickr8k_Dataset.zip"
            EXPECTED_MD5 "f18a1e2920de5bd84dae7cf08ec78978"
            SHOW_PROGRESS)

        file(MAKE_DIRECTORY ${FLICKR8K_DATA_SOURCE_DIR})

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xf
                    "${FLICKR8K_DOWNLOAD_DIR}/Flickr8k_Dataset.zip"
            WORKING_DIRECTORY ${FLICKR8K_DATA_SOURCE_DIR})

        file(REMOVE_RECURSE ${FLICKR8K_DOWNLOAD_DIR})

        message(STATUS "Fetching Flickr8k dataset - done")
    endif()
endfunction()

fetch_flickr8k(${DATA_DIR})
