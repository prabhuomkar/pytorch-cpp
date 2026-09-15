cmake_minimum_required(VERSION 3.28.6 FATAL_ERROR)

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

        # Extract into the download dir and move into place only on success,
        # so an interrupted extraction is retried on the next build.
        file(REMOVE_RECURSE ${FLICKR8K_TEXT_SOURCE_DIR} ${FLICKR8K_DATA_SOURCE_DIR})
        file(MAKE_DIRECTORY "${FLICKR8K_DOWNLOAD_DIR}/text" "${FLICKR8K_DOWNLOAD_DIR}/data")

        file(
            DOWNLOAD ${FLICKR8K_TEXT_URL}
            "${FLICKR8K_DOWNLOAD_DIR}/Flickr8k_text.zip"
            EXPECTED_MD5 "bf6c1abcb8e4a833b7f922104de18627")

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xf
                    "${FLICKR8K_DOWNLOAD_DIR}/Flickr8k_text.zip"
            WORKING_DIRECTORY "${FLICKR8K_DOWNLOAD_DIR}/text"
            COMMAND_ERROR_IS_FATAL ANY)

        file(
            DOWNLOAD ${FLICKR8K_DATA_URL}
            "${FLICKR8K_DOWNLOAD_DIR}/Flickr8k_Dataset.zip"
            EXPECTED_MD5 "f18a1e2920de5bd84dae7cf08ec78978"
            SHOW_PROGRESS)

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xf
                    "${FLICKR8K_DOWNLOAD_DIR}/Flickr8k_Dataset.zip"
            WORKING_DIRECTORY "${FLICKR8K_DOWNLOAD_DIR}/data"
            COMMAND_ERROR_IS_FATAL ANY)

        file(RENAME "${FLICKR8K_DOWNLOAD_DIR}/text" ${FLICKR8K_TEXT_SOURCE_DIR})
        file(RENAME "${FLICKR8K_DOWNLOAD_DIR}/data" ${FLICKR8K_DATA_SOURCE_DIR})
        file(REMOVE_RECURSE ${FLICKR8K_DOWNLOAD_DIR})

        message(STATUS "Fetching Flickr8k dataset - done")
    endif()
endfunction()

fetch_flickr8k(${DATA_DIR})
