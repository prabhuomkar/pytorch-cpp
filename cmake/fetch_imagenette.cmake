cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

function(fetch_imagenette DATA_DIR)
    set(IMAGENETTE_DIR "${DATA_DIR}/imagenette2-160")
    set(IMAGENETTE_DOWNLOAD_DIR "${DATA_DIR}/imagenette_download")

    set(IMAGENETTE_DATA_URL
        "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    )

    if(NOT EXISTS ${IMAGENETTE_DIR})
        message(STATUS "Fetching Imagenette dataset...")

        file(
            DOWNLOAD ${IMAGENETTE_DATA_URL}
            "${IMAGENETTE_DOWNLOAD_DIR}/imagenette2-160.tgz"
            EXPECTED_MD5 "e793b78cc4c9e9a4ccc0c1155377a412"
            SHOW_PROGRESS)

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xf  
                    "${IMAGENETTE_DOWNLOAD_DIR}/imagenette2-160.tgz"
                    "imagenette2-160/train"
                    "imagenette2-160/val"
            WORKING_DIRECTORY ${DATA_DIR})

        file(REMOVE_RECURSE ${IMAGENETTE_DOWNLOAD_DIR})

        message(STATUS "Fetching Imagenette dataset - done")
    endif()
endfunction()

fetch_imagenette(${DATA_DIR})
