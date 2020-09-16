cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

function(fetch_neural_style_transfer_imagers DATA_DIR)
    set(NEURAL_STYLE_TRANSFER_IMAGES_DIR
        "${DATA_DIR}/neural_style_transfer_images")
    set(NEURAL_STYLE_TRANSFER_IMAGES_URL
        "https://github.com/yunjey/pytorch-tutorial/raw/master/tutorials/03-advanced/neural_style_transfer/png/"
    )

    set(NEURAL_STYLE_TRANSFER_IMAGE_FILES "content.png" "style.png")

    set(NEURAL_STYLE_TRANSFER_IMAGE_FILE_MD5S
        "03741688105456fb5c8400bde7e06363" "5d9cdb15e34930e8e9cba5fc8416cdcd")

    include(${CMAKE_CURRENT_LIST_DIR}/check_files.cmake)
    check_files(
        ${NEURAL_STYLE_TRANSFER_IMAGES_DIR} NEURAL_STYLE_TRANSFER_IMAGE_FILES
        NEURAL_STYLE_TRANSFER_IMAGE_FILE_MD5S MISSING_FILES)

    if(MISSING_FILES)
        message(STATUS "Fetching Neural Style Transfer images...")

        foreach(FILE_TO_FETCH ${MISSING_FILES})
            list(FIND NEURAL_STYLE_TRANSFER_IMAGE_FILES ${FILE_TO_FETCH}
                 MD5_IDX)
            list(GET NEURAL_STYLE_TRANSFER_IMAGE_FILE_MD5S ${MD5_IDX}
                 EXPECTED_FILE_MD5)

            file(DOWNLOAD "${NEURAL_STYLE_TRANSFER_IMAGES_URL}/${FILE_TO_FETCH}"
                 "${NEURAL_STYLE_TRANSFER_IMAGES_DIR}/${FILE_TO_FETCH}"
                 EXPECTED_MD5 ${EXPECTED_FILE_MD5})
        endforeach()

        message(STATUS "Fetching Neural Style Transfer images - done")
    endif()
endfunction()

fetch_neural_style_transfer_imagers(${DATA_DIR})
