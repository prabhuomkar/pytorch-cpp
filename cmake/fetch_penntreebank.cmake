cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

function(fetch_penntreebank DATA_DIR)
    set(PENNTREEBANK_DIR "${DATA_DIR}/penntreebank")
    set(PENNTREEBANK_DOWNLOAD_NAME "train.txt")
    set(PENNTREEBANK_URL
        "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt"
    )
    set(PENNTREEBANK_MD5 "f26c4b92c5fdc7b3f8c7cdcb991d8420")

    include(${CMAKE_CURRENT_LIST_DIR}/check_files.cmake)
    check_files(${PENNTREEBANK_DIR} PENNTREEBANK_DOWNLOAD_NAME PENNTREEBANK_MD5
                MISSING_FILES)

    if(MISSING_FILES)
        message(STATUS "Fetching Penn Treebank dataset...")
        file(DOWNLOAD ${PENNTREEBANK_URL}
             "${PENNTREEBANK_DIR}/${PENNTREEBANK_DOWNLOAD_NAME}"
             EXPECTED_MD5 ${PENNTREEBANK_MD5})
        message(STATUS "Fetching Penn Treebank datase - done")
    endif()
endfunction()

fetch_penntreebank(${DATA_DIR})
