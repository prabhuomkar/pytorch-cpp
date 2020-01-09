cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(MNIST_DOWNLOAD_DIR "${CMAKE_SOURCE_DIR}/data/mnist/download")
set(MNIST_DIR "${CMAKE_SOURCE_DIR}/data/mnist")
set(MNIST_URL "http://yann.lecun.com/exdb/mnist")

set(MNIST_EXTRACTED_FILES "t10k-images-idx3-ubyte" 
                          "t10k-labels-idx1-ubyte" 
                          "train-images-idx3-ubyte" 
                          "train-labels-idx1-ubyte")
                
set(MNIST_EXTRACTED_FILE_MD5S "2646ac647ad5339dbf082846283269ea" 
                              "27ae3e4e09519cfbb04c329615203637" 
                              "6bbc9ace898e44ae57da46a324031adb" 
                              "a25bea736e30d166cdddb491f175f624")
                             
include(check_files)

check_files(${MNIST_DIR} MNIST_EXTRACTED_FILES MNIST_EXTRACTED_FILE_MD5S MISSING_FILES) 

if(NOT MISSING_FILES)
    message(STATUS "MNIST dataset already present, skipping...")
else()
    message(STATUS "Fetching MNIST dataset...")
    
    set(MNIST_ARCHIVE_MD5S "9fb629c4189551a2d022fa330f9573f3" 
                           "ec29112dd5afa0611ce80d1b7f02629c" 
                           "f68b3c2dcbeaaa9fbdd348bbdeb94873" 
                           "d53e105ee54ea40749a09fcbcd1e9432") 
                           
    include(find_gz_extractor)
    
    foreach(FILE_TO_FETCH ${MISSING_FILES})
        list(FIND MNIST_EXTRACTED_FILES ${FILE_TO_FETCH} MD5_IDX)
        list(GET MNIST_ARCHIVE_MD5S ${MD5_IDX} EXPECTED_FILE_MD5)
        
        file(DOWNLOAD "${MNIST_URL}/${FILE_TO_FETCH}.gz" "${MNIST_DIR}/${FILE_TO_FETCH}.gz" EXPECTED_MD5 ${EXPECTED_FILE_MD5})
        
        if(EXTRACTOR_COMMAND)
            execute_process(COMMAND ${EXTRACTOR_COMMAND} "${MNIST_DIR}/${FILE_TO_FETCH}.gz" OUTPUT_FILE "${MNIST_DIR}/${FILE_TO_FETCH}")
            file(REMOVE "${MNIST_DIR}/${FILE_TO_FETCH}.gz")
        endif()
    endforeach()

    message(STATUS "Fetching MNIST dataset - done") 
endif()