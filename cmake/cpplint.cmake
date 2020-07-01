cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

find_program(CPPLINT cpplint)

if(CPPLINT)
    set(CPPLINT_COMMAND ${CPPLINT})
else()
    get_filename_component(CPPLINT_PY_FILE "cpplint.py" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/../tools/")
    
    if(NOT EXISTS ${CPPLINT_PY_FILE})
        message(STATUS "Downloading cpplint python script from https://raw.githubusercontent.com/cpplint/cpplint/master/cpplint.py...")
        file(DOWNLOAD "https://raw.githubusercontent.com/cpplint/cpplint/master/cpplint.py" ${CPPLINT_PY_FILE})
        message(STATUS "Downloading cpplint python script - done")
    endif()
    
    message(STATUS "Using cpplint python script at ${CPPLINT_PY_FILE}")
    
    set(CPPLINT_COMMAND python ${CPPLINT_PY_FILE})
endif()

execute_process(COMMAND ${CPPLINT_COMMAND}
                "--linelength=120" 
                "--recursive" 
                "--filter=-build/include_subdir,-build/include_what_you_use,-build/c++11,-runtime/references"
                "main.cpp"
                "tutorials"
                "utils"
                WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/..")
