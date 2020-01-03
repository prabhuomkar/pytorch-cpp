cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Find program that can extract .gz files.
# Based on: https://github.com/Amber-MD/cmake-buildscripts/blob/master/gzip.cmake
find_program(7ZIP_EXECUTABLE 7z 7za DOC "Path to 7zip executable")
find_program(GZIP_EXECUTABLE gzip DOC "Path to gzip executable")
find_program(GUNZIP_EXECUTABLE gunzip DOC "Path to the gunzip executable")

if(7ZIP_EXECUTABLE)
    set(EXTRACTOR_COMMAND ${7ZIP_EXECUTABLE} e -so)
elseif(GZIP_EXECUTABLE)
    set(EXTRACTOR_COMMAND ${GZIP_EXECUTABLE} -dc)
elseif(GUNZIP_EXECUTABLE)
    set(EXTRACTOR_COMMAND ${GUNZIP_EXECUTABLE} -c)
else()
    unset(GUNZIP_EXECUTABLE CACHE)
    unset(GZIP_EXECUTABLE CACHE)
    unset(7ZIP_EXECUTABLE CACHE)

    message(WARNING "Could not find 7zip or gzip program to extract .gz files. Please install or extract manually.")
endif()