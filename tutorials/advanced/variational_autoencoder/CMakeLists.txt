cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(variational-autoencoder VERSION 1.0.0 LANGUAGES CXX)

# Files
set(SOURCES src/main.cpp
			src/variational_autoencoder.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils/src/save_image.cpp
)

set(HEADERS include/variational_autoencoder.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include/save_image.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include/external/stb_image_write.h
)

set(EXECUTABLE_NAME variational-autoencoder)


add_executable(${EXECUTABLE_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${EXECUTABLE_NAME} PRIVATE include
                                                      ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include
                                                      ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include/external)

target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}")

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# See https://pytorch.org/cppdocs/installing.html.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${EXECUTABLE_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${EXECUTABLE_NAME}>)
endif (MSVC)