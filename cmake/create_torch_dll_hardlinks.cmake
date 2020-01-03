cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
                       
foreach(TORCH_DLL_FILE ${TORCH_DLLS})
    get_filename_component(TORCH_DLL_NAME ${TORCH_DLL_FILE} NAME)
    # Create hardlink to the DLL files, if possible, otherwise fall back to copy.
    file(CREATE_LINK ${TORCH_DLL_FILE} "${DESTINATION_DIR}/${TORCH_DLL_NAME}" COPY_ON_ERROR)
endforeach()