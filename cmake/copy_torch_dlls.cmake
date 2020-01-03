cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

function(copy_torch_dlls TARGET_NAME)
    # According to https://github.com/pytorch/pytorch/issues/25457
    # the torch DLLs need to be copied to avoid memory errors.
    # (see also: https://pytorch.org/cppdocs/installing.html )
    # Here hardlinks to the DLLs are created instead (if possible), to safe disk space.
    list(GET CMAKE_MODULE_PATH 0 CMAKE_SCRIPT_DIR)
    
    add_custom_command(TARGET ${TARGET_NAME}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND}
                       -D "TORCH_INSTALL_PREFIX=${TORCH_INSTALL_PREFIX}"
                       -D "DESTINATION_DIR=$<TARGET_FILE_DIR:${TARGET_NAME}>" 
                       -P "${CMAKE_SCRIPT_DIR}/create_torch_dll_hardlinks.cmake")
endfunction()