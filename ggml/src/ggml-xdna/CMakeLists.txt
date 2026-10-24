if (DEFINED ENV{XILINX_XRT})
    find_package(XRT REQUIRED CONFIG HINTS "$ENV{XILINX_XRT}/share/cmake/XRT")
else()
    find_package(XRT REQUIRED CONFIG)
endif()

ggml_add_backend_library(ggml-xdna
    ggml-xdna.cpp
)

if (MSVC)
    target_compile_options(ggml-xdna PRIVATE /Zc:__cplusplus)
endif()

target_link_libraries(ggml-xdna PRIVATE XRT::xrt_coreutil ${CMAKE_DL_LIBS})

set(XDNA_KERNEL_RUNTIME_DIR "$<TARGET_FILE_DIR:ggml-xdna>")
if (NOT GGML_BACKEND_DL)
    if (CMAKE_CONFIGURATION_TYPES)
        set(XDNA_KERNEL_RUNTIME_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>")
    else()
        set(XDNA_KERNEL_RUNTIME_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    endif()
endif()

add_custom_target(ggml-xdna-kernels)
add_dependencies(ggml-xdna ggml-xdna-kernels)

set(XDNA_Q4_0_MATVEC_SHAPES
    "16x1x256"
    "256x1x1536"
    "1536x1x1536"
    "8960x1x1536"
    "1536x1x8960"
)
if (GGML_BACKEND_DL AND GGML_BACKEND_DIR)
    set(XDNA_KERNEL_INSTALL_DIR "${GGML_BACKEND_DIR}")
else()
    set(XDNA_KERNEL_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
endif()

foreach(shape IN LISTS XDNA_Q4_0_MATVEC_SHAPES)
    set(XDNA_KERNEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/q4_0_matvec_${shape}")
    set(XDNA_KERNEL_RUNTIME_SUBDIR "${XDNA_KERNEL_RUNTIME_DIR}/xdna-kernels/q4_0_matvec_${shape}")
    add_custom_command(TARGET ggml-xdna-kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${XDNA_KERNEL_RUNTIME_SUBDIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${XDNA_KERNEL_SOURCE_DIR}/q4_0_matvec.xclbin"
            "${XDNA_KERNEL_RUNTIME_SUBDIR}/q4_0_matvec.xclbin"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${XDNA_KERNEL_SOURCE_DIR}/q4_0_matvec.insts.bin"
            "${XDNA_KERNEL_RUNTIME_SUBDIR}/q4_0_matvec.insts.bin"
    )
    install(FILES
        "${XDNA_KERNEL_SOURCE_DIR}/q4_0_matvec.xclbin"
        "${XDNA_KERNEL_SOURCE_DIR}/q4_0_matvec.insts.bin"
        DESTINATION "${XDNA_KERNEL_INSTALL_DIR}/xdna-kernels/q4_0_matvec_${shape}"
    )
endforeach()
