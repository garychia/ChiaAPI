set(CHIA_RUNTIME_HEADER_FILES
    "include/Exceptions.hpp"
)
set(CHIA_RUNTIME_SOURCE_FILES
    "src/Exceptions.cpp"
)

if (CHIA_CUDA_ENABLED)
    list(APPEND CHIA_RUNTIME_HEADER_FILES
        "include/ChiaCUDA/CUDAUtilities.cuh"
    )
    list(APPEND CHIA_RUNTIME_SOURCE_FILES
        "src/ChiaCUDA/CUDAUtilities.cu"
    )
endif()

add_library(ChiaRuntime STATIC
    ${CHIA_RUNTIME_HEADER_FILES}
    ${CHIA_RUNTIME_SOURCE_FILES}
)

if (CHIA_CUDA_ENABLED)
    find_package(CUDAToolkit)
    target_link_libraries(ChiaRuntime CUDA::cudart)
    set_target_properties(ChiaRuntime PROPERTIES
        LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_compile_options(ChiaRuntime PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda>)
endif()

set_target_properties(ChiaRuntime PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(ChiaRuntime PRIVATE
    "include"
    "${CMAKE_SOURCE_DIR}/ChiaData/include"
)
