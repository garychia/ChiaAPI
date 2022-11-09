set(CHIA_ALGS_HEADER_FILES
    "include/BinarySearch.hpp"
)
if (CHIA_CUDA_ENABLED)
    list(APPEND CHIA_ALGS_HEADER_FILES
        "include/ChiaCUDA/Arithmetic.cuh"
        "include/ChiaCUDA/ArrayUtilities.cuh"
    )
endif()

add_library(ChiaAlgs INTERFACE ${CHIA_ALGS_HEADER_FILES})
target_include_directories(ChiaAlgs INTERFACE
    "include"
    "${CMAKE_SOURCE_DIR}/ChiaMath/include"
    "${CMAKE_SOURCE_DIR}/ChiaData/include"
)
