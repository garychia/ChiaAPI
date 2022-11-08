set(CHIA_ALGS_HEADER_FILES
    "include/BinarySearch.hpp"
)

add_library(ChiaAlgs INTERFACE ${CHIA_ALGS_HEADER_FILES})
target_include_directories(ChiaAlgs INTERFACE
    "include"
    ${CMAKE_SOURCE_DIR}/ChiaData/include
)
