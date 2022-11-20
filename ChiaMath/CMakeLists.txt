set(CHIA_MATH_HEADER_FILES
    "include/Math.hpp"
    "include/Random.hpp"
)
set(CHIA_MATH_SOURCE_FILES
    "src/Math.cpp"
    "src/Random.cpp"
)

add_library(ChiaMath STATIC
    ${CHIA_MATH_HEADER_FILES}
    ${CHIA_MATH_SOURCE_FILES}
)
target_include_directories(ChiaMath PUBLIC
    "include"
    "${CMAKE_SOURCE_DIR}/ChiaData/include"
    "${CMAKE_SOURCE_DIR}/ChiaRuntime/include"
)
target_link_libraries(ChiaMath PRIVATE ChiaRuntime)
