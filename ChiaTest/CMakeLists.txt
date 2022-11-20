set(CHIA_TEST_HEADER_FILES
    "include/Test.hpp"
    "include/Tester.hpp"
)

set(CHIA_TEST_SOURCE_FILES
    "src/Tester.cpp"
)

add_library(ChiaTest
    ${CHIA_TEST_HEADER_FILES}
    ${CHIA_TEST_SOURCE_FILES}
)
target_include_directories(ChiaTest PUBLIC
    "include"
)
target_link_libraries(ChiaTest
    ChiaData
    ChiaMath
)
