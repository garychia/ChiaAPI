set(CHIA_SYSTEM_HEADER_FILES
    "include/Version.hpp"
    "include/App/ChiaApp.hpp"
    "include/Sync/Mutex.hpp"
    "include/Window/ChiaWindow.hpp"
    "include/Window/MainLoop.hpp"
)
set(CHIA_SYSTEM_SOURCE_FILES
    "src/Version.cpp"
    "src/App/ChiaApp.cpp"
    "src/Sync/Mutex.cpp"
    "src/Window/ChiaWindow.cpp"
    "src/Window/MainLoop.cpp"
)

add_library(ChiaSystem STATIC
    ${CHIA_SYSTEM_HEADER_FILES}
    ${CHIA_SYSTEM_SOURCE_FILES}
)
target_include_directories(ChiaSystem PUBLIC
    "include"
    "${CMAKE_SOURCE_DIR}/ChiaData/include"
)
