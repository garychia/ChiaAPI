cmake_minimum_required(VERSION 3.22)
project(ChiaAPI)

set(CMAKE_CXX_STANDARD 17)

option(CHIA_CUDA_ENABLED "Use CUDA for acceleration." ON)

include(CheckLanguage)
if(CHIA_CUDA_ENABLED)
    check_language(CUDA)
    if (NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "No CUDA compiler is found on this system.")
    endif()
	enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    add_compile_definitions(__CHIA_CUDA_ENABLED__=1)
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/archives)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)

add_subdirectory(ChiaData)
add_subdirectory(ChiaRuntime)
add_subdirectory(ChiaMath)
add_subdirectory(ChiaAlgs)
