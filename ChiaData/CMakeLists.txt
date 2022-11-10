set(CHIA_DATA_HEADER_FILES
    "include/Array.hpp"
    "include/DynamicArray.hpp"
    "include/HashTable.hpp"
    "include/List.hpp"
    "include/Maybe.hpp"
    "include/Pair.hpp"
    "include/Pointers.hpp"
    "include/Set.hpp"
    "include/Str.hpp"
    "include/String.hpp"
    "include/Math/Container.hpp"
    "include/Math/Matrix.hpp"
    "include/Math/Tuple.hpp"
    "include/Math/Vector.hpp"
    "include/ChiaCUDA/CUDAArray.cuh"
    "include/ChiaCUDA/Math/CUDAVector.cuh"
)

add_library(ChiaData INTERFACE
    ${CHIA_DATA_HEADER_FILES}
)
target_include_directories(ChiaData INTERFACE
    "include"
    "${CMAKE_SOURCE_DIR}/ChiaAlgs/include"
    "${CMAKE_SOURCE_DIR}/ChiaRuntime/include"
)
target_link_libraries(ChiaData
    ChiaRuntime
)
