cmake_minimum_required(VERSION 3.20)

project(ml_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Main compile
add_library(ml_core
    src/vector.cpp
    src/matrix.cpp 
)

target_include_directories(ml_core
    PUBLIC include
)

enable_testing()

# Vector
add_executable(vector_tests
    tests/VectorTests.cpp
)

target_link_libraries(vector_tests
    PRIVATE ml_core
)

add_test(NAME VectorTests COMMAND vector_tests)

# Matrix
add_executable(matrix_tests
    tests/MatrixTests.cpp
)

target_link_libraries(matrix_tests
    PRIVATE ml_core
)

add_test(NAME MatrixTests COMMAND matrix_tests)