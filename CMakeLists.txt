cmake_minimum_required(VERSION 3.20)

project(ml_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Main compile
add_library(ml_core
    src/Vector.cpp
    src/Matrix.cpp 
    src/Operations.cpp
    src/LinearModel.cpp
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


# Operations 
add_executable(operations_tests
    tests/OperationsTests.cpp
)

target_link_libraries(operations_tests
    PRIVATE ml_core
)

add_test(NAME OperationsTests COMMAND operations_tests)


# LinearModel
add_executable(linearmodel_tests
    tests/LinearModelTests.cpp
)

target_link_libraries(linearmodel_tests
    PRIVATE ml_core
)

add_test(NAME LinearModelTests COMMAND linearmodel_tests)