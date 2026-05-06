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
    src/LossFunctions.cpp
    src/Gradients.cpp
    src/Trainer.cpp
    src/StandardScaler.cpp
    src/LinearRegressionPipeline.cpp
    src/LogisticModel.cpp
    src/DatasetLoader.cpp
    src/DataSplitter.cpp
)

target_include_directories(ml_core
    PUBLIC include
)

enable_testing()


# Main
add_executable(main
    apps/main.cpp
)

target_link_libraries(main
    PRIVATE ml_core
)


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


# LossFunction
add_executable(loss_functions_tests
    tests/LossFunctionsTests.cpp
)

target_link_libraries(loss_functions_tests
    PRIVATE ml_core
)

add_test(NAME LossFunctionsTests COMMAND loss_functions_tests)


# Gradients
add_executable(gradients_tests
    tests/GradientsTests.cpp
)

target_link_libraries(gradients_tests
    PRIVATE ml_core
)

add_test(NAME GradientsTests COMMAND gradients_tests)


# Trainer
add_executable(trainer_tests
    tests/TrainerTests.cpp
)

target_link_libraries(trainer_tests
    PRIVATE ml_core
)

add_test(NAME TrainerTests COMMAND trainer_tests)


# Trainer
add_executable(standard_scaler_tests
    tests/StandardScalerTests.cpp
)

target_link_libraries(standard_scaler_tests
    PRIVATE ml_core
)

add_test(NAME StandardScaler COMMAND standard_scaler_tests)


# LinearRegressionPipeline
add_executable(linear_regression_pipeline_tests
    tests/LinearRegressionPipelineTests.cpp
)

target_link_libraries(linear_regression_pipeline_tests
    PRIVATE ml_core
)

add_test(NAME LinearRegressionPipeline COMMAND linear_regression_pipeline_tests)


# LogisticModel
add_executable(logistic_model_tests
    tests/LogisticModelTests.cpp
)

target_link_libraries(logistic_model_tests
    PRIVATE ml_core
)

add_test(NAME LogisticModel COMMAND logistic_model_tests)


# LogisticRegressionTests
add_executable(logistic_regression_trainer_tests
    tests/LogisticRegressionTrainerTests.cpp
)

target_link_libraries(logistic_regression_trainer_tests
    PRIVATE ml_core
)

add_test(NAME LogisticRegressionTrainer COMMAND logistic_regression_trainer_tests)