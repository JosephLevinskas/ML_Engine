# ML Engine

A production-level Machine Learning engine built from first principles in modern C++. This repository is focused on rigorous math, clean architecture, and minimal dependencies.

## Engineering Approach

This project is structured like a senior-engineer codebase:

- **WHY before HOW**: Every interface and implementation is justified by algorithmic behavior and ML requirements.
- **Design questions first**: Before implementing, we question dimension correctness, performance trade-offs, and invariants.
- **Incremental correctness**: Build one primitive at a time, with tests driving each step.
- **Minimal abstractions**: No unnecessary layers, no hidden allocations, no external ML frameworks.
- **Production-minded**: Clean API, clear ownership, and explicit error handling.

## Current Project Goals

Build a complete ML system from scratch in C++:

- No external ML libraries (PyTorch, TensorFlow, etc.)
- Minimal dependencies: Standard C++ and CMake only
- Thorough tests for every module
- Clear, maintainable architecture suitable for real systems

## Progress Status

1. ✅ Vector math system
2. ✅ Matrix system
3. ✅ Matrix × Vector multiplication and dot product operations
4. ✅ Linear prediction model (`LinearModel`)
5. ✅ Mean squared error loss
6. ✅ Gradient computation for linear models
7. ✅ Gradient descent training loop (`Trainer`)
8. ✅ Feature scaling (`StandardScaler`)
9. ✅ Linear regression pipeline (`LinearRegressionPipeline`)
10. ⬜ Logistic regression
11. ⬜ Neural network layers and backpropagation
12. ⬜ Training loop, evaluation, and application

## Current Architecture

```
ml_engine/
├── apps/
│   └── main.cpp
├── include/ml/
│   ├── Gradients.h
│   ├── LinearModel.h
│   ├── LinearRegressionPipeline.h
│   ├── LossFunctions.h
│   ├── Matrix.h
│   ├── Operations.h
│   ├── StandardScaler.h
│   ├── Trainer.h
│   └── Vector.h
├── src/
│   ├── Gradients.cpp
│   ├── LinearModel.cpp
│   ├── LinearRegressionPipeline.cpp
│   ├── LossFunctions.cpp
│   ├── Matrix.cpp
│   ├── Operations.cpp
│   ├── StandardScaler.cpp
│   ├── Trainer.cpp
│   └── Vector.cpp
├── tests/
│   ├── GradientsTests.cpp
│   ├── LinearModelTests.cpp
│   ├── LinearRegressionPipelineTests.cpp
│   ├── LossFunctionsTests.cpp
│   ├── MatrixTests.cpp
│   ├── OperationsTests.cpp
│   ├── StandardScalerTests.cpp
│   ├── TrainerTests.cpp
│   └── VectorTests.cpp
├── CMakeLists.txt
├── LICENSE
└── README.md
```

## Implemented Components

### Vector

- `Vector` stores data as `std::vector<double>`.
- Enforces non-empty construction.
- Provides bounds-checked access.
- Supports core arithmetic and dot-product semantics.
- Designed for safe, predictable primitive operations.

### Matrix

- `Matrix` stores contiguous row-major data.
- Invariants: positive row and column counts.
- Access via `matrix(row, col)` with bounds checks.
- No `Vector` dependency inside storage.
- Designed for efficient linear algebra primitives.

### Operations

- `dot(Vector, Vector)` validates size and computes the inner product.
- `Matrix * Vector` implements linear algebra multiplication.
- This is the exact primitive behind `predictions = X * weights`.
- Dimension checks are enforced and reported via exceptions.

### LinearModel

- `LinearModel` stores `weights` and `bias`.
- Supports single-sample prediction `predict(Vector)`.
- Supports batch prediction `predict(Matrix)`.
- Uses matrix-vector multiplication for batched inference.
- Validates feature dimension alignment.

### LossFunctions

- `meanSquaredError(predictions, targets)` computes the standard MSE.
- Validates matching vector lengths.
- Provides scalar loss for regression evaluation.

### Gradients

- `computeLinearModelMSEGradients(X, predictions, targets)` computes gradients for linear regression.
- Returns both weight gradients and bias gradient.
- Uses explicit matrix/vector math with no external helpers.
- This is the next step before adding gradient descent.

### Trainer

- `Trainer` implements gradient descent optimization.
- Configurable learning rate and maximum epochs.
- Tracks loss progression during training.
- Updates model parameters iteratively to minimize loss.

### StandardScaler

- `StandardScaler` performs feature standardization (z-score normalization).
- Computes mean and standard deviation from training data.
- Transforms features to zero mean and unit variance.
- Essential preprocessing step for gradient-based optimization.

### LinearRegressionPipeline

- `LinearRegressionPipeline` combines scaling and training into a complete workflow.
- Integrates `StandardScaler` and `Trainer` with `LinearModel`.
- Provides end-to-end training and prediction capabilities.
- Demonstrates a production-ready ML pipeline.

## Testing

Every component has a dedicated executable and test file:

- `vector_tests`
- `matrix_tests`
- `operations_tests`
- `linearmodel_tests`
- `loss_functions_tests`
- `gradients_tests`
- `trainer_tests`
- `standard_scaler_tests`
- `linear_regression_pipeline_tests`

Tests verify correct results, dimension mismatches, and edge-case behavior.

## Build and Run

```bash
cd build
cmake ..
cmake --build . --config Release
ctest -C Release
```

### Demo Application

The `apps/main.cpp` demonstrates a complete linear regression pipeline:

- Generates synthetic training data
- Trains a linear model using gradient descent
- Evaluates performance on training data
- Makes predictions on new data
- Shows loss progression during training

Run the demo with:

```bash
./Release/main.exe
```

## Next Immediate Step

The current codebase has a complete linear regression pipeline with training, scaling, and prediction. The next concrete steps are to implement logistic regression for binary classification, including sigmoid activation, binary cross-entropy loss, and corresponding gradients.

## License

MIT License - see LICENSE file for details.