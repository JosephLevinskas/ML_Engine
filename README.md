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

## What This Project Demonstrates

This project implements a complete machine learning workflow from scratch:

- Raw CSV data ingestion
- Dataset representation and preprocessing
- Train/test splitting for proper evaluation
- Feature scaling for stable optimization
- Model training using gradient descent
- Prediction and accuracy evaluation on unseen data

All components are implemented manually without external ML libraries.

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
10. ✅ Logistic regression model (`LogisticModel`) with sigmoid activation
11. ✅ Binary cross-entropy loss and logistic gradients
12. ✅ Logistic regression training with end-to-end pipeline
13. ✅ Dataset loading from CSV (`DataSetLoader`)
14. ✅ Train/validation/test split (`DataSplitter`)
15. ✅ Neural network layers (dense layers, activation functions)
16. ✅ Backpropagation for multi-layer networks
17. ⬜ Class-balanced loss / imbalance-aware training
18. ⬜ Advanced training techniques (momentum, regularization)

## Recent Additions

- `include/ml/activations/ReLUActivation.h`
- `include/ml/activations/SigmoidActivation.h`
- `include/ml/layers/DenseLayer.h`
- `src/activations/ReLUActivation.cpp`
- `src/activations/SigmoidActivation.cpp`
- `src/layers/DenseLayer.cpp`
- `src/data/DataSplitter.cpp`
- `apps/main.cpp` now includes debug metrics and class-balance evaluation
- Class-balanced binary cross-entropy is now part of the training demo

## Current Architecture

```
ml_engine/
├── apps/
│   └── main.cpp
├── include/ml/
│   ├── activations/
│   │   ├── ReLUActivation.h
│   │   └── SigmoidActivation.h
│   ├── core/
│   │   ├── Matrix.h
│   │   ├── Operations.h
│   │   └── Vector.h
│   ├── data/
│   │   ├── DataSetLoader.h
│   │   └── DataSplitter.h
│   ├── layers/
│   │   └── DenseLayer.h
│   ├── models/
│   │   ├── LinearModel.h
│   │   └── LogisticModel.h
│   ├── pipelines/
│   │   ├── LinearRegressionPipeline.h
│   │   └── LogisticRegressionPipeline.h
│   └── training/
│       ├── Gradients.h
│       ├── LossFunctions.h
│       ├── StandardScaler.h
│       └── Trainer.h
├── src/
│   ├── core/
│   │   ├── Matrix.cpp
│   │   ├── Operations.cpp
│   │   └── Vector.cpp
│   ├── data/
│   │   ├── DataSetLoader.cpp
│   │   └── DataSplitter.cpp
│   ├── models/
│   │   ├── LinearModel.cpp
│   │   └── LogisticModel.cpp
│   ├── pipelines/
│   │   ├── LinearRegressionPipeline.cpp
│   │   └── LogisticRegressionPipeline.cpp
│   └── training/
│       ├── Gradients.cpp
│       ├── LossFunctions.cpp
│       ├── StandardScaler.cpp
│       └── Trainer.cpp
├── tests/
│   ├── GradientsTests.cpp
│   ├── LinearModelTests.cpp
│   ├── LinearRegressionPipelineTests.cpp
│   ├── LogisticModelTests.cpp
│   ├── LogisticRegressionTrainerTests.cpp
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

- `meanSquaredError(predictions, targets)` computes the standard MSE for regression.
- `binaryCrossEntropy(probabilities, targets)` computes BCE loss for binary classification.
- Validates matching vector lengths.
- Provides scalar loss for evaluation.

### Gradients

- `computeLinearModelMSEGradients(X, predictions, targets)` computes gradients for linear regression.
- `computeLogisticModelBCEGradients(X, probabilities, targets)` computes gradients for logistic regression.
- Returns both weight gradients and bias gradient for each model type.
- Uses explicit matrix/vector math with no external helpers.

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

### LogisticRegressionPipeline

- Combines `StandardScaler`, `Trainer`, and `LogisticModel`.
- Handles feature scaling internally for both training and inference.
- Ensures consistent preprocessing between training and prediction.
- Represents a clean, production-style ML pipeline abstraction.

### LogisticModel

- `LogisticModel` stores `weights` and `bias` for binary classification.
- Supports single-sample prediction `predict(Vector)` returning probability.
- Supports batch prediction `predict(Matrix)` returning probability vector.
- Uses sigmoid activation for binary classification.
- Validates feature dimension alignment.

### DataSetLoader

- `DataSetLoader::loadCSV()` reads CSV files into feature matrices and target vectors.
- Automatically handles feature/target column separation.
- Provides a `Dataset` struct containing features (Matrix) and targets (Vector).
- Essential for loading real-world data for model training and evaluation.

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
- `logistic_model_tests`
- `logistic_regression_trainer_tests`

Tests verify correct results, dimension mismatches, and edge-case behavior.

## Build and Run

```bash
cd build
cmake ..
cmake --build . --config Release
ctest -C Release
```

### Demo Application

The `apps/main.cpp` demonstrates a complete classification workflow with a small neural network:

- Loads CSV data from `assets/User_Data.csv`
- Splits data into training, validation, and test sets (60/20/20)
- Applies feature scaling using `StandardScaler`
- Builds a two-layer network with `DenseLayer`, `ReLUActivation`, and `SigmoidActivation`
- Trains with binary cross-entropy and class-balanced weighting
- Evaluates validation and test accuracy
- Prints debugging metrics per epoch:
  - validation accuracy
  - predicted positive rate
  - average probability
  - min/max probability
- Helps identify majority-class bias and probability drift during training

Run the demo with:

```bash
./Release/main.exe
```

The demo is designed to show realistic ML debugging: class imbalance, probability calibration, and decision-boundary learning rather than just raw accuracy.

## Next Steps

- Add L2 regularization to improve generalization
- Extend pipelines to support configurable preprocessing and training options
- Implement neural network layers (dense layers, activations)
- Add backpropagation for multi-layer models

## License

MIT License - see LICENSE file for details.