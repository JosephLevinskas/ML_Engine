# ML Engine

A production-level Machine Learning engine built from scratch in C++. This is a long-term project focused on deep understanding of mathematics, algorithms, and systems design.

## Learning Approach

This project follows a senior engineering methodology:

- **Explain WHY before HOW**: Every decision is justified mathematically and architecturally.
- **Design questions first**: Critical thinking about trade-offs precedes implementation.
- **Incremental progress**: Build and test step-by-step, no shortcuts.
- **Strict correctness**: Mathematical accuracy and clean architecture are non-negotiable.
- **Minimal implementations**: No unnecessary code, abstractions, or dependencies.
- **First principles**: Everything built from fundamental concepts, tied directly to ML applications.

## Project Goals

Build a complete ML system from scratch in C++:

- **No ML libraries**: No PyTorch, TensorFlow, or similar dependencies.
- **Minimal dependencies**: Only standard C++ and CMake.
- **Fully tested**: Comprehensive test suite for all components.
- **Clean architecture**: Production-quality design like a real ML framework.

## Planned Progression

1. ✅ Vector math system
2. ✅ Matrix system (core completed)
3. Linear regression (gradient descent)
4. Logistic regression
5. Loss functions and optimization
6. Neural networks (forward + backpropagation)
7. Training loops and evaluation
8. Real-world project (audio-related application)

## Architecture

```
ml_engine/
├── include/ml/
│   ├── Vector.h
│   └── Matrix.h
├── src/
│   ├── Vector.cpp
│   └── Matrix.cpp
├── tests/
│   ├── VectorTests.cpp
│   └── MatrixTests.cpp
├── apps/
│   └── main.cpp
├── CMakeLists.txt
└── README.md
```

Built with CMake for proper build/test infrastructure.

## Completed Components

### Vector (COMPLETE)

**Design Decisions:**
- Storage: `std::vector<double>` for contiguous memory and standard library compatibility.
- Invariants: Rejects empty vectors at construction.
- Safety: Bounds-checked indexing via `operator[]`.
- Math operations: Dot product as free function, arithmetic operators (`+`, `-`, `*` scalar, compound assignments).
- Efficiency: Free functions reuse compound operators (single source of truth), no hidden allocations in primitives.
- Syntax: Clean mathematical notation (`v[i]`, `a + b`).

**Key Principle:** Public API is safe, inner math is clean and efficient.

### Matrix (CORE COMPLETED)

**Design Decisions:**
- Storage: Contiguous `std::vector<double>` in row-major order.
- Abstraction: Does NOT use `Vector` internally (proper separation of concerns).
- Invariants: `rows > 0`, `cols > 0`.
- Indexing: `matrix(row, col)` with internal mapping `row * cols + col`.
- Safety: Bounds checking throws `std::out_of_range`.
- Constructors: `(rows, cols)` for zero-initialized, `(rows, cols, data)` with size validation.

**Key Principle:** Storage is low-level and contiguous for performance. Abstraction enforced at interface.

## Current Understanding

- **Dot product**: Core ML primitive for similarity and projections.
- **ML fundamentals**: Parameter optimization to minimize loss functions.
- **Data layout**: Rows = samples, columns = features (standard ML convention).
- **Performance**: Contiguous memory and allocation avoidance critical early.
- **Architecture**: Clear separation between storage efficiency and abstraction safety.
- **Code quality**: Operator design directly impacts readability and mathematical correctness.

## Building and Testing

```bash
# Configure
cd build
cmake ..

# Build
cmake --build . --config Release

# Test
ctest -C Release
```

## License

MIT License - see LICENSE file for details.