# C-Tensor-Engine

A minimal, zero-dependency tensor engine written in pure C. This is a learning project built to explore the internal mechanics of dynamic computation graphs, automatic differentiation, and neural network training loops.

## Iteration Journey

### v0.1: The Foundation (Graph & Autograd)
- **Core Data Structure**: Designed the foundational `Tensor` struct to handle N-dimensional arrays.
- **Math Operators**: Implemented forward pass for matrix multiplication (`tensor_matmul`) and addition (`tensor_add`).
- **Dynamic Computation Graph**: Built the Directed Acyclic Graph (DAG) by tracking `_prev` nodes during mathematical operations.
- **Automatic Differentiation**: Implemented topological sorting for reverse-mode Autograd chain-rule propagation.

### v0.2: The Alchemist (Training Loop)
- **Non-Linearity**: Added the `ReLU` activation function and its derivative.
- **Objective Function**: Implemented Mean Squared Error (`MSE`) loss for regression tasks.
- **Optimization**: Built a Stochastic Gradient Descent (`SGD`) optimizer with `.zero_grad()` and `.step()`.
- **Memory Safety**: Conquered C's manual memory management, ensuring zero memory leaks during the continuous building and tearing down of the computation graph across thousands of epochs.

### v0.3: The Reality (MNIST Classification) - *Current*
- **Memory-Efficient Broadcasting**: Implemented 2D Row/Column broadcasting (zero-copy) and automatic gradient reduction (Reduce Sum).
- **Probabilistic Magic**: Fused `Softmax` and `Cross-Entropy Loss` into a single operator, utilizing Max-Shift defensive programming to prevent `float32` NaN overflows.
- **Binary Data Loader**: Wrote a pure C binary IO parser (with Endianness swapping) to ingest the 60,000-image MNIST dataset (`.idx-ubyte`).
- **Model Serialization**: Added memory dump capabilities to save and load trained tensor weights as custom `.ctensor` binary files.

## Build & Run

### Prerequisites
- GCC compiler
- `make`
- MNIST dataset binary files (unzipped) placed in a `data/` directory at the project root:
  - `train-images-idx3-ubyte`
  - `train-labels-idx1-ubyte`

### Quick Start
```bash
# Compile the engine and the MNIST training test suite
make test

# (Optional) Run memory leak checks if Valgrind is installed
make memcheck

# Clean build artifacts
make clean