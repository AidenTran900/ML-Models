# LinearModel

A C++ implementation of linear regression using gradient descent optimization. This project demonstrates fundamental machine learning concepts including cost function computation and parameter optimization through gradient descent.

## Features

- **Matrix Operations**: Efficient matrix addition and multiplication for linear algebra computations
- **Linear Regression**: Implementation of univariate and multivariate linear regression
- **Cost Function**: Mean Squared Error (MSE) calculation for model evaluation
- **Gradient Descent**: Iterative optimization algorithm to minimize the cost function
- **CMake Build System**: Cross-platform build configuration

## Mathematical Background

### Linear Regression Model

The hypothesis function for linear regression is:

```
h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

Or in vectorized form: `h(x) = θᵀX`

### Cost Function (Mean Squared Error)

```
J(θ) = (1/2m) Σ(h(xⁱ) - yⁱ)²
```

Where:
- `m` is the number of training examples
- `h(xⁱ)` is the predicted value
- `yⁱ` is the actual value

### Gradient Descent

The parameter update rule:

```
θⱼ := θⱼ - α(∂/∂θⱼ)J(θ)
```

Where:
- `α` is the learning rate
- The partial derivative represents the gradient

## Prerequisites

- C++14 or higher
- CMake 3.16 or higher
- A C++ compiler (GCC, Clang, MSVC)

## Building the Project

### Unix/Linux/macOS

```bash
# Clone the repository
git clone <your-repo-url>
cd LinearModel

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
cmake --build .

# Run the executable
./Build
```

### Windows

```bash
# Clone the repository
git clone <your-repo-url>
cd LinearModel

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
cmake --build . --config Release

# Run the executable
.\Release\Build.exe
```

## Project Structure

```
LinearModel/
├── source/
│   ├── main.cpp              # Entry point and demonstration
│   └── Helpers/
│       └── matrix.cpp        # Matrix operations implementation
├── external/                 # Third-party dependencies
│   ├── fmt/                  # Formatting library
│   └── spdlog/              # Logging library
├── CMakeLists.txt           # CMake configuration
└── README.md
```

## Usage Example

```cpp
#include "Helpers/matrix.cpp"

int main() {
    // Define matrices
    Matrix X = {{1, 2},
                {3, 4}};
    Matrix Y = {{5, 6},
                {7, 8}};

    // Matrix addition
    Matrix sum = addMatrix(X, Y);
    printMatrix(sum);

    // Matrix multiplication
    Matrix product = multiplyMatrix(X, Y);
    printMatrix(product);

    return 0;
}
```

## Roadmap

- [x] Basic matrix operations (addition, multiplication)
- [ ] Implement linear regression hypothesis function
- [ ] Implement cost function (MSE)
- [ ] Implement gradient descent algorithm
- [ ] Add data normalization/feature scaling
- [ ] Support for CSV data loading
- [ ] Visualization of cost function convergence
- [ ] Model serialization and loading
- [ ] Multiple optimization algorithms (Batch, Stochastic, Mini-batch)

## Theory

### Why Gradient Descent?

Gradient descent is an optimization algorithm used to minimize the cost function by iteratively moving in the direction of steepest descent. It's particularly useful for linear regression because:

1. It can handle large datasets efficiently
2. It scales well to multiple features
3. It provides a foundation for understanding more complex optimization algorithms

### Learning Rate (α)

The learning rate determines the step size at each iteration. Choosing the right learning rate is crucial:
- Too small: Slow convergence
- Too large: May overshoot the minimum or diverge

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is available for educational purposes.

## References

- Andrew Ng's Machine Learning Course
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman

## Acknowledgments

- Built using [fmt](https://github.com/fmtlib/fmt) for modern C++ formatting
- CMake boilerplate structure for cross-platform compatibility
