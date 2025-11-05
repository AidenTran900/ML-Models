# ML Models

A C++ machine learning library built from the ground up. Implementing various ML algorithms and models using fundamental linear algebra and optimization techniques.

## Features

- Linear regression with gradient descent optimization
- Matrix operations library (addition, multiplication, transpose, inverse)
- Loss functions: MSE, MAE, RMSE, L1, L2
- Evaluation metrics: R2, Adjusted R2, MSE, MAE, RMSE
- Regularization: L1 (Lasso) & L2 (Ridge)
- Housing price prediction example with real-world dataset

## Prerequisites

- C++14 or higher
- CMake 3.16+
- A C++ compiler (GCC, Clang, or MSVC)

## Building

### Linux/macOS

```bash
# Clone the repository
git clone https://github.com/ProdigiousPersonn/ML-Models
cd ML-Models

# Create and enter build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .

# Run the executable
./Build
```

### Windows

```bash
# Clone the repository
git clone https://github.com/ProdigiousPersonn/ML-Models
cd ML-Models

# Create and enter build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build . --config Release

# Run the executable
.\Release\Build.exe
```

## Project Structure

```
LinearModel/
├── source/
│   ├── main.cpp           # Entry point
│   ├── math/              # Matrix operations
│   ├── core/              # Loss, optimizer, regularizer, metrics
│   └── models/            # ML model implementations
├── include/
│   └── ml_lib/            # Public headers
├── examples/
│   └── linear-regression/
│       └── housing/       # Housing price prediction example
├── tests/                 # Unit tests
├── external/              # Dependencies (fmt, spdlog, doctest)
└── CMakeLists.txt        # Build configuration
```

## Examples

### Housing Price Prediction

A complete example demonstrating linear regression on a real-world housing dataset (https://www.kaggle.com/datasets/yasserh/housing-prices-dataset):

- Dataset: 545 housing samples with 12 features (area, bedrooms, bathrooms, etc.)
- Features: Z-score normalization
- Model: Linear regression with L2 regularization
- Optimizer: Batch gradient descent
- Metrics: MSE, RMSE, MAE, R²

Run the example:
```bash
./Build
```

## Roadmap

### Regression [X]
- [x] **Linear Regression**
- [x] **Evaluation Metrics (Regression):** MSE, MAE, RMSE, R-squared
- [X] **Regularization:** L1 (Lasso) & L2 (Ridge)

### Classification [ ]
- [ ] **Logistic Regression**
- [ ] **Evaluation Metrics (Classification):**
    - [ ] Confusion Matri
    - [ ] Accuracy, Precision, Recall, F1-Score
    - [ ] ROC Curve and AUC
- [ ] **K-Nearest Neighbors (KNN)**
- [ ] **Support Vector Machines (SVMs)**

### Tree-Based Models [ ]
- [ ] **Decision Trees**
- [ ] **Random Forests**

### Unsupervised Learning [ ]
- [ ] **K-Means Clustering**

### Deep Learning Foundations [ ]
- [ ] **Neural Networks (Feedforward)**
- [ ] **Backpropagation**
- [ ] **Optimizers:**
    - [ ] **Mini-Batch Gradient Descent**
    - [ ] **Adam Optimizer**
- [ ] **Model Serialization**

### DL Architectures [ ]
- [ ] **Convolutional Neural Networks (CNNs)** (For images)
- [ ] **Recurrent Neural Networks (RNNs)** (For sequences)

Stuff I'll probably never get to
### Modern NLP (Language) [ ]
- [ ] **Embeddings (Word2Vec, GloVe)**
- [ ] **Attention Mechanisms**
- [ ] **Transformers**
- [ ] **Language Models (Basic LLM architecture)**

## License

This project is available for educational purposes.
