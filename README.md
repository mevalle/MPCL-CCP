# Training Morphological Perceptron with Competitive Layer using Convex-Concave Procedure

This repository contains the implementation of the **Morphological Perceptron with Competitive Layer (MPCL)**, a feedforward neural architecture based on mathematical morphology operations. Unlike traditional neural networks that rely on summation and multiplication, the MPCL uses nonlinear operations such as **erosion** and **anti-dilation**.

## Overview

The MPCL is designed for multi-class classification tasks. It partitions the feature space into a union of **hyperboxes**, providing a highly interpretable model where an input is assigned to a class if it lies within the geometric boundaries defined by the network.

Due to the non-differentiability of morphological extrema operations (max/min), this project implements two primary training strategies that avoid gradient-based methods:
1. **MPCL-CCP**: A training framework based on the **Convex-Concave Procedure (CCP)** that treats the learning task as a Difference-of-Convex (DC) optimization problem.
2. **MPCL-Greedy**: An efficient, incremental approach that sequentially adds hyperboxes until a stopping condition is met.

## Project Structure

* `MPCL.py`: The core Python module containing the class implementations.
* `Experiment - Credit Card`: A Jupyter Notebook demonstrating the application of the model to a real-world Credit Card Fraud Detection dataset from Kaggle.

## Installation & Dependencies

To run this code, you will need the following Python libraries:

* `numpy`, `scipy`, `pandas`
* `cvxpy` and `dccp` (for the DCCP implementation)
* `scikit-learn` (for base estimators and preprocessing)
* `matplotlib` & `seaborn` (for visualization)
* `tensorflow` (for backend utilities)

## Usage

### Training the Model

The models are designed to be compatible with the `scikit-learn` API. You can initialize and fit them as follows:

```python
from MPCL import MPCL_CCP

mpcl = MPCL_CCP(verbose=True)
mpcl.fit(X_train, y_train)

# Making predictions
predictions = mpcl.predict(X_test)

```

### Visualization

The model provides a `show_boxes` method to visualize the learned hyperboxes in a 2D feature space, which is useful for interpreting the decision boundaries.

```python
mpcl.show_boxes(X_train, y_train)

```

## Authors

* **Marcos Eduardo Valle** - University of Campinas (UNICAMP).
* **Iara Cunha** - Universidade Tecnológica Federal do Paraná (UTFPR).



## License

*Recommended License: **Apache License 2.0** or **MIT License**.*
