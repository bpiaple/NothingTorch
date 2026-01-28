# NothingTorch

NothingTorch is a minimal, educational deep learning framework built from scratch using **JAX**. It aims to provide a PyTorch-like API while leveraging the speed and composability of JAX for automatic differentiation and XLA acceleration.

## 🚀 Features

- **Tensor Engine**: A custom `Tensor` class wrapping JAX arrays for seamless integration.
- **Layers**: Modular neural network layers (e.g., `Linear`).
- **Activations**: Standard activation functions like `ReLU`.
- **Autograd**: Built-in automatic differentiation via JAX's backend.
- **Modular Design**: Organized into `core` modules for `activations`, `layers`, `losses`, `optimizers`, and `autograd`.

## 📦 Installation

To use NothingTorch, ensure you have Python 3.10 or higher installed. You can install the primary dependencies using:

```bash
pip install jax jaxlib
```

Or, if you use `uv`:

```bash
uv sync
```

## 🛠️ Usage Example

Here is a simple example of how to create a tensor and pass it through a basic neural network:

```python
from core.activations.activations import ReLU
from core.layers.linear import Linear
from core.tensor import Tensor

# Define a simple multi-layer perceptron
layer1 = Linear(3, 4)
layer2 = Linear(4, 2)
relu = ReLU()

# Create input data
x = Tensor([[1.0, 2.0, 3.0]])

# Forward pass
h1 = relu(layer1(x))
output = layer2(h1)

print(f"Output: {output.data}")
```

## 📂 Project Structure

- `core/`: Main library code.
  - `tensor.py`: The base Tensor class.
  - `layers/`: Neural network layers.
  - `activations/`: Non-linear activation functions.
  - `optimizers/`: Optimization algorithms.
  - `losses/`: Loss functions for training.
- `tests/`: Unit tests to ensure framework stability.
- `main.py`: Example script demonstrating framework usage.

## 🧪 Testing

Run the test suite to ensure everything is working correctly:

```bash
python -m pytest tests/
```