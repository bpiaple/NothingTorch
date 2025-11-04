from pathlib import Path
import sys
import jax.numpy as jnp
import jax


# Add parent directory to path to import core module
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.layers.linear import Linear
from core.tensor import Tensor

seed = 42

def test_unit_linear_layer():
    """🔬 Test Linear layer implementation."""
    print("🔬 Unit Test: Linear Layer...")

    # Test layer creation
    layer = Linear(784, 256)
    assert layer.in_features == 784
    assert layer.out_features == 256
    assert layer.weight.shape == (784, 256)
    assert layer.bias.shape == (256,)
    assert layer.weight.requires_grad
    assert layer.bias.requires_grad

    # Test Xavier initialization (weights should be reasonably scaled)
    weight_std = jnp.std(layer.weight.data)
    expected_std = jnp.sqrt(1.0 / 784)
    assert 0.5 * expected_std < weight_std < 2.0 * expected_std, f"Weight std {weight_std} not close to Xavier {expected_std}"

    # Test bias initialization (should be zeros)
    assert jnp.allclose(layer.bias.data, 0), "Bias should be initialized to zeros"

    # Test forward pass

    key = jax.random.PRNGKey(seed)
    x = Tensor(jax.random.normal(key, (32, 784)))  # Batch of 32 samples
    y = layer.forward(x)
    assert y.shape == (32, 256), f"Expected shape (32, 256), got {y.shape}"

    # Test no bias option
    layer_no_bias = Linear(10, 5, bias=False)
    assert layer_no_bias.bias is None
    params = layer_no_bias.parameters()
    assert len(params) == 1  # Only weight, no bias

    # Test parameters method
    params = layer.parameters()
    assert len(params) == 2  # Weight and bias
    assert params[0] is layer.weight
    assert params[1] is layer.bias

    print("✅ Linear layer works correctly!")

if __name__ == "__main__":
    test_unit_linear_layer()