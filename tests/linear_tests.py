from pathlib import Path
import sys
import jax.numpy as jnp
import jax


# Add parent directory to path to import core module
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.layers.linear import Dropout, Linear
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


def test_unit_dropout_layer():
    """🔬 Test Dropout layer implementation."""
    print("🔬 Unit Test: Dropout Layer...")

    # Test dropout creation
    dropout = Dropout(0.5)
    assert dropout.p == 0.5

    # Test inference mode (should pass through unchanged)
    x = Tensor([1, 2, 3, 4])
    y_inference = dropout.forward(x, training=False)
    assert jnp.array_equal(x.data, y_inference.data), "Inference should pass through unchanged"

    # Test training mode with zero dropout (should pass through unchanged)
    dropout_zero = Dropout(0.0)
    y_zero = dropout_zero.forward(x, training=True)
    assert jnp.array_equal(x.data, y_zero.data), "Zero dropout should pass through unchanged"

    # Test training mode with full dropout (should zero everything)
    dropout_full = Dropout(1.0)
    y_full = dropout_full.forward(x, training=True)
    assert jnp.allclose(y_full.data, 0), "Full dropout should zero everything"

    # Test training mode with partial dropout
    # Note: This is probabilistic, so we test statistical properties
    key = jax.random.PRNGKey(42)
    # jnp.random.seed(42)  # For reproducible test
    x_large = Tensor(jnp.ones((1000,)))  # Large tensor for statistical significance
    y_train = dropout.forward(x_large, training=True)

    # Count non-zero elements (approximately 50% should survive)
    non_zero_count = jnp.count_nonzero(y_train.data)
    expected_survival = 1000 * 0.5
    # Allow 10% tolerance for randomness
    assert 0.4 * 1000 < non_zero_count < 0.6 * 1000, f"Expected ~500 survivors, got {non_zero_count}"

    # Test scaling (surviving elements should be scaled by 1/(1-p) = 2.0)
    surviving_values = y_train.data[y_train.data != 0]
    expected_value = 2.0  # 1.0 / (1 - 0.5)
    assert jnp.allclose(surviving_values, expected_value), f"Surviving values should be {expected_value}"

    # Test no parameters
    params = dropout.parameters()
    assert len(params) == 0, "Dropout should have no parameters"

    # Test invalid probability
    try:
        Dropout(-0.1)
        assert False, "Should raise ValueError for negative probability"
    except ValueError:
        pass

    try:
        Dropout(1.1)
        assert False, "Should raise ValueError for probability > 1"
    except ValueError:
        pass

    print("✅ Dropout layer works correctly!")



if __name__ == "__main__":
    test_unit_linear_layer()
    test_unit_dropout_layer()
