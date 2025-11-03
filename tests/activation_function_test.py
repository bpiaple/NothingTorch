from pathlib import Path
import sys
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tensor import Tensor
from core.activations.activations import GELU, ReLU, Sigmoid, Softmax, Tanh


def test_unit_tanh():
    """🔬 Test Tanh implementation."""
    print("🔬 Unit Test: Tanh...")

    tanh = Tanh()

    # Test zero
    x = Tensor([0.0])
    result = tanh.forward(x)
    assert jnp.allclose(result.data, jnp.array([0.0])), f"tanh(0) should be 0, got {result.data}"

    # Test range property - all outputs should be in (-1, 1)
    x = Tensor([-10, -1, 0, 1, 10])
    result = tanh.forward(x)
    assert jnp.all(result.data >= -1) and jnp.all(result.data <= 1), "All tanh outputs should be in [-1, 1]"

    # Test symmetry: tanh(-x) = -tanh(x)
    x = Tensor([2.0])
    pos_result = tanh.forward(x)
    x_neg = Tensor([-2.0])
    neg_result = tanh.forward(x_neg)
    assert jnp.allclose(pos_result.data, -neg_result.data), "tanh should be symmetric: tanh(-x) = -tanh(x)"

    # Test extreme values
    x = Tensor([-1000, 1000])
    result = tanh.forward(x)
    assert jnp.allclose(result.data[0], -1, atol=1e-10), "tanh(-∞) should approach -1"
    assert jnp.allclose(result.data[1], 1, atol=1e-10), "tanh(+∞) should approach 1"

    print("✅ Tanh works correctly!")


def test_unit_gelu():
    """🔬 Test GELU implementation."""
    print("🔬 Unit Test: GELU...")

    gelu = GELU()

    # Test zero (should be approximately 0)
    x = Tensor([0.0])
    result = gelu.forward(x)
    assert jnp.allclose(result.data, jnp.array([0.0]), atol=1e-10), f"GELU(0) should be ≈0, got {result.data}"

    # Test positive values (should be roughly preserved)
    x = Tensor([1.0])
    result = gelu.forward(x)
    assert result.data[0] > 0.8, f"GELU(1) should be ≈0.84, got {result.data[0]}"

    # Test negative values (should be small but not zero)
    x = Tensor([-1.0])
    result = gelu.forward(x)
    assert result.data[0] < 0 and result.data[0] > -0.2, f"GELU(-1) should be ≈-0.16, got {result.data[0]}"

    # Test smoothness property (no sharp corners like ReLU)
    x = Tensor([-0.001, 0.0, 0.001])
    result = gelu.forward(x)
    # Values should be close to each other (smooth)
    diff1 = abs(result.data[1] - result.data[0])
    diff2 = abs(result.data[2] - result.data[1])
    assert diff1 < 0.01 and diff2 < 0.01, "GELU should be smooth around zero"

    print("✅ GELU works correctly!")


def test_unit_softmax():
    """🔬 Test Softmax implementation."""
    print("🔬 Unit Test: Softmax...")

    softmax = Softmax()

    # Test basic probability properties
    x = Tensor([1, 2, 3])
    result = softmax.forward(x)

    # Should sum to 1
    assert jnp.allclose(jnp.sum(result.data), 1.0), f"Softmax should sum to 1, got {jnp.sum(result.data)}"

    # All values should be positive
    assert jnp.all(result.data > 0), "All softmax values should be positive"

    # All values should be less than 1
    assert jnp.all(result.data < 1), "All softmax values should be less than 1"

    # Largest ijnput should get largest output
    max_ijnput_idx = jnp.argmax(x.data)
    max_output_idx = jnp.argmax(result.data)
    assert max_ijnput_idx == max_output_idx, "Largest ijnput should get largest softmax output"

    # Test numerical stability with large numbers
    x = Tensor([1000, 1001, 1002])  # Would overflow without max subtraction
    result = softmax.forward(x)
    assert jnp.allclose(jnp.sum(result.data), 1.0), "Softmax should handle large numbers"
    assert not jnp.any(jnp.isnan(result.data)), "Softmax should not produce NaN"
    assert not jnp.any(jnp.isinf(result.data)), "Softmax should not produce infinity"

    # Test with 2D tensor (batch dimension)
    x = Tensor([[1, 2], [3, 4]])
    result = softmax.forward(x, dim=-1)  # Softmax along last dimension
    assert result.shape == (2, 2), "Softmax should preserve ijnput shape"
    # Each row should sum to 1
    row_sums = jnp.sum(result.data, axis=-1)
    assert jnp.allclose(row_sums, jnp.array([1.0, 1.0])), "Each row should sum to 1"

    print("✅ Softmax works correctly!")

def test_unit_sigmoid():
    """🔬 Test Sigmoid implementation."""
    print("🔬 Unit Test: Sigmoid...")

    sigmoid = Sigmoid()

    # Test basic cases
    x = Tensor([0.0])
    result = sigmoid.forward(x)
    assert jnp.allclose(result.data, jnp.array([0.5])), f"sigmoid(0) should be 0.5, got {result.data}"

    # Test range property - all outputs should be in (0, 1)
    x = Tensor([-10, -1, 0, 1, 10])
    result = sigmoid.forward(x)
    assert jnp.all(result.data > 0) and jnp.all(result.data < 1), "All sigmoid outputs should be in (0, 1)"

    # Test specific values
    x = Tensor([-1000, 1000])  # Extreme values
    result = sigmoid.forward(x)
    assert jnp.allclose(result.data[0], 0, atol=1e-10), "sigmoid(-∞) should approach 0"
    assert jnp.allclose(result.data[1], 1, atol=1e-10), "sigmoid(+∞) should approach 1"

    print("✅ Sigmoid works correctly!")


def test_unit_relu():
    """🔬 Test ReLU implementation."""
    print("🔬 Unit Test: ReLU...")

    relu = ReLU()

    # Test mixed positive/negative values
    x = Tensor([-2, -1, 0, 1, 2])
    result = relu.forward(x)
    expected = [0, 0, 0, 1, 2]
    assert jnp.allclose(result.data, jnp.array(expected)), f"ReLU failed, expected {expected}, got {result.data}"

    # Test all negative
    x = Tensor([-5, -3, -1])
    result = relu.forward(x)
    assert jnp.allclose(result.data, jnp.array([0, 0, 0])), "ReLU should zero all negative values"

    # Test all positive
    x = Tensor([1, 3, 5])
    result = relu.forward(x)
    assert jnp.allclose(result.data, jnp.array([1, 3, 5])), "ReLU should preserve all positive values"

    # Test sparsity property
    x = Tensor([-1, -2, -3, 1])
    result = relu.forward(x)
    zeros = jnp.sum(result.data == 0)
    assert zeros == 3, f"ReLU should create sparsity, got {zeros} zeros out of 4"

    print("✅ ReLU works correctly!")


def test_module():
    """
    Comprehensive test of entire module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_sigmoid()
    test_unit_relu()
    test_unit_tanh()
    test_unit_gelu()
    test_unit_softmax()

    print("\nRunning integration scenarios...")

    # Test 1: All activations preserve tensor properties
    print("🔬 Integration Test: Tensor property preservation...")
    test_data = Tensor([[1, -1], [2, -2]])  # 2D tensor

    activations = [Sigmoid(), ReLU(), Tanh(), GELU()]
    for activation in activations:
        result = activation.forward(test_data)
        assert result.shape == test_data.shape, f"Shape not preserved by {activation.__class__.__name__}"
        assert isinstance(result, Tensor), f"Output not Tensor from {activation.__class__.__name__}"

    print("✅ All activations preserve tensor properties!")

    # Test 2: Softmax works with different dimensions
    print("🔬 Integration Test: Softmax dimension handling...")
    data_3d = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])  # (2, 2, 3)
    softmax = Softmax()

    # Test different dimensions
    result_last = softmax(data_3d, dim=-1)
    assert result_last.shape == (2, 2, 3), "Softmax should preserve shape"

    # Check that last dimension sums to 1
    last_dim_sums = jnp.sum(result_last.data, axis=-1)
    assert jnp.allclose(last_dim_sums, 1.0), "Last dimension should sum to 1"

    print("✅ Softmax handles different dimensions correctly!")

    # Test 3: Activation chaining (simulating neural network)
    print("🔬 Integration Test: Activation chaining...")

    # Simulate: Ijnput → Linear → ReLU → Linear → Softmax (like a simple network)
    x = Tensor([[-1, 0, 1, 2]])  # Batch of 1, 4 features

    # Apply ReLU (hidden layer activation)
    relu = ReLU()
    hidden = relu.forward(x)

    # Apply Softmax (output layer activation)
    softmax = Softmax()
    output = softmax.forward(hidden)

    # Verify the chain
    assert hidden.data[0, 0] == 0, "ReLU should zero negative ijnput"
    assert jnp.allclose(jnp.sum(output.data), 1.0), "Final output should be probability distribution"

    print("✅ Activation chaining works correctly!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 02")

# Run comprehensive module test
if __name__ == "__main__":
    test_module()