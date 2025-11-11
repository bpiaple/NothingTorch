from pathlib import Path
import sys
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent))


from core.autograd.autograd import AddBackward, MatmulBackward, MulBackward
from core.tensor import Tensor


def test_unit_function_classes():
    """🔬 Test Function classes."""
    print("🔬 Unit Test: Function Classes...")

    # Test AddBackward
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    add_func = AddBackward(a, b)
    grad_output = jnp.array([1, 1, 1])
    grad_a, grad_b = add_func.apply(grad_output)
    assert jnp.allclose(grad_a, grad_output), f"AddBackward grad_a failed: {grad_a}"
    assert jnp.allclose(grad_b, grad_output), f"AddBackward grad_b failed: {grad_b}"

    # Test MulBackward
    mul_func = MulBackward(a, b)
    grad_a, grad_b = mul_func.apply(grad_output)
    assert jnp.allclose(grad_a, b.data), f"MulBackward grad_a failed: {grad_a}"
    assert jnp.allclose(grad_b, a.data), f"MulBackward grad_b failed: {grad_b}"

    # Test MatmulBackward
    a_mat = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b_mat = Tensor([[5, 6], [7, 8]], requires_grad=True)
    matmul_func = MatmulBackward(a_mat, b_mat)
    grad_output = jnp.ones((2, 2))
    grad_a, grad_b = matmul_func.apply(grad_output)
    assert grad_a.shape == a_mat.shape, f"MatmulBackward grad_a shape: {grad_a.shape}"
    assert grad_b.shape == b_mat.shape, f"MatmulBackward grad_b shape: {grad_b.shape}"

    print("✅ Function classes work correctly!")


if __name__ == "__main__":
    test_unit_function_classes()
