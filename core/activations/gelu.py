import jax.numpy as jnp

from core.tensor import Tensor


class GELU:
    """
    Gaussian Error Linear Unit (GELU) activation function.
    """

    def forward(x):
        """Applies the Gaussian Error Linear Unit (GELU) activation function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying GELU.
        """

        # Compute the sigmoid approximation of GELU
        sigmoid_approx = 1.0 / (1.0 + jnp.exp(-1.702 * x.data))
        return Tensor(x.data * sigmoid_approx, requires_grad=x.requires_grad)

    @staticmethod
    def backward(x, grad: Tensor) -> Tensor:
        """
        Backward pass for GELU.
        """
        
        pass
