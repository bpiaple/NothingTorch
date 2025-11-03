import jax.numpy as jnp
from core.tensor import Tensor

class Tanh:
    """
    Tanh activation function.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply tanh activation element-wise.
        """
        return Tensor(jnp.tanh(x.data), requires_grad=x.requires_grad)

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass for Tanh.
        """
        pass

    