import jax.numpy as jnp

from core.tensor import Tensor


class Softmax:
    """
    Softmax activation function.
    """
    
    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        exp_logits = jnp.exp(x.data - jnp.max(x.data, axis=dim, keepdims=True))
        softmax = exp_logits / jnp.sum(exp_logits, axis=dim, keepdims=True)

        return Tensor(softmax, requires_grad=x.requires_grad)

    def __call__(self, x: Tensor, dim: int = -1) -> Tensor:
        return self.forward(x, dim=dim)

    def backward(self, grad: Tensor) -> Tensor:
        pass


class Sigmoid():
    """
    Sigmoid activation function.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply sigmoid activation element-wise.
        """
        sigmoid = 1.0 / (1.0 + jnp.exp(-x.data))

        return Tensor(sigmoid, requires_grad=x.requires_grad)

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass for Sigmoid.
        """
        pass


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

    
class ReLU:
    """
    ReLU activation function.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply ReLU activation element-wise.
        """
        return Tensor(jnp.maximum(0, x.data), requires_grad=x.requires_grad)

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass for ReLU.
        """
        pass



class GELU:
    """
    Gaussian Error Linear Unit (GELU) activation function.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Applies the Gaussian Error Linear Unit (GELU) activation function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying GELU.
        """

        # Compute the sigmoid approximation of GELU
        sigmoid_approx = 1.0 / (1.0 + jnp.exp(-1.702 * x.data))
        return Tensor(x.data * sigmoid_approx, requires_grad=x.requires_grad)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def backward(x, grad: Tensor) -> Tensor:
        """
        Backward pass for GELU.
        """
        
        pass
