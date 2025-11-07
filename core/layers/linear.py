import jax
import jax.numpy as jnp

from core.tensor import Tensor

seed = 42


class Linear:
    """
    A fully connected neural network layer (y = xW + b).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features

        key = jax.random.PRNGKey(seed)
        self.weight = Tensor(
            jax.random.normal(key, (in_features, out_features))
            * jnp.sqrt(1.0 / in_features),
            requires_grad=True,
        )
        if not bias:
            self.bias = None
        else:
            self.bias = Tensor(jnp.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the dense layer (y = xW + b).
        """
        output = x.matmul(self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output

    def __call__(self, x: Tensor) -> Tensor:
        """Allow the layer to be called like a function."""
        return self.forward(x)

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

    def parameters(self):
        """
        Return the list of trainable parameters (weight and bias).
        """
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


class Dropout:
    """
    Dropout layer for regularization.
    """

    def __init__(self, p: float = 0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError("Dropout probability must be in the range [0.0, 1.0]")
        self.p = p
        self.key = jax.random.PRNGKey(42)

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        """
        Forward pass through the dropout layer.
        """
        if not training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return Tensor(jnp.zeros_like(x.data), requires_grad=x.requires_grad)
        
        keep_prob = 1 - self.p
        self.key, subkey = jax.random.split(self.key)
        mask = jax.random.bernoulli(subkey, p=keep_prob, shape=x.shape).astype(jnp.float32)
        
        output = x * mask / keep_prob
        return output

    def __call__(self, x: Tensor, training: bool = False) -> Tensor:
        """Allow the layer to be called like a function."""
        return self.forward(x, training)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"
