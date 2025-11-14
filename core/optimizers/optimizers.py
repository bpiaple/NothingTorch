from abc import ABC, abstractmethod
import jax.numpy as jnp
from core.tensor import Tensor


class Optimizer(ABC):
    """Base class for all optimizers."""

    def __init__(self, parameters: list[Tensor]):
        """Initialize the optimizer with parameters to optimize."""

        # validate and store parameters
        if not isinstance(parameters, list):
            parameters = list(parameters)

            # Check that parameters require gradients
            for i, param in enumerate(parameters):
                if not isinstance(param, Tensor):
                    raise TypeError(f"Parameter at index {i} is not a Tensor.")
                if not param.requires_grad:
                    raise ValueError(
                        f"Parameter at index {i} does not require gradients."
                    )
        self.parameters = parameters
        self.step_count = 0

    @abstractmethod
    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        for param in self.parameters:
            param.grad = None

    @abstractmethod
    def step(self):
        """update parameters based on their gradients."""
        raise NotImplementedError("This method should be overridden by subclasses.")


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum and weight decay."""

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters)

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self):
        """Update parameters based on their gradients."""
        for i, param in enumerate(self.parameters):
            if param._grad_fn is None:
                continue

            grad = param.grad

            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param.data

                # Update with momentum
            if self.momentum != 0.0:
                if self.momentum_buffer[i] is None:
                    self.momentum_buffer[i] = jnp.zeros_like(param.data)

                self.momentum_buffer[i] = self.momentum * self.momentum_buffer[i] + grad
                grad = self.momentum_buffer[i]

            param.data = param.data - self.lr * grad

        self.step_count += 1

    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        super().zero_grad()
        self.momentum_buffer = [None] * len(self.parameters)

class Adam(Optimizer):
    """Adam optimizer with adaptive learning rates."""
    def __init__(self, parameters: list[Tensor], lr: float = 0.001, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(parameters)

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize first and second moment estimates
        self.m_buffers = [None for _ in self.parameters]
        self.v_buffers = [None for _ in self.parameters]

    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        super().zero_grad()
        self.m_buffers = [None for _ in self.parameters]
        self.v_buffers = [None for _ in self.parameters]

    def step(self):
        """Update parameters based on their gradients."""
        for i, param in enumerate(self.parameters):
            if param._grad_fn is None:
                continue

            grad = param.grad

            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param.data

            # Initialize moment buffers if they are None
            if self.m_buffers[i] is None:
                self.m_buffers[i] = jnp.zeros_like(param.data)
            if self.v_buffers[i] is None:
                self.v_buffers[i] = jnp.zeros_like(param.data)

            # Update biased first moment estimate
            self.m_buffers[i] = self.betas[0] * self.m_buffers[i] + (1 - self.betas[0]) * grad
            # Update biased second moment estimate
            self.v_buffers[i] = self.betas[1] * self.v_buffers[i] + (1 - self.betas[1]) * (grad ** 2)

            # Compute bias-corrected first and second moment estimates
            m_hat = self.m_buffers[i] / (1 - self.betas[0] ** (self.step_count + 1))
            v_hat = self.v_buffers[i] / (1 - self.betas[1] ** (self.step_count + 1))

            # Update parameters
            param.data = param.data - self.lr * m_hat / (jnp.sqrt(v_hat) + self.eps)

        self.step_count += 1

class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay."""
    def __init__(self, parameters: list[Tensor], lr: float = 0.001, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(parameters)

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize first and second moment estimates
        self.m_buffers = [None for _ in self.parameters]
        self.v_buffers = [None for _ in self.parameters]

    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        super().zero_grad()
        self.m_buffers = [None for _ in self.parameters]
        self.v_buffers = [None for _ in self.parameters]
        
    def step(self):
        """Update parameters based on their gradients."""
        for i, param in enumerate(self.parameters):
            if param._grad_fn is None:
                continue

            grad = param.grad

            # Initialize moment buffers if they are None
            if self.m_buffers[i] is None:
                self.m_buffers[i] = jnp.zeros_like(param.data)
            if self.v_buffers[i] is None:
                self.v_buffers[i] = jnp.zeros_like(param.data)

            # Update biased first moment estimate
            self.m_buffers[i] = self.betas[0] * self.m_buffers[i] + (1 - self.betas[0]) * grad
            # Update biased second moment estimate
            self.v_buffers[i] = self.betas[1] * self.v_buffers[i] + (1 - self.betas[1]) * (grad ** 2)

            # Compute bias-corrected first and second moment estimates
            m_hat = self.m_buffers[i] / (1 - self.betas[0] ** (self.step_count + 1))
            v_hat = self.v_buffers[i] / (1 - self.betas[1] ** (self.step_count + 1))

            # Update parameters with decoupled weight decay
            param.data = param.data - self.lr * (m_hat / (jnp.sqrt(v_hat) + self.eps) + self.weight_decay * param.data)

        self.step_count += 1
