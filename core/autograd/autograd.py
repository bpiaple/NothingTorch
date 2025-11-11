from abc import ABC, abstractmethod
from typing import Tuple  # Changed from ast to typing
import jax.numpy as jnp

from core.tensor import Tensor


class Function(ABC):
    """Base class for all differentiable functions."""

    def __init__(self, *tensors):
        """Initialize the Function with input tensors.
        
        Args:
            *tensors: Input tensors to the function.
        """
        self.saved_tensors = tensors
        self.next_functions = []

        # Build the computational graph
        for tensor in tensors:
            if isinstance(tensor, Tensor) and tensor.requires_grad:
                if tensor._grad_fn is not None:
                    self.next_functions.append(tensor._grad_fn)

    @abstractmethod
    def apply(self, grad_output: Tensor) -> Tuple[Tensor]:
        """Apply the function to compute the gradient.
        
        Args:
            grad_output (Tensor): Gradient of the output with respect to some scalar value.
        
        Returns: 
            Tensor: Tuple[Tensor] Gradient of the input tensors.

        ***Must be implemented in subclasses.***
        """
        raise NotImplementedError("Function.apply must be implemented in subclasses.")
    

class AddBackward(Function):
    """Gradient computation for addition operation.
    
    **Mathematical Rule:** If z = a + b, then ∂z/∂a = 1 and ∂z/∂b = 1***
    """

    def apply(self, grad_output: Tensor) -> Tuple[Tensor]:
        """Compute gradients for addition operation.
        
        Args:
            grad_output (Tensor): Gradient of the output.
        Returns:
            Tuple[Tensor]: Gradients with respect to each input tensor.
        """
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output

        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output
        
        return grad_a, grad_b
    

class MulBackward(Function):
    """Gradient computation for multiplication operation.
    
    ***Mathematical Rule:** If z = a * b, then ∂z/∂a = b and ∂z/∂b = a***
    """

    def apply(self, grad_output) -> Tuple[Tensor]:
        """Compute gradients for multiplication operation.
        
        Args:
            grad_output (Tensor): Gradient of the output.
        Returns:
            Tuple[Tensor]: Gradients with respect to each input tensor.
        """
        a, b = self.saved_tensors
        grad_a, grad_b = None, None

        if isinstance(a, Tensor) and a.requires_grad:
            if isinstance(b, Tensor):
                grad_a = grad_output * b.data
            else:
                grad_a = grad_output * b

        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output * a.data
        
        return grad_a, grad_b


class MatmulBackward(Function):
    """Gradient computation for matrix multiplication operation.
    
    ***Mathematical Rule:** If C = A @ B, then ∂C/∂A = B^T and ∂C/∂B = A^T***
    """

    def apply(self, grad_output: Tensor) -> Tuple[Tensor]:
        """Compute gradients for matrix multiplication operation.
        
        Args:
            grad_output (Tensor): Gradient of the output.
        Returns:
            Tuple[Tensor]: Gradients with respect to each input tensor.
        """
        a, b = self.saved_tensors
        grad_a, grad_b = None, None

        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = jnp.dot(grad_output, b.data.T)

        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = jnp.dot(a.data.T, grad_output)
        
        return grad_a, grad_b
    

class SumBackward(Function):
    """Gradient computation for summation operation.
    
    ***Mathematical Rule:** If y = sum(x), then ∂y/∂x = 1 (broadcasted to x's shape)***
    """

    def apply(self, grad_output: Tensor) -> Tuple[Tensor]:
        """Compute gradients for summation operation.
        
        Args:
            grad_output (Tensor): Gradient of the output.
        Returns:
            Tuple[Tensor]: Gradient with respect to the input tensor.
        """
        tensor, = self.saved_tensors
        grad_x = None

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            grad_x = jnp.ones_like(tensor.data) * grad_output
        
        return (grad_x,)


class ReLUBackward(Function):
    """Gradient computation for ReLU activation function.
    
    ***Mathematical Rule:** If y = ReLU(x), then ∂y/∂x = 1 if x > 0 else 0***
    """

    def apply(self, grad_output: Tensor) -> Tuple[Tensor]:
        """Compute gradients for ReLU operation.

        ReLU: f(x) = max(0, x)
        Derivative: f'(x) = 1 if x > 0, else 0
        
        Args:
            grad_output (Tensor): Gradient of the output.
        Returns:
            Tuple[Tensor]: Gradient with respect to the input tensor.
        """
        tensor, = self.saved_tensors
        grad_x = None

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            relu_grad = jnp.where(tensor.data > 0, 1, 0)
            grad_x = grad_output * relu_grad
        
        return (grad_x,)
    

class SigmoidBackward(Function):
    """Gradient computation for Sigmoid activation function.
    
    ***Mathematical Rule:** If y = Sigmoid(x), then ∂y/∂x = y * (1 - y)***
    """

    def apply(self, grad_output: Tensor) -> Tuple[Tensor]:
        """Compute gradients for Sigmoid operation.

        Sigmoid: f(x) = 1 / (1 + exp(-x))
        Derivative: f'(x) = f(x) * (1 - f(x))
        
        Args:
            grad_output (Tensor): Gradient of the output.
        Returns:
            Tuple[Tensor]: Gradient with respect to the input tensor.
        """
        tensor, = self.saved_tensors
        grad_x = None

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            sigmoid_output = 1 / (1 + jnp.exp(-tensor.data))
            sigmoid_grad = sigmoid_output * (1 - sigmoid_output)
            grad_x = grad_output * sigmoid_grad
        
        return (grad_x,)
    
class MSEBackward(Function):
    """Gradient computation for Mean Squared Error loss function.

    MSE: L = mean((predictions - targets)²)
    Derivative: ∂L/∂predictions = 2 * (predictions - targets) / N
    """
    def __init__(self, y_pred: Tensor, y_true: Tensor):
        """Initialize MSEBackward with predicted and true tensors.
        
        Args:
            y_pred (Tensor): Predicted tensor.
            y_true (Tensor): True tensor.
        """
        super().__init__(y_pred)
        self.targets_data = y_true.data
        self.num_samples = jnp.size(y_true.data)


    def apply(self, grad_output: Tensor) -> Tuple[Tensor]:
        """Compute gradients for Mean Squared Error operation.
        
        Args:
            grad_output (Tensor): Gradient of the output.
        Returns:
            Tuple[Tensor]: Gradient with respect to the predicted tensor.
        """
        y_pred, = self.saved_tensors
        grad_y_pred = None

        if isinstance(y_pred, Tensor) and y_pred.requires_grad:
            grad_y_pred = (2 / self.num_samples) * (y_pred.data - self.targets_data)
            grad_y_pred = grad_y_pred * grad_output
        
        return (grad_y_pred,)
    

class BCEBackward(Function):
    """Gradient computation for Binary Cross Entropy loss function.

    BCE: L = -[y*log(p) + (1-y)*log(1-p)]
    Derivative: ∂L/∂y_pred = -∂L/∂p = (p - y) / (p*(1-p)*N)
    """
    def __init__(self, y_pred: Tensor, y_true: Tensor):
        """Initialize BCEBackward with predicted and true tensors.
        
        Args:
            y_pred (Tensor): Predicted tensor.
            y_true (Tensor): True tensor.
        """
        super().__init__(y_pred)
        self.targets_data = y_true.data
        self.num_samples = jnp.size(y_true.data)


    def apply(self, grad_output: Tensor) -> Tuple[Tensor]:
        """Compute gradients for Binary Cross Entropy operation.
        
        Args:
            grad_output (Tensor): Gradient of the output.
        Returns:
            Tuple[Tensor]: Gradient with respect to the predicted tensor.
        """
        y_pred, = self.saved_tensors
        grad_y_pred = None

        if isinstance(y_pred, Tensor) and y_pred.requires_grad:
            eps = 1e-7  # To avoid division by zero
            p = jnp.clip(y_pred.data, eps, 1 - eps)
            y = self.targets_data
            
            grad_y_pred = -(p - y) / (p * (1 - p))
            grad_y_pred = grad_y_pred * self.num_samples
            grad_y_pred = grad_y_pred * grad_output
        
        return (grad_y_pred,)
    

class CrossEntropyBackward(Function):
    """Gradient computation for Cross Entropy loss function.

    CE: L = -sum(y_true * log(softmax(y_pred)))
    Derivative: ∂L/∂y_pred = softmax(y_pred) - y_true
    """
    def __init__(self, logits: Tensor, y_true: Tensor):
        """Initialize CrossEntropyBackward with predicted and true tensors.
        
        Args:
            logits (Tensor): Predicted tensor.
            y_true (Tensor): True tensor.
        """
        super().__init__(logits)
        self.targets_data = y_true.data.astype(jnp.int32)
        self.batch_size = y_true.data.shape[0]
        self.num_classes = logits.data.shape[1]


    def apply(self, grad_output: Tensor) -> Tuple[Tensor]:
        """Compute gradients for Cross Entropy operation.
        
        Args:
            grad_output (Tensor): Gradient of the output.
        Returns:
            Tuple[Tensor]: Gradient with respect to the predicted tensor.
        """
        logits, = self.saved_tensors
        grad_y_pred = None

        if isinstance(logits, Tensor) and logits.requires_grad:
            logits_data = logits.data
            max_logits = jnp.max(logits_data, axis=1, keepdims=True)
            exp_logits = jnp.exp(logits_data - max_logits)
            softmax = exp_logits / jnp.sum(exp_logits, axis=1, keepdims=True)

            # Create one-hot encoding of targets
            one_hot_targets = jnp.zeros((self.batch_size, self.num_classes), dtype=jnp.int32)
            one_hot_targets[jnp.arange(self.batch_size), self.targets_data] = 1.0
            
            # Compute gradient (Softmax - OneHot) / batch_size
            grad_y_pred = (softmax - one_hot_targets) / self.batch_size

            grad_y_pred = grad_y_pred * grad_output
        return (grad_y_pred,)
    

def enable_autograd():
    """Enable autograd functionality in the framework."""
    Tensor.enable_autograd()
    Function.enable_autograd()
    