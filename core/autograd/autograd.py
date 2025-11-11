from abc import ABC, abstractmethod
from ast import Tuple
from tkinter import NO

from traitlets import Instance

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
        grad_a, grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            if isinstance(b, Tensor):
                grad_a = grad_output * b.data
            else:
                grad_a = grad_output * b

        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output * a.data
        
        return grad_a, grad_b