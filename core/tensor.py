import jax.numpy as jnp


class Tensor(object):
    """
    docstring
    """

    def __init__(self, data, requires_grad=False):
        """
        Create a new tensor from data

        Args:
            data (jnp.ndarray): The data to initialize the tensor.
            requires_grad (bool, optional): Whether to track gradients. Defaults to False.
        """

        # Core tensor data * always present
        self.data = jnp.array(data, dtype=jnp.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.requires_grad = requires_grad

        # Gradient features
        self.requires_grad = requires_grad
        self.grad = None

    def __repr__(self):
        """
        String representation of the tensor for debugging.
        """
        grad_info = (
            f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        )
        return f"Tensor(shape={self.shape}, size={self.size}{grad_info})"

    def __str__(self):
        """
        String conversion of the tensor for display.
        """
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def numpy(self):
        """
        Return the underlying Numpy array.
        """
        return self.data

    def __add__(self, other):
        """
        Element-wise addition of two tensors.
        """
        if isinstance(other, Tensor):
            return Tensor(
                self.data + other.data,
                requires_grad=self.requires_grad or other.requires_grad,
            )
        else:
            return Tensor(self.data + other, requires_grad=self.requires_grad)

    def __sub__(self, other):
        """
        Element-wise subtraction of two tensors.
        """
        if isinstance(other, Tensor):
            return Tensor(
                self.data - other.data,
                requires_grad=self.requires_grad or other.requires_grad,
            )
        else:
            return Tensor(self.data - other, requires_grad=self.requires_grad)

    def __mul__(self, other):
        """
        Element-wise multiplication of two tensors.
        """
        if isinstance(other, Tensor):
            return Tensor(
                self.data * other.data,
                requires_grad=self.requires_grad or other.requires_grad,
            )
        else:
            return Tensor(self.data * other, requires_grad=self.requires_grad)

    def __truediv__(self, other):
        """
        Element-wise division of two tensors.
        """
        if isinstance(other, Tensor):
            return Tensor(
                self.data / other.data,
                requires_grad=self.requires_grad or other.requires_grad,
            )
        else:
            return Tensor(self.data / other, requires_grad=self.requires_grad)

    def matmul(self, other):
        """
        Matrix multiplication of two tensors.
        """
        if not isinstance(other, Tensor):
            raise ValueError(
                f"The other operand must be a Tensor, but got {type(other).__name__}."
            )

        # Handle edge case for 1D tensors
        if self.shape == () or other.shape == ():
            return Tensor(
                self.data * other.data,
                requires_grad=self.requires_grad or other.requires_grad,
            )

        # Handle 0D tensors (scalars)
        if self.data.ndim == 0 or other.data.ndim == 0:
            return Tensor(
                self.data * other.data,
                requires_grad=self.requires_grad or other.requires_grad,
            )

        # Check for compatible shapes
        if self.data.ndim >= 2 and other.data.ndim >= 2:
            if self.data.shape[-1] != other.data.shape[-2]:
                raise ValueError(
                    f"Cannot perform matrix multiplication: {self.shape} @ {other.shape}. "
                    f"Inner dimensions must match: {self.shape[-1]} ≠ {other.shape[-2]}. "
                    f"💡 HINT: For (M,K) @ (K,N) → (M,N), the K dimensions must be equal."
                )

        # Vector @ Matrix
        elif self.data.ndim == 1 and other.data.ndim == 2:
            if self.data.shape[0] != other.data.shape[0]:
                raise ValueError(
                    f"Cannot perform matrix multiplication: {self.shape} @ {other.shape}. "
                    f"Inner dimensions must match: {self.shape[0]} ≠ {other.shape[0]}. "
                    f"💡 HINT: For (K,) @ (K,N) → (N,), the K dimensions must be equal."
                )

        # Matrix @ Vector
        elif self.data.ndim == 2 and other.data.ndim == 1:
            if self.data.shape[1] != other.data.shape[0]:
                raise ValueError(
                    f"Cannot perform matrix multiplication: {self.shape} @ {other.shape}. "
                    f"Inner dimensions must match: {self.shape[1]} ≠ {other.shape[0]}. "
                    f"💡 HINT: For (M,K) @ (K,) → (M,), the K dimensions must be equal."
                )

        # Perform optimized multiplication using jnp.matmul not jnp.dot
        result_data = jnp.matmul(self.data, other.data)
        return Tensor(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
    
    def reshape(self, *shape):
        """
        Reshape the tensor to the specified shape.
        """
        
        # Handle both reshape((2,3)) and reshape(2,3)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape

        # Handle -1 in shape for automatic dimension inference
        if -1 in new_shape:
            if new_shape.count(-1) > -1:
                raise ValueError("Can only specify one unknown dimension.")
            
            # Calculate the unknown dimension
            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim

            unknown_dim = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)

        # Validate total elements remain the same
        if jnp.prod(jnp.array(new_shape)) != self.size:
            raise ValueError(
                f"Cannot reshape tensor of size {self.size} into shape {new_shape}."
            )
        
        # Reshape data
        reshaped_data = jnp.reshape(self.data, new_shape)
        return Tensor(reshaped_data, requires_grad=self.requires_grad)
    
    def transpose(self, dim0=None, dim1=None):
        """
        Transpose two dimensions of the tensor.
        If no dimensions are specified, reverse the dimensions.
        """
        if dim0 is None and dim1 is None:
            if self.data.ndim < 2:
                # For 0D and 1D tensors, return a copy
                return Tensor(self.data.copy(), requires_grad=self.requires_grad)
            else:
                axes = list(range(self.data.ndim))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = jnp.transpose(self.data, axes=axes)

        else:
            # Specific dimensions to swap
            if dim0 is None or dim1 is None:
                raise ValueError("Both dim0 and dim1 must be specified together.")
            
            # validate dimensions exist
            if dim0 >= self.data.ndim or dim1 >= self.data.ndim or dim0 <0 or dim1 <0:
                raise ValueError(
                    f"Dimension out of range (got dim0={dim0}, dim1={dim1} for tensor with {self.data.ndim} dimensions)."
                )
            
            axes = list(range(self.data.ndim))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = jnp.transpose(self.data, axes=axes)

        return Tensor(transposed_data, requires_grad=self.requires_grad)
    
    def sum(self, axis=None, keepdims=False):
        """
        Sum of tensor elements over a given axis.
        """
        summed_data = jnp.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor(summed_data, requires_grad=self.requires_grad)

    def mean(self, axis=None, keepdims=False):
        """
        Mean of tensor elements over a given axis.
        """
        mean_data = jnp.mean(self.data, axis=axis, keepdims=keepdims)
        return Tensor(mean_data, requires_grad=self.requires_grad)
    
    def max(self, axis=None, keepdims=False):
        """
        Maximum of tensor elements over a given axis.
        """
        max_data = jnp.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(max_data, requires_grad=self.requires_grad)
    
    def min(self, axis=None, keepdims=False):
        """
        Minimum of tensor elements over a given axis.
        """
        min_data = jnp.min(self.data, axis=axis, keepdims=keepdims)
        return Tensor(min_data, requires_grad=self.requires_grad)
    
    def backward(self):
        """
        Compute gradients for the tensor.
        """
        pass  # Placeholder for gradient computation logic
    