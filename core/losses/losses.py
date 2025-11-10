import jax.numpy as jnp
from core.tensor import Tensor

def log_softmax(predictions: Tensor, dim: int = -1) -> Tensor:
    """Compute the log softmax of the predictions along the specified dimension.

    Args:
        predictions (Tensor): The input tensor containing raw prediction scores (logits).
        dim (int): The dimension along which to compute the log softmax.

    Returns:
        Tensor: The log softmax of the input tensor.
    """
    max_vals = jnp.max(predictions.data, axis=dim, keepdims=True)

    shifted = predictions.data - max_vals

    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted), axis=dim, keepdims=True))

    result = shifted - log_sum_exp

    return Tensor(result)

class MSELoss :
    """Mean Squared Error Loss implementation."""

    def __init__(self):
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute the Mean Squared Error loss.

        Args:
            predictions (list of float): The predicted values.
            targets (list of float): The ground truth values.

        Returns:
            float: The computed MSE loss.
        """
        if predictions.data.shape != targets.data.shape:
            raise ValueError("Predictions and targets must have the same shape.")

        # mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
        mse = jnp.mean((predictions.data - targets.data) ** 2)
        return Tensor(mse)
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)
    
    def backward(self) :
        """Computer the gradient of the loss with respect to the predictions."""
        pass

class CrossEntropyLoss:
    """Cross Entropy Loss implementation."""

    def __init__(self):
        pass
    
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the Cross Entropy Loss."""

        log_probs = log_softmax(logits, dim=-1)

        batch_size = logits.data.shape[0]
        target_indices = targets.data.astype(int)

        selected_log_probs = log_probs.data[jnp.arange(batch_size), target_indices]

        cross_entropy_loss = -jnp.mean(selected_log_probs)
        return Tensor(cross_entropy_loss)

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Allow the instance to be called like a function."""
        return self.forward(logits, targets)
    
    def backward(self):
        """Compute the gradient of the loss with respect to the logits."""
        pass

class BinaryCrossEntropyLoss:
    """Binary Cross Entropy Loss implementation."""

    def __init__(self):
        """Initialize the BinaryCrossEntropyLoss instance."""
        pass

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the Binary Cross Entropy Loss.

        Args:
            logits (Tensor): The predicted logits.
            targets (Tensor): The ground truth binary labels (0 or 1).

        Returns:
            Tensor: The computed BCE loss.
        """
        eps = 1e-7

        clamped_preds = jnp.clip(logits.data, eps, 1 - eps)

        log_preds = jnp.log(clamped_preds)
        log_one_minus_preds = jnp.log(1 - clamped_preds)

        bce_per_sample = - (targets.data * log_preds + (1 - targets.data) * log_one_minus_preds)

        bce_loss = jnp.mean(bce_per_sample)

        return Tensor(bce_loss)

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.forward(logits, targets)
    
    def backward(self):
        """Compute the gradient of the loss with respect to the logits."""
        pass
