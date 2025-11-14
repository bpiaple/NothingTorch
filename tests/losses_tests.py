# Add parent directory to path to import core module
from pathlib import Path
import sys
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.losses.losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss, log_softmax
from core.tensor import Tensor


def test_unit_log_softmax():
    """🔬 Test log_softmax numerical stability and correctness."""
    print("🔬 Unit Test: Log-Softmax...")

    # Test basic functionality
    x = Tensor([[1.0, 2.0, 3.0], [0.1, 0.2, 0.9]])
    result = log_softmax(x, dim=-1)

    # Verify shape preservation
    assert result.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {result.shape}"

    # Verify log-softmax properties: exp(log_softmax) should sum to 1
    softmax_result = jnp.exp(result.data)
    row_sums = jnp.sum(softmax_result, axis=-1)
    assert jnp.allclose(row_sums, 1.0, atol=1e-6), f"Softmax doesn't sum to 1: {row_sums}"

    # Test numerical stability with large values
    large_x = Tensor([[100.0, 101.0, 102.0]])
    large_result = log_softmax(large_x, dim=-1)
    assert not jnp.any(jnp.isnan(large_result.data)), "NaN values in result with large inputs"
    assert not jnp.any(jnp.isinf(large_result.data)), "Inf values in result with large inputs"

    print("✅ log_softmax works correctly with numerical stability!")

def test_unit_mse_loss():
    """🔬 Test MSELoss implementation and properties."""
    print("🔬 Unit Test: MSE Loss...")

    loss_fn = MSELoss()

    # Test perfect predictions (loss should be 0)
    predictions = Tensor([1.0, 2.0, 3.0])
    targets = Tensor([1.0, 2.0, 3.0])
    perfect_loss = loss_fn.forward(predictions, targets)
    assert jnp.allclose(perfect_loss.data, 0.0, atol=1e-7), f"Perfect predictions should have 0 loss, got {perfect_loss.data}"

    # Test known case
    predictions = Tensor([1.0, 2.0, 3.0])
    targets = Tensor([1.5, 2.5, 2.8])
    loss = loss_fn.forward(predictions, targets)

    # Manual calculation: ((1-1.5)² + (2-2.5)² + (3-2.8)²) / 3 = (0.25 + 0.25 + 0.04) / 3 = 0.18
    expected_loss = (0.25 + 0.25 + 0.04) / 3
    assert jnp.allclose(loss.data, expected_loss, atol=1e-6), f"Expected {expected_loss}, got {loss.data}"

    # Test that loss is always non-negative
    seed = 42
    key = jax.random.PRNGKey(seed)
    
    random_pred = Tensor(jax.random.normal(key, (10,)))
    random_target = Tensor(jax.random.normal(key, (10,)))
    random_loss = loss_fn.forward(random_pred, random_target)
    assert random_loss.data >= 0, f"MSE loss should be non-negative, got {random_loss.data}"

    print("✅ MSELoss works correctly!")

def test_unit_cross_entropy_loss():
    """🔬 Test CrossEntropyLoss implementation and properties."""
    print("🔬 Unit Test: Cross-Entropy Loss...")

    loss_fn = CrossEntropyLoss()

    # Test perfect predictions (should have very low loss)
    perfect_logits = Tensor([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])  # Very confident predictions
    targets = Tensor([0, 1])  # Matches the confident predictions
    perfect_loss = loss_fn.forward(perfect_logits, targets)
    assert perfect_loss.data < 0.01, f"Perfect predictions should have very low loss, got {perfect_loss.data}"

    # Test uniform predictions (should have loss ≈ log(num_classes))
    uniform_logits = Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])  # Equal probabilities
    uniform_targets = Tensor([0, 1])
    uniform_loss = loss_fn.forward(uniform_logits, uniform_targets)
    expected_uniform_loss = jnp.log(3)  # log(3) ≈ 1.099 for 3 classes
    assert jnp.allclose(uniform_loss.data, expected_uniform_loss, atol=0.1), f"Uniform predictions should have loss ≈ log(3) = {expected_uniform_loss:.3f}, got {uniform_loss.data:.3f}"

    # Test that wrong confident predictions have high loss
    wrong_logits = Tensor([[10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]])  # Confident but wrong
    wrong_targets = Tensor([1, 1])  # Opposite of confident predictions
    wrong_loss = loss_fn.forward(wrong_logits, wrong_targets)
    assert wrong_loss.data > 5.0, f"Wrong confident predictions should have high loss, got {wrong_loss.data}"

    # Test numerical stability with large logits
    large_logits = Tensor([[100.0, 50.0, 25.0]])
    large_targets = Tensor([0])
    large_loss = loss_fn.forward(large_logits, large_targets)
    assert not jnp.isnan(large_loss.data), "Loss should not be NaN with large logits"
    assert not jnp.isinf(large_loss.data), "Loss should not be infinite with large logits"

    print("✅ CrossEntropyLoss works correctly!")

def test_unit_binary_cross_entropy_loss():
    """🔬 Test BinaryCrossEntropyLoss implementation and properties."""
    print("🔬 Unit Test: Binary Cross-Entropy Loss...")

    loss_fn = BinaryCrossEntropyLoss()

    # Test perfect predictions
    perfect_predictions = Tensor([0.9999, 0.0001, 0.9999, 0.0001])
    targets = Tensor([1.0, 0.0, 1.0, 0.0])
    perfect_loss = loss_fn.forward(perfect_predictions, targets)
    assert perfect_loss.data < 0.01, f"Perfect predictions should have very low loss, got {perfect_loss.data}"

    # Test worst predictions
    worst_predictions = Tensor([0.0001, 0.9999, 0.0001, 0.9999])
    worst_targets = Tensor([1.0, 0.0, 1.0, 0.0])
    worst_loss = loss_fn.forward(worst_predictions, worst_targets)
    assert worst_loss.data > 5.0, f"Worst predictions should have high loss, got {worst_loss.data}"

    # Test uniform predictions (probability = 0.5)
    uniform_predictions = Tensor([0.5, 0.5, 0.5, 0.5])
    uniform_targets = Tensor([1.0, 0.0, 1.0, 0.0])
    uniform_loss = loss_fn.forward(uniform_predictions, uniform_targets)
    expected_uniform = -jnp.log(0.5)  # Should be about 0.693
    assert jnp.allclose(uniform_loss.data, expected_uniform, atol=0.01), f"Uniform predictions should have loss ≈ {expected_uniform:.3f}, got {uniform_loss.data:.3f}"

    # Test numerical stability at boundaries
    boundary_predictions = Tensor([0.0, 1.0, 0.0, 1.0])
    boundary_targets = Tensor([0.0, 1.0, 1.0, 0.0])
    boundary_loss = loss_fn.forward(boundary_predictions, boundary_targets)
    assert not jnp.isnan(boundary_loss.data), "Loss should not be NaN at boundaries"
    assert not jnp.isinf(boundary_loss.data), "Loss should not be infinite at boundaries"

    print("✅ BinaryCrossEntropyLoss works correctly!")

def compare_loss_behaviors():
    """
    🔬 Compare how different loss functions behave with various prediction patterns.

    This helps students understand when to use each loss function.
    """
    print("🔬 Integration Test: Loss Function Behavior Comparison...")

    # Initialize loss functions
    mse_loss = MSELoss()
    ce_loss = CrossEntropyLoss()
    bce_loss = BinaryCrossEntropyLoss()

    print("\n1. Regression Scenario (House Price Prediction)")
    print("   Predictions: [200k, 250k, 300k], Targets: [195k, 260k, 290k]")
    house_pred = Tensor([200.0, 250.0, 300.0])  # In thousands
    house_target = Tensor([195.0, 260.0, 290.0])
    mse = mse_loss.forward(house_pred, house_target)
    print(f"   MSE Loss: {mse.data:.2f} (thousand²)")

    print("\n2. Multi-Class Classification (Image Recognition)")
    print("   Classes: [cat, dog, bird], Predicted: confident about cat, uncertain about dog")
    # Logits: [2.0, 0.5, 0.1] suggests model is most confident about class 0 (cat)
    image_logits = Tensor([[2.0, 0.5, 0.1], [0.3, 1.8, 0.2]])  # Two samples
    image_targets = Tensor([0, 1])  # First is cat (0), second is dog (1)
    ce = ce_loss.forward(image_logits, image_targets)
    print(f"   Cross-Entropy Loss: {ce.data:.3f}")

    print("\n3. Binary Classification (Spam Detection)")
    print("   Predictions: [0.9, 0.1, 0.7, 0.3] (spam probabilities)")
    spam_pred = Tensor([0.9, 0.1, 0.7, 0.3])
    spam_target = Tensor([1.0, 0.0, 1.0, 0.0])  # 1=spam, 0=not spam
    bce = bce_loss.forward(spam_pred, spam_target)
    print(f"   Binary Cross-Entropy Loss: {bce.data:.3f}")

    print("\n💡 Key Insights:")
    print("   - MSE penalizes large errors heavily (good for continuous values)")
    print("   - Cross-Entropy encourages confident correct predictions")
    print("   - Binary Cross-Entropy balances false positives and negatives")

    return mse.data, ce.data, bce.data
     
def analyze_loss_sensitivity():
    """
    📊 Analyze how sensitive each loss function is to prediction errors.

    This demonstrates the different error landscapes created by each loss.
    """
    print("\n📊 Analysis: Loss Function Sensitivity to Errors...")

    # Create a range of prediction errors for analysis
    true_value = 1.0
    predictions = jnp.linspace(0.1, 1.9, 50)  # From 0.1 to 1.9

    # Initialize loss functions
    mse_loss = MSELoss()
    bce_loss = BinaryCrossEntropyLoss()

    mse_losses = []
    bce_losses = []

    for pred in predictions:
        # MSE analysis
        pred_tensor = Tensor([pred])
        target_tensor = Tensor([true_value])
        mse = mse_loss.forward(pred_tensor, target_tensor)
        mse_losses.append(mse.data)

        # BCE analysis (clamp prediction to valid probability range)
        clamped_pred = max(0.01, min(0.99, pred))
        bce_pred_tensor = Tensor([clamped_pred])
        bce_target_tensor = Tensor([1.0])  # Target is "positive class"
        bce = bce_loss.forward(bce_pred_tensor, bce_target_tensor)
        bce_losses.append(bce.data)

    # Find minimum losses
    min_mse_idx = jnp.argmin(mse_losses)
    min_bce_idx = jnp.argmin(bce_losses)

    print(f"MSE Loss:")
    print(f"  Minimum at prediction = {predictions[min_mse_idx]:.2f}, loss = {mse_losses[min_mse_idx]:.4f}")
    print(f"  At prediction = 0.5: loss = {mse_losses[24]:.4f}")  # Middle of range
    print(f"  At prediction = 0.1: loss = {mse_losses[0]:.4f}")

    print(f"\nBinary Cross-Entropy Loss:")
    print(f"  Minimum at prediction = {predictions[min_bce_idx]:.2f}, loss = {bce_losses[min_bce_idx]:.4f}")
    print(f"  At prediction = 0.5: loss = {bce_losses[24]:.4f}")
    print(f"  At prediction = 0.1: loss = {bce_losses[0]:.4f}")

    print(f"\n💡 Sensitivity Insights:")
    print("   - MSE grows quadratically with error distance")
    print("   - BCE grows logarithmically, heavily penalizing wrong confident predictions")
    print("   - Both encourage correct predictions but with different curvatures")

def analyze_numerical_stability():
    """
    📊 Demonstrate why numerical stability matters in loss computation.

    Shows the difference between naive and stable implementations.
    """
    print("📊 Analysis: Numerical Stability in Loss Functions...")

    # Test with increasingly large logits
    test_cases = [
        ("Small logits", [1.0, 2.0, 3.0]),
        ("Medium logits", [10.0, 20.0, 30.0]),
        ("Large logits", [100.0, 200.0, 300.0]),
        ("Very large logits", [500.0, 600.0, 700.0])
    ]

    print("\nLog-Softmax Stability Test:")
    print("Case                 | Max Input | Log-Softmax Min | Numerically Stable?")
    print("-" * 70)

    for case_name, logits in test_cases:
        x = Tensor([logits])

        # Our stable implementation
        stable_result = log_softmax(x, dim=-1)

        max_input = jnp.max(logits)
        min_output = jnp.min(stable_result.data)
        is_stable = not (jnp.any(jnp.isnan(stable_result.data)) or jnp.any(jnp.isinf(stable_result.data)))

        print(f"{case_name:20} | {max_input:8.0f} | {min_output:15.3f} | {'✅ Yes' if is_stable else '❌ No'}")

    print(f"\n💡 Key Insight: Log-sum-exp trick prevents overflow")
    print("   Without it: exp(700) would cause overflow in standard softmax")
    print("   With it: We can handle arbitrarily large logits safely")

def analyze_loss_memory():
    """
    📊 Analyze memory usage patterns of different loss functions.

    Understanding memory helps with batch size decisions.
    """
    print("\n📊 Analysis: Loss Function Memory Usage...")

    batch_sizes = [32, 128, 512, 1024]
    num_classes = 1000  # Like ImageNet

    print("\nMemory Usage by Batch Size:")
    print("Batch Size | MSE (MB) | CrossEntropy (MB) | BCE (MB) | Notes")
    print("-" * 75)

    for batch_size in batch_sizes:
        # Memory calculations (assuming float32 = 4 bytes)
        bytes_per_float = 4

        # MSE: predictions + targets (both same size as output)
        mse_elements = batch_size * 1  # Regression usually has 1 output
        mse_memory = mse_elements * bytes_per_float * 2 / 1e6  # Convert to MB

        # CrossEntropy: logits + targets + softmax + log_softmax
        ce_logits = batch_size * num_classes
        ce_targets = batch_size * 1  # Target indices
        ce_softmax = batch_size * num_classes  # Intermediate softmax
        ce_total_elements = ce_logits + ce_targets + ce_softmax
        ce_memory = ce_total_elements * bytes_per_float / 1e6

        # BCE: predictions + targets (binary, so smaller)
        bce_elements = batch_size * 1
        bce_memory = bce_elements * bytes_per_float * 2 / 1e6

        notes = "Linear scaling" if batch_size == 32 else f"{batch_size//32}× first"

        print(f"{batch_size:10} | {mse_memory:8.2f} | {ce_memory:13.2f} | {bce_memory:7.2f} | {notes}")

    print(f"\n💡 Memory Insights:")
    print("   - CrossEntropy dominates due to large vocabulary (num_classes)")
    print("   - Memory scales linearly with batch size")
    print("   - Intermediate activations (softmax) double CE memory")
    print(f"   - For batch=1024, CE needs {ce_memory:.1f}MB just for loss computation")

def analyze_production_patterns():
    """
    🚀 Analyze loss function patterns in production ML systems.

    Real insights from systems perspective.
    """
    print("🚀 Production Analysis: Loss Function Engineering Patterns...")

    print("\n1. Loss Function Choice by Problem Type:")

    scenarios = [
        ("Recommender Systems", "BCE/MSE", "User preference prediction", "Billions of interactions"),
        ("Computer Vision", "CrossEntropy", "Image classification", "1000+ classes, large batches"),
        ("NLP Translation", "CrossEntropy", "Next token prediction", "50k+ vocabulary"),
        ("Medical Diagnosis", "BCE", "Disease probability", "Class imbalance critical"),
        ("Financial Trading", "MSE/Huber", "Price prediction", "Outlier robustness needed")
    ]

    print("System Type          | Loss Type    | Use Case              | Scale Challenge")
    print("-" * 80)
    for system, loss_type, use_case, challenge in scenarios:
        print(f"{system:20} | {loss_type:12} | {use_case:20} | {challenge}")

    print("\n2. Engineering Trade-offs:")

    trade_offs = [
        ("CrossEntropy vs Label Smoothing", "Stability vs Confidence", "Label smoothing prevents overconfident predictions"),
        ("MSE vs Huber Loss", "Sensitivity vs Robustness", "Huber is less sensitive to outliers"),
        ("Full Softmax vs Sampled", "Accuracy vs Speed", "Hierarchical softmax for large vocabularies"),
        ("Per-Sample vs Batch Loss", "Accuracy vs Memory", "Batch computation is more memory efficient")
    ]

    print("\nTrade-off                    | Spectrum              | Production Decision")
    print("-" * 85)
    for trade_off, spectrum, decision in trade_offs:
        print(f"{trade_off:28} | {spectrum:20} | {decision}")

    print("\n💡 Production Insights:")
    print("   - Large vocabularies (50k+ tokens) dominate memory in CrossEntropy")
    print("   - Batch computation is 10-100× more efficient than per-sample")
    print("   - Numerical stability becomes critical at scale (FP16 training)")
    print("   - Loss computation is often <5% of total training time")

def test_module():
    """
    Comprehensive test of entire losses module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_log_softmax()
    test_unit_mse_loss()
    test_unit_cross_entropy_loss()
    test_unit_binary_cross_entropy_loss()

    print("\nRunning integration scenarios...")

    # Test realistic end-to-end scenario with previous modules
    print("🔬 Integration Test: Realistic training scenario...")

    # Simulate a complete prediction -> loss computation pipeline

    # 1. MSE for regression (house price prediction)
    house_predictions = Tensor([250.0, 180.0, 320.0, 400.0])  # Predicted prices in thousands
    house_actual = Tensor([245.0, 190.0, 310.0, 420.0])       # Actual prices
    mse_loss = MSELoss()
    house_loss = mse_loss.forward(house_predictions, house_actual)
    assert house_loss.data > 0, "House price loss should be positive"
    assert house_loss.data < 1000, "House price loss should be reasonable"

    # 2. CrossEntropy for classification (image recognition)
    image_logits = Tensor([[2.1, 0.5, 0.3], [0.2, 2.8, 0.1], [0.4, 0.3, 2.2]])  # 3 images, 3 classes
    image_labels = Tensor([0, 1, 2])  # Correct class for each image
    ce_loss = CrossEntropyLoss()
    image_loss = ce_loss.forward(image_logits, image_labels)
    assert image_loss.data > 0, "Image classification loss should be positive"
    assert image_loss.data < 5.0, "Image classification loss should be reasonable"

    # 3. BCE for binary classification (spam detection)
    spam_probabilities = Tensor([0.85, 0.12, 0.78, 0.23, 0.91])
    spam_labels = Tensor([1.0, 0.0, 1.0, 0.0, 1.0])  # True spam labels
    bce_loss = BinaryCrossEntropyLoss()
    spam_loss = bce_loss.forward(spam_probabilities, spam_labels)
    assert spam_loss.data > 0, "Spam detection loss should be positive"
    assert spam_loss.data < 5.0, "Spam detection loss should be reasonable"

    # 4. Test numerical stability with extreme values
    extreme_logits = Tensor([[100.0, -100.0, 0.0]])
    extreme_targets = Tensor([0])
    extreme_loss = ce_loss.forward(extreme_logits, extreme_targets)
    assert not jnp.isnan(extreme_loss.data), "Loss should handle extreme values"
    assert not jnp.isinf(extreme_loss.data), "Loss should not be infinite"

    print("✅ End-to-end loss computation works!")
    print("✅ All loss functions handle edge cases!")
    print("✅ Numerical stability verified!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 04")

# Run comprehensive module test
if __name__ == "__main__":
    test_module()
