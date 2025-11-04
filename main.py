from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parent.parent))


from core.activations.activations import ReLU
from core.layers.linear import Linear
from core.tensor import Tensor

def main():
    print("Hello from nothingtorch!")

    # Test basic layer functionality
    layer = Linear(input_size=3, output_size=2)
    x = Tensor([[1.0, 2.0, 3.0]])
    y = layer(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

    # Test layer composition
    layer1 = Linear(3, 4)
    layer2 = Linear(4, 2)
    relu = ReLU()

    # Forward pass
    h1 = relu(layer1(x))
    output = layer2(h1)
    print(f"Final output: {output.data}")


if __name__ == "__main__":
    main()
