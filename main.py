from core.tensor import Tensor

def main():
    print("Hello from nothingtorch!")

    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([[4], [5], [6]], requires_grad=False)
    z = x.matmul(y)
    
    # print(x + 10)

    print(y.reshape((3, -1)).shape)
    # print(z.numpy())

if __name__ == "__main__":
    main()
