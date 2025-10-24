import torch

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")   # Fall back to CPU
    print("GPU not available, using CPU")

# Example: create a tensor and move it to the device
x = torch.tensor([1.0, 2.0, 3.0])
x = x.to(device)
print(f"Tensor device: {x.device}")

# Example: simple tensor operation on GPU
y = x * 2
print(y)
