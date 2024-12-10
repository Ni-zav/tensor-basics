import torch
device = "cuda" if torch.cuda.is_available() else "cpu" 
x = torch.tensor([[1, 2], [3, 4]], device=device,dtype=torch.float32, requires_grad=True)
print(x)

print(x.dtype)
print(x.device)
print(x.shape)
print(x.requires_grad)

# Other common initialization methods
x = torch.empty(size = (3,3))
x = torch.zeros((3,3))
x = torch.ones((3,3))
x = torch.rand((3,3))
x = torch.eye(5,5)
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=1, steps=6)
x = torch.empty(size=(1,5)).normal_(mean=0, std=1) # normal distribution
x = torch.empty(size=(1,5)).uniform_(0, 1)  # uniform distribution
x = torch.diag(torch.ones(3)) # diagonal matrix 3x3 (same as eye but preserve diagonal)

# how to initialize and convert tensor to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool()) # bool
print(tensor.short()) # int16
print(tensor.int()) # int32
print(tensor.long()) # int64
print(tensor.half()) # float16
print(tensor.float()) # float32, important
print(tensor.double())  # float64

# array to tensor conversion and vice-versa
import numpy as np
np_array = np.ones((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()