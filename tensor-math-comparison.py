import torch

x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])


# addition

z1 = torch.empty(3)
torch.add(x, y, out=z1)

z2 = torch.add(x,y)
z = x + y # do exactly the same. regular

# subtraction
z = x - y

# division,  a bit clunky?
z = torch.true_divide(x,y) # elementwise

# inplace
t = torch.zeros(3)
t.add_(x) # adds x to t
t += x # same
t = t + x # not inplace

# exponentiation
z = x.pow(2) # elementwise
z = x ** 2 # same

# simple comparison
z = x < y
z = x > 0

# matrix multiplication
x1 = torch.rand((2,5))
y1 = torch.rand((5,3))
z = torch.mm(x1,y1) # 2x3
z = x1.mm(y1)

# matrix exponentiation
matrix_exp = torch.rand(5,5)
matrix_exp.matrix_power(3)

# element wise multiplication
z = x * y

# dot product
z = torch.dot(x,y)

# batch matrix multiplication
# tensor1 and tensor2 are 3D tensors where the first dimension is a batch
# of 2D matrices. The batch matrix multiplication is done on the last two
# dimensions so the result is a 3D tensor with the same batch dimension
# and the last two dimensions are the result of the matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand(batch,n,m)
tensor2 = torch.rand(batch,m,p)

out1 = torch.bmm(tensor1, tensor2)  # (batch, n, p)

# example of broadcasting
x1 = torch.rand(5, 5)
x2 = torch.rand(1, 5)
x3 = torch.rand(5, 1)
x4 = torch.rand(1, 1)

# automatically expanding the smaller tensor to match the shape of the larger tensor

z = x1 - x2 # x2 will be broadcasted to 5x5　
z = x1 + x3 # x3 will be broadcasted to 5x5
z = x1 * x4 # x4 will be broadcasted to 5x5

# some other useful tensor operations
sum_x = torch.sum(x, dim=0) # sum of all elements in the tensor
sum_x = torch.sum(x, dim=1) # sum of each row in the tensor
sum_x = torch.sum(x, dim=2) # sum of each column in the tensor
values, indices = torch.max(x, dim=0) # max of all elements in the tensor
values, indices = torch.min(x, dim=0) # min of all elements in the tensor
abs_x = torch.abs(x) # absolute value of all elements in the tensor
z = torch.argmax(x, dim=0) # index of the maximum element in each row
z = torch.argmin(x, dim=0) # index of the minimum element in each row
mean_x = torch.mean(x, dim=0) # mean of all elements in the tensor
z = torch.eq(x,y) # elementwise comparison returns a tensor of booleans

z = torch.clamp(x, min=0, max=10) # clamp all elements to be greater than 0, less than 10, if greater than 10, set to 10
z = torch.clamp(x, max=0) # clamp all elements to be less than 0, if less than 0, set to 0, if greater than 0, set to 0
sorted_y, indices = torch.sort(y, dim=0, descending=False) # increasing order
sorted_y, indices = torch.sort(y, dim=0, descending=True) # decreasing order

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x) # returns True if any element is True
z = torch.all(x) # returns True if all elements are True