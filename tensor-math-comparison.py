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
print(z)

# dot product
z = torch.dot(x,y)
print(z)