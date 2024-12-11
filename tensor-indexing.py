import torch

batch_size = 10
feature_dim = 25
x = torch.rand(batch_size, feature_dim)
print(x[0].shape) # equivalent to x[0, :], x[0, :].shape, meaning that is a row vector.

print(x[:, 0].shape) # equivalent to x[:, 0, :], x[:, 0, :].shape, meaning that is a column vector

# get the 3rd example of the batch, and the first 10 features

print(x[2, 0:10]) # creates a list of 10 elements

x[0,0] = 100

# fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices]) # pick out the indices that we want from the tensor x that matches the indices

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols].shape) # prints a 2x2 matrix of values that matches the 2nd row and the 5th column, and then 1st row and 2nd column

# more advanced indexing
x = torch.arange(10)
# pick array that strictly smaller than 2 or the strictly greater than 8
print(x[(x < 2) | (x > 8)])

# pick array that strictly smaller than 2 and the strictly greater than 8
print(x[(x < 2) & (x > 8)])
print(x[x.remainder(2) == 0]) # if the remainder of the number is 0 then it is even. (mod 2)


# useful operations
print(torch.where(x > 5, x, x*2)) # where(condition, if true, if false).
print(torch.tensor([0,0,1,2,2,3,4,4,4,3,3,2]).unique()) # unique values
print(x.ndimension()) # check dimension, if 5x5x5, ndimension would be 3
print(x.numel()) # number of elements