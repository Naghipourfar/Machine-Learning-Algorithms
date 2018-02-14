from __future__ import print_function

import sys
import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

sys.setrecursionlimit(10000000)

"""
    Created by Mohsen Naghipourfar on 2/9/18.
    Email : mn7697np@gmail.com
"""

x = torch.Tensor(5, 3)  # create a tensor with 5 * 3 size --> FloatTensor
# print(x)

x = torch.rand(5, 3)  # Create Random Tensor with size 5 * 3
# print(x)

# print(x.size()) # get x's size


# Operations
x = torch.rand(4, 5)
y = torch.rand(4, 5)

# Addition
# print(x + y)  # x + y
# print(torch.add(x, y))  # x + y
# print(y.add_(x))  # y += x

# z = torch.Tensor(4, 5)
# torch.add(x, y, out=z)
# print(z)
# print(x)
# print(x[:, 1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

a = torch.ones(5, 6)
print(a)

b = a.numpy()  # convert Tensor to numpy.ndarray
print(b)

a.add_(1)  # b changes too!
# print(a)
# print(b)

# converting numpy to tensor
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# Tensors can be moved onto GPU using the `cuda()` method.
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x + y)

# Autograd package in Torch
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

y = x + 2  # Operation Support
print(x.grad_fn)  # Created by user --> grad_fn is None
print(y.grad_fn)  # Created by an Operation --> grad_fn is not None

z = y * y * 3
mean = z.mean()
print(z, mean)

# Backprop
mean.backward()  # mean.backward(torch.Tensor([1.0]))
print(x.grad)  # compute d(mean) / dx_i when x_i = 1

# Another Example
x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
print(y)
while y.data.norm() < 1000:
    y = y * 2
print(y)

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)


# Neural Networks

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)  # in, out, kernel_size
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # in, out
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # input, kernel_size
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))  # reshape the data
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # output of Neural Network

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


nnet = Net()
print(nnet)

input = Variable(torch.randn(1, 1, 32, 32))
out = nnet.forward(input)
print(out)

params = list(nnet.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

nnet.zero_grad()  # Zero the gradient buffers of all parameters and backprops with random gradients
out.backward(torch.randn(1, 10))

# Loss Function
output = nnet.forward(input)
target = Variable(torch.arange(1, 11))  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


nnet.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(nnet.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(nnet.conv1.bias.grad)

# Updating the weights
learning_rate = 0.01
for f in nnet.parameters():
    f.data.sub_(f.grad.data * learning_rate)


# Create your own optimizer
import torch.optim as optim

my_optimizer = optim.SGD(nnet.parameters(), lr=0.01)

# in your training loop:
my_optimizer.zero_grad()   # zero the gradient buffers
output = nnet.forward(input)
loss = criterion(output, target)
loss.backward()
my_optimizer.step()    # Does the update



