import torch

def func(x):
    x = x.squeeze()
    result = torch.stack([4*x[0]**3 + 2*x[1] + x[2],
                          x[0]**2 + 4*x[1]**3 + 2*x[2],
                          2*x[0] + x[1]**2 + x[2]])
    return result.unsqueeze(0)

input = torch.tensor([[2., 3., 4.]], requires_grad=True)

#JACOBIAN TAKES FUNCTION THAT PRODUCES 3 OUTPUTS W.R.T. 3 INPUTS
output = torch.autograd.functional.jacobian(func, input)
# print(output.shape)
# print(output)

output_squeezed = output.sum(dim=2)
print(output_squeezed.shape) #[1, 3, 3]
print(output_squeezed)
'''
[[[ 48.,   2.,   1.],
[  4., 108.,   2.],
[  2.,   6.,   1.]]]
'''
