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
print(output.shape)
print(output)

output_squeezed = output.sum(dim=2)
print(output_squeezed)

jacobian = torch.stack([output_squeezed, output_squeezed, output_squeezed])
# print(jacobian)

#generating X
x_tmp = torch.arange(1, 10)
x = x_tmp.view(3, 3)
# print("x", x)

#multiplication
x_expanded = x.unsqueeze(1) #[3, 1, 3]
# print("x_expanded:", x_expanded)
# print(jacobian * x_expanded)

result = torch.sum(jacobian * x_expanded, dim=2)
# print("result", result)
# print(result.shape)
