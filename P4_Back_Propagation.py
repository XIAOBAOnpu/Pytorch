import numpy as np
import torch
from torch.onnx.symbolic_opset9 import tensor

'''
In P3, we have simplfied the calc. of the grad quite a lot.
However, there is still one thing that we need to calc. by hand, i.e. the derivative below:
                    2 * input_x * (input_x * self.weight - output_y)
                    
In practice, we have lots of layers, multiple inputs/outputs and non-linear functions, which 
makes it impossible to get this kind of expression

Therefore in this part, we create a so-called computation graph to calc. the grad
The tensor in pytorch is used, as long as you use a tensor, you are actually creating computation graph
                    !!! tensor stores the data et grad !!!
'''

# Training set
x = np.arange(1.0, 10.0, 1.0)
y = 3 * x

# We need to calc. the grad of the weight, therefore set requires_grad = True
weight = torch.tensor([1.0], requires_grad=True)
lr = 0.01

# Define our linear model in here:
class BackPropagationModel:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, input_x):
        # attention that if the weight here is a tensor, the return value of
        # forward will be a tensor type as well
        # the computation graph has been created
        return input_x * self.weight

    def loss_function(self, input_x, true_output_y):
        output_prediction = self.forward(input_x)
        return (output_prediction - true_output_y) ** 2

print('Before starting, the weight is: ', weight.data.item())

for epoch in range(100):
    for training_x, training_y in zip(x, y):
        model = BackPropagationModel(weight)
        # forward to get the loss
        loss = model.loss_function(training_x, training_y)
        # backward to get the opt
        loss.backward()
        print('\tGrad = ', weight.grad.item(), '\t| Input = ', training_x, '\t| Output = ', model.forward(training_x).item())
        # update the weight
        weight.data -= lr * weight.grad.data
        # the grad computed by .bakcward() will be accumulated -> therefore always reset after update
        weight.grad.data.zero_()
    print('Epoch: ', epoch, '\t | Weight = ', weight.data.item(), '\t | Loss = ', loss.item())
print('After starting, the weight is: ', weight.data.item())