import numpy as np
import matplotlib.pyplot as plt

'''
Previously, we use a stupid 'visual' method to determine the opt, which
is not realistic in real case.
Therefore, we will use gradient descent, i.e. go to the opposite direction
of the grad.
In this case, the opt (min) can be found after several iterations.
'''

# Define our model:
class GradDescentModel:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, input_x):
        return input_x * self.weight

    # for traditional grad descent, we will not use loss function
    # we use cost function.
    # Differences? -> loss refers to single sample
    #              -> cost refers to average error over all samples
    def cost_function(self, input_x, true_output_y):
        loss = 0
        for x, y in zip(input_x, true_output_y):
            output_prediction = self.forward(x)
            # sum up all the losses for later averaging
            loss += (output_prediction - y) ** 2
        return loss / len(input_x)

    def grad(self, input_x, output_y):
        # reset grad after using it
        grad = 0
        # accumulate all the grad of the sample and calc. its average
        for x, y in zip(input_x, output_y):
            grad += 2 * x * (x * self.weight - y)
        return grad / len(input_x)

# Define initial value in here:
weight = 1
lr = 0.01   # can try play with learning rate here

# Training set
x = np.arange(1.0, 10.0, 1.0)
y = 3 * x

# Initialize some data for plotting
epoch_list = []
cost_list = []

print('Before starting, the weight is: ', weight)

for epoch in range(100):
    model = GradDescentModel(weight)
    cost_validation = model.cost_function(x, y)
    grad_validation = model.grad(x, y)
    weight -= lr * grad_validation
    print('\tEpoch = ', epoch, '\t | Weight = ', weight, '\t | Cost = ', cost_validation)
    cost_list.append(cost_validation)
    epoch_list.append(epoch)
print('After starting, the weight is: ', weight)

plt.plot(epoch_list, cost_list)
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()
