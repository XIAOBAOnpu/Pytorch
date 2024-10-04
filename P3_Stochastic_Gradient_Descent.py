import numpy as np

'''
In P2, the power gradient descent algorithm is developed to obtain the opt
However, we have to realize that it is not always possible to calculate the 
grad over all the samples (or more specifically sum up all gradients and averaging)

Therefore, the so-called SGD method is developed
Instead of calc. the grad over all samples, it only calc. the grad of single sample
'''

# Define our model:
class SGDModel:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, input_x):
        return input_x * self.weight

    # N.B. Here we use loss function once again instead of cost function
    # def of the loss et cost has been illustrated in P2
    def loss(self, input_x, true_output_y):
        # get the predicted output using an ASSIGNED weight
        output_prediction = self.forward(input_x)
        # calculate the differences between true output and predicted output
        return (output_prediction - true_output_y) ** 2

    def grad(self, input_x, output_y):
        return 2 * input_x * (input_x * self.weight - output_y)

# Define initial value in here:
weight = 1
lr = 0.001  # can try play with learning rate here

# Training set
x = np.arange(1.0, 10.0, 1.0)
y = 3 * x

print('Before starting, the weight is: ', weight)

for epoch in range(100):
    # here the differneces with P2 is
    # we update weight by every grad of the sample of the training set
    for training_x, training_y in zip(x, y):
        model = SGDModel(weight)
        gradient = model.grad(training_x, training_y)
        weight -= lr * gradient
        print('\tGrad = ', gradient, '\t| Input = ', training_x, '\t| Output = ', model.forward(training_x))
        loss = model.loss(training_x, training_y)
    print('Epoch: ', epoch, '\t | Weight = ', weight, '\t | Loss = ', loss)

print('After starting, the weight is: ', weight)