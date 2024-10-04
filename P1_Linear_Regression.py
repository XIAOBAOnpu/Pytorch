import numpy as np
import matplotlib.pyplot as plt

'''
This part is quite straight-forward:
first we define a model (an easy one for now): y = 3x
then we iterate over the weight, from 0 to 5 in this case
and calculate the TOTAL differences between the true output and estimated one
finally we can obtain the MSE by dividing the TOTAL difference with #
plotting and see the opt weight visually
'''

# Define our linear model in here:
class LinearModel:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, input_x):
        return input_x * self.weight

    def loss_function(self, input_x, true_output_y):
        # get the predicted output using an ASSIGNED weight
        output_prediction = self.forward(input_x)
        # calculate the differences between true output and predicted output
        return (output_prediction - true_output_y) ** 2


# Training set
x = np.arange(1.0, 10.0, 1.0)
y = 3 * x

# Initialize some data for plotting
weight_list = []
mse_list = []

# Try different weight and iterate -> in order to find the opt weight
for weight in np.arange(0.0, 5.1, 0.1):
    print('=======================')
    print('Weight = ', weight)
    # Print headers with proper formatting
    print(f"{'True input':>12} {'Pred output':>12} {'True output':>12} {'Loss':>12}")

    model = LinearModel(weight)
    loss_sum = 0
    # for each weight, calculate the total loss
    for x_validation, y_validation in zip(x, y):
        loss_validation = model.loss_function(x_validation, y_validation)
        y_predict = model.forward(x_validation)
        loss_sum += loss_validation
        # Print the values for each iteration with proper formatting
        print(f"{x_validation:12.2f} {y_predict:12.2f} {y_validation:12.2f} {loss_validation:12.4f}")
    mse = loss_sum / len(x)
    print('MSE = ', mse)
    weight_list.append(weight)
    mse_list.append(mse)

# Plot
plt.plot(weight_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('Weight')
plt.show()