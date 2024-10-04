import torch

'''
In previous section, the model is built without using any torch tools
In this part, the torch tool will be used to build the NN

The main advantage is that by using pytorch, we can no longer focus on
calculating grad, loss, etc. and we can focus on building our 
computational graph + structure of the NN
'''

class LinearModel(torch.nn.Module):
    def __init__(self):
        super.__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_predict = self.linear(x)
        return y_predict

model = LinearModel()