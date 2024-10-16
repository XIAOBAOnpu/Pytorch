import torch

'''
In previous section, the model is built without using any torch tools
In this part, the torch tool will be used to build the NN

The main advantage is that by using pytorch, we can no longer focus on
calculating grad, loss, etc. and we can focus on building our 
computational graph + structure of the NN
'''

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_predict = self.linear(x)
        return y_predict

model = LinearModel()

criterion = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# training cycle forward, backward, update
for epoch in range(100):
    y_pred = model(x_data)  # forward:predict
    loss = criterion(y_pred, y_data)  # forward: loss
    print(epoch, loss.item())

    optimizer.zero_grad()  # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
    loss.backward()  # backward: autograd，自动计算梯度
    optimizer.step()  # update 参数，即更新w和b的值

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
