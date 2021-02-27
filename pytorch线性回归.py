import torch
from matplotlib import pyplot as plt
import torch.nn as nn

torch.manual_seed(10)
x=torch.linspace(1,10,50)
y=2*x+3*torch.rand(50)

plt.style.use("ggplot")
plt.scatter(x,y)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel,self).__init__()
        self.Linear=nn.Linear(1,1)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegressionModel()  # 实例化模型

loss_fn=nn.MSELoss()
opt=torch.optim.SGD(model.parameters(),lr=0.01)
print(loss_fn,opt)

iters=100
for i in range(iters):

    x=x.reshape(len(x),1)
    y=y.reshape(len(x),1)

    y_=model(x)             #向前传播
    loss=loss_fn(y_,y)      #计算损失
    opt.zero_grad()         #优化器梯度清零，否则会累计
    loss.backward()         #最后loss开始反向传播
    opt.step()              #优化器迭代

    if (i+1) % 10==0:
        print('Iteration [{}/{}], Loss: {:.3f}'
              .format(i+1, iters, loss.item()))

weight = model.state_dict()['linear.weight']  # 权重
bias = model.state_dict()['linear.bias']  # 偏置项

plt.scatter(x, y, c='black')
plt.plot([0, 11], [bias, weight * 11 + bias], 'r')
plt.show()