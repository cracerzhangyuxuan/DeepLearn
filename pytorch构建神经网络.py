import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

# 加载训练数据，参数 train=True，供 60000 条
train = torchvision.datasets.MNIST(
    root='.', train=True, transform=torchvision.transforms.ToTensor(), download=True)
# 加载测试数据，参数 train=False，供 10000 条
test = torchvision.datasets.MNIST(
    root='.', train=False, transform=torchvision.transforms.ToTensor(), download=True)

print(train.data.shape, train.targets.shape, test.data.shape, test.targets.shape)

#训练数据打乱，使用64小批量
train_loader=torch.utils.data.DataLoader(dataset=train,batch_size=64,shuffle=True)
#测试数据不需要打乱，使用64小批量
test_loader=torch.utils.data.DataLoader(dataset=test,batch_size=64,shuffle=False)

#利用类的方式继承nn.Module类搭建神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)  # 784 是因为训练是我们会把 28*28 展平
        self.fc2 = nn.Linear(512, 128)  # 使用 nn 类初始化线性层（全连接层）
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x=F.relu(self.fc1(x))   #relu激活函数
        x=F.relu(self.fc2(x))
        x=self.fc3(x)           #输出层一般不激活
        return x

model=Net()
print(model)
print(model(torch.randn(1,784)))

loss_fn=nn.CrossEntropyLoss()
opt=torch.optim.Adam(model.parameters(),lr=0.002)

# 手动训练模型
def fit(epochs,model,opt):
    print("Start training, please be patient.")
    # 全数据集迭代 epochs 次
    for epoch in range(epochs):
        # 从数据加载器中读取 Batch 数据开始训练
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28 * 28)  # 对特征数据展平，变成 784
            labels = labels  # 真实标签
            outputs = model(images)  # 前向传播
            loss = loss_fn(outputs, labels)  # 传入模型输出和真实标签
            opt.zero_grad()  # 优化器梯度清零，否则会累计
            loss.backward()  # 从最后 loss 开始反向传播
            opt.step()  # 优化器迭代
            # 自定义训练输出样式
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Train loss: {:.3f}'
                      .format(epoch + 1, epochs, i + 1, len(train_loader), loss.item()))
        # 每个 Epoch 执行一次测试
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28)
            labels = labels
            outputs = model(images)
            # 得到输出最大值 _ 及其索引 predicted
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()  # 如果预测结果和真实值相等则计数 +1
            total += labels.size(0)  # 总测试样本数据计数
        print('============ Test accuracy: {:.3f} ============='.format(
            correct / total))

# 使用Sequential搭建神经网络，比构建类的发方法更加简洁
model_s = nn.Sequential(
    nn.Linear(784, 512),  # 线性类
    nn.ReLU(),  # 激活函数类
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

opt_s = torch.optim.Adam(model_s.parameters(), lr=0.002)  # Adam 优化器
fit(epochs=1, model=model_s, opt=opt_s)  # 训练 1 个 Epoch


# 利用训练好的模型测试
result=model_s(test.data[0].reshape(-1, 28*28).type(torch.FloatTensor))
# 打印第一条测试结果
print(torch.argmax(result))
# 打印第一条目标结果
print(test.targets[0])