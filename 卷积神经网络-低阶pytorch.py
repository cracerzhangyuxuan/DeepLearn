import gzip
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


def read_mnist(images_path, labels_path):
    with gzip.open("MNIST_data/" + labels_path ,'rb') as labelsFile:
        y=np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)

    with gzip.open("MNIST_data/" + images_path ,'rb') as imagesFile:
        X=np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16).reshape(len(y), 784).reshape(len(y), 28, 28, 1)

    return X,y

train={}
test={}
# 获取训练集
train['X'], train['y']=read_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
# 获取测试集
test['X'], test['y'] = read_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

# 样本 padding 填充
X_train = np.pad(train['X'], ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(test['X'], ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
# 标签独热编码
y_train = np.eye(10)[train['y'].reshape(-1)]
y_test = np.eye(10)[test['y'].reshape(-1)]

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()

        # 卷积层1
        self.conv1=nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=1)

        # 池化层1
        self.pool1=nn.AvgPool2d(kernel_size=(2,2))

        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=1)

        # 池化层2
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2))

        # 全连接层 输入5*5*16，输出120
        self.fc1=nn.Linear(in_features=5*5*16, out_features=120)
        self.fc2=nn.Linear(in_features=120, out_features=84)
        self.fc3=nn.Linear(in_features=84, out_features=10)

    # 向前传播---将卷积层数放入模型当中
    def forward(self, x):
        x=F.relu(self.conv1(x))
        x=self.pool1(x)
        x = F.relu(self.conv2(x))
        x=self.pool2(x)
        x=x.reshape(-1, 5*5*16)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.softmax(self.fc3(x), dim=1)
        return x

model=LeNet()

model(torch.Tensor(X_train[0]).reshape(-1, 1, 32, 32))

# 依次传入样本和标签张量，制作训练数据集和测试数据集
train_data = torch.utils.data.TensorDataset(torch.Tensor(
    X_train), torch.Tensor(train['y']))
test_data = torch.utils.data.TensorDataset(torch.Tensor(
    X_test), torch.Tensor(test['y']))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
opt = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

def fit(epochs, mode, opt):
    print('=============== Start Training ================')
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1,1,32,32)
            labels = labels.type(torch.LongTensor)

            outputs=model(images)
            loss=loss_fn(outputs,labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Train loss: {:.3f}'
                      .format(epoch + 1, epochs, i + 1, len(train_loader), loss.item()))

        # 每个 Epoch 执行一次测试
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 1, 32, 32)
            labels = labels.type(torch.LongTensor)

            outputs = model(images)
            # 得到输出最大值 _ 及其索引 predicted
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum().item()  # 如果预测结果和真实值相等则计数 +1
            total += labels.size(0)  # 总测试样本数据计数

        print('============ Test accuracy: {:.3f} ============='.format(
            correct / total))



fit(epochs=2,model=model,opt=opt)