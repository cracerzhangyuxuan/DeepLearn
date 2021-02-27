import torch
from torchvision import datasets, transforms
from torch import nn
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from IPython import display

# 定义图片处理方法
transforms = transforms.Compose([
    transforms.Resize(64),  # 调整图片大小到 64*64
    transforms.CenterCrop(64), # 中心裁剪
    # 将 PIL Image 或者 numpy.ndarray 转化为 PyTorch 中的 Tensor，并转化像素范围从 [0, 255] 到 [0.0, 1.0]
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将图片归一化到(-1,1)
])
# 读取自定义图片数据集
dataset = datasets.ImageFolder('avatar/',  # 数据路径，一个类别的图片在一个文件夹中
                               transform=transforms)
# 制作数据加载器
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=16,  # 批量大小
                                         shuffle=True,  # 乱序
                                         num_workers=2  # 多进程
                                         )

print(dataloader)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (64*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (64*4) x 8 x 8
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (64*2) x 16 x 16
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (3) x 64 x 64
        )


print(Generator())

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (3) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

print(Discriminator())

dev=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(dev)

# 生成器网络
netD = Discriminator().to(dev)
# 判别器网络
netG = Generator().to(dev)
criterion = nn.BCELoss().to(dev)

lr = 0.0002  # 学习率
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))  # Adam 优化器
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))


epochs = 100
for epoch in range(epochs):
    for n, (images, _) in enumerate(dataloader):

        real_labels = torch.ones(images.size(0)).to(dev)  # 真实数据的标签为 1
        fake_labels = torch.zeros(images.size(0)).to(dev) # 伪造数据的标签为 0

        # 使用真实图片训练判别器网络
        netD.zero_grad() # 梯度置零
        output = netD(images.to(dev)) # 输入真实数据
        lossD_real = criterion(output.squeeze(), real_labels) # 计算损失

        # 使用伪造图片训练判别器网络
        noise = torch.randn(images.size(0), 100, 1, 1).to(dev) # 随机噪声，生成器输入
        fake_images = netG(noise) # 通过生成器得到输出
        output2 = netD(fake_images.detach()) # 输入伪造数据
        lossD_fake = criterion(output2.squeeze(), fake_labels) # 计算损失
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # 训练生成器网络
        netG.zero_grad()
        output3 = netD(fake_images)
        lossG = criterion(output3.squeeze(), real_labels)
        lossG.backward()
        optimizerG.step()

        # 生成 64 组测试噪声样本，最终绘制 8x8 测试网格图像
        fixed_noise = torch.randn(64, 100, 1, 1).to(dev)
        fixed_images = netG(fixed_noise)
        fixed_images = make_grid(fixed_images.data, nrow=8, normalize=True).cpu()
        plt.figure(figsize=(6, 6))
        plt.title("Epoch[{}/{}], Batch[{}/{}]".format(epoch+1, epochs, n+1, len(dataloader)))
        plt.imshow(fixed_images.permute(1, 2, 0).numpy())
        display.display(plt.gcf())
        display.clear_output(wait=True)


