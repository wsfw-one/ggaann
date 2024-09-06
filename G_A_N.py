import torch               # 导入PyTorch库
import torch.nn as nn      # 导入神经网络模块
import torchvision.transforms as transforms # 导入图像转换工具
from torchvision import datasets          # 导入数据集工具
from torch.utils.data import DataLoader   # 导入数据加载器
import numpy as np                        # 导入NumPy库
import matplotlib.pyplot as plt           # 导入绘图库
# 数据预处理和加载
mnist = datasets.MNIST(       # 加载MNIST数据集
    root='./data',         # 数据集保存目录
    train=False,              # 使用测试集（这里可能是误用，通常GAN训练使用训练集）
    download=False,           # 不下载数据集（假设已下载）
    transform=transforms.Compose([   # compose将多个变换连接起来
        transforms.Resize((28, 28)), # 调整图像大小到28x28
        transforms.ToTensor(),       # 转换为张量，将PIL图像或NumPy图像转换为Tensor,并将像素值缩放到[0.0,1.0]
        transforms.Normalize([0.5], [0.5]) # 归一化,对Tensor进行标准化，第一个参数是均值，第二个是标准差，
        # 这里所有的像素值都会减去0.5，(因为他们都是[0.0,1.0]范围内)，然后除以0.5，意味着图像数据将会被缩放到[-1,1]
    ])
)

dataloader = DataLoader(     # 创建数据加载器
    dataset=mnist,            # 数据集
    batch_size=64,            # 批次大小
    shuffle=True              # 是否打乱数据
)

# 定义图像生成网络
def gen_img_plot(model, epoch, text_input): # 定义生成图像并绘图的函数
    prediction = np.squeeze(model(text_input).detach().cpu().numpy()[:16]) # 生成预测图像，squeeze会移除那些1的维度，使形状更加紧凑，
    # 例如变为(C, H, W)或(B, C, H, W)，这在可视化单个图像或批量图像时很有帮助。
    # .detach():这个方法用于从计算图中分离出张量，这意味着它不再会被跟踪梯度，
    # 当我们从模型中取出输出时，我们通常会调用.detach()来确保不会在这些张量上累积不必要的梯度历史，这样可以节省内存。
    plt.figure(figsize=(4, 4)) # 创建绘图窗口
    for i in range(16): # 绘制16张图像
        plt.subplot(4, 4, i + 1) # 设置子图
        plt.imshow((prediction[i] + 1) / 2) # 绘制图像
        plt.axis('off') # 关闭坐标轴
    plt.show() # 显示图像


# 生成器定义
class Generator(nn.Module):
    def __init__(self):  #构造函数
        super(Generator, self).__init__()  #初始化父类

        def block(in_feat, out_feat, normalize=True):
            # block函数，用于构建生成器网络中的一个块（block）
            # 这个块包含一个线性层，一个可选的归一化，还追加了一个LeakyReLU激活函数
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            #Leaky ReLU（Leaky Rectified Linear Unit）是一种修正线性单元（Rectified Linear Unit, ReLU）的变体，
            # 其设计目的是解决ReLU激活函数在负半轴区域的“死亡”问题，
            # 即当输入为负数时，ReLU的导数为0，导致神经元无法更新权重，从而变得“死掉”。
            return layers
        #构建生成器网络
        self.mean = nn.Sequential(
            *block(100, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x): # 前向传播函数
        imgs = self.mean(x) # 应用神经网络层
        imgs = imgs.view(-1, 1, 28, 28) # 调整输出尺寸
        return imgs # 返回生成的图像

# 定义判别器类
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.mean = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # 该函数将任意实数值映射到 (0, 1) 区间内
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        img = self.mean(x)  # 对 64条数据的每一条都进行模型运算
        return img


# 实例化  初始化
generator = Generator()
discriminator = Discriminator()

# 定义优化器
G_Apim = torch.optim.Adam(generator.parameters(), lr=0.0001) # 定义生成器优化器
D_Apim = torch.optim.Adam(discriminator.parameters(), lr=0.0002) # 定义判别器优化器

# 损失函数
criterion = torch.nn.BCELoss()  # 对应 Sigmoid，计算二元交叉墒损失

epoch_num = 100  # 设定训练轮数
G_loss_save = []  # 保存生成器损失
D_loss_save = []  # 保存判别器损失
for epoch in range(epoch_num):  # 循环每个epoch
    G_epoch_loss = 0  # 初始化生成器损失
    D_epoch_loss = 0  # 初始化判别器损失
    count = len(dataloader)  # 数据批次总数
    for i, (img, _) in enumerate(dataloader):  # 循环每个batch
        # 训练判别器
        size = img.size(0)  # 获取批次大小
        fake_img = torch.randn(size, 100)  # 生成随机噪声
        output_fake = generator(fake_img)  # 生成假图像
        fake_socre = discriminator(output_fake.detach())  # 判别假图像
        D_fake_loss = criterion(fake_socre, torch.zeros_like(fake_socre))  # 计算假图像损失
        real_socre = discriminator(img)  # 判别真图像
        D_real_loss = criterion(real_socre, torch.ones_like(real_socre))  # 计算真图像损失
        D_loss = D_fake_loss + D_real_loss  # 合并损失
        D_Apim.zero_grad()  # 清空判别器梯度
        D_loss.backward()  # 反向传播
        D_Apim.step()  # 更新判别器权重

        # 训练生成器
        fake_G_socre = discriminator(output_fake)  # 判别生成的假图像
        G_fake_loss = criterion(fake_G_socre, torch.ones_like(fake_G_socre))  # 计算生成器损失
        G_Apim.zero_grad()  # 清空生成器梯度
        G_fake_loss.backward()  # 反向传播
        G_Apim.step()  # 更新生成器权重

        # 计算平均损失
        G_epoch_loss += G_fake_loss
        D_epoch_loss += D_loss

    # 计算平均损失
    G_epoch_loss /= count
    D_epoch_loss /= count

    # 保存损失并打印
    G_loss_save.append(G_epoch_loss.item())
    D_loss_save.append(D_epoch_loss.item())

    print('Epoch: [%d/%d] | G_loss: %.3f | D_loss: %.3f' % (epoch, epoch_num, G_epoch_loss, D_epoch_loss))
    text_input = torch.randn(64, 100)  # 生成随机噪声
    gen_img_plot(generator, epoch, text_input)  # 生成图像并绘图

x = [epoch + 1 for epoch in range(epoch_num)] # 创建epoch列表
plt.figure() # 创建绘图窗口
plt.plot(x, G_loss_save, 'r') # 绘制生成器损失
plt.plot(x, D_loss_save, 'b') # 绘制判别器损失
plt.ylabel('loss') # y轴标签
plt.xlabel('epoch') # x轴标签
plt.legend(['G_loss','D_loss']) # 图例
plt.show() # 显示图像

