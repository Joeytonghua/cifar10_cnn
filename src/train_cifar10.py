import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models import *
import time

from models.dal_simple import SimpleDLA
from models.model import Qiao
from models.vgg import VGG

# 数据增广 沐神选择了对测试数据集不做操作，这里模仿一下
image_size = (32, 32)
train_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # # 图片太小了，不要过分进行增量操作？！
    # transforms.Resize(image_size),
    # transforms.CenterCrop(image_size),
    # transforms.Grayscale(num_output_channels=3),
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset", download=True, train=True,
                                          transform=train_transform)

test_data = torchvision.datasets.CIFAR10(root="../dataset", download=True, train=False,
                                         transform=test_transform)

# 设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device：", device)

# length
train_data_len = len(train_data)
test_data_len = len(test_data)
print("test_data_len：{}".format(test_data_len))
print("train_data_len：{}".format(train_data_len))

# 使用DataLoader加载数据集
train_data_loader = DataLoader(train_data, batch_size=128)
test_data_loader = DataLoader(test_data, batch_size=128)

# 搭建神经网络
print('==> Building model..')
# joe = Qiao()
# joe = SimpleDLA()
joe = VGG('VGG13')
joe = joe.to(device)

# joe = torchvision.models.vgg19()
# joe.to(device)

# 损失函数
loss_function = nn.CrossEntropyLoss()
loss_function.to(device)

# 优化器 这里采用随机梯度下降
learning_rate = 0.13
optimizer = torch.optim.SGD(params=joe.parameters(),lr=learning_rate,momentum=0.01)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮s数
epoch = 60

# tensorboard
writter = SummaryWriter("logs_model_1")

s_time = time.time()
for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i + 1))

    # 训练开始
    joe.train()
    for data in train_data_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = joe(imgs)
        loss = loss_function(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 300 == 0:
            print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))
            writter.add_scalar("model/train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    joe.eval()
    total_test_loss = 0
    total_acc = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = joe(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_acc += accuracy.item()

    print("Total loss :{}".format(total_test_loss))
    print("Total acc:{}%".format((total_acc / test_data_len) * 100))

    writter.add_scalar("model/test_loss", total_test_loss, total_test_step)
    writter.add_scalar("total_acc", total_acc / test_data_len, total_test_step)
    total_test_step += 1

    # 模型保存
    torch.save(joe, "../model_save/model_gpu{}.pth".format(i + 1))
    print("模型已经保存！")

e_time = time.time()
print("Whole time:{}".format(e_time - s_time))
writter.close()
