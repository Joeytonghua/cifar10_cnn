import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import *
import time

image_size = (32, 32)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset", download=True, train=True,
                                          transform=transform)

test_data = torchvision.datasets.CIFAR10(root="../dataset", download=True, train=False,
                                         transform=transform)

# 设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device", device)

# length
train_data_len = len(train_data)
test_data_len = len(test_data)
print("test_data_len：{}".format(test_data_len))
print("train_data_len：{}".format(train_data_len))

# 使用DataLoader加载数据集
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
joe = Qiao()
joe.to(device)

# 损失函数
loss_function = nn.CrossEntropyLoss()
loss_function.to(device)

# 优化器 这里采用随机梯度下降
learning_rate = 1e-2
optimizer = torch.optim.SGD(params=joe.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮s数
epoch = 50

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
            writter.add_scalar("train_loss", loss.item(), total_train_step)

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

    writter.add_scalar("test_loss", total_test_loss, total_test_step)
    writter.add_scalar("total_acc", total_acc / test_data_len, total_test_step)
    total_test_step += 1

    # 模型保存
    torch.save(joe, "../model_save/model_gpu{}.pth".format(i + 1))
    print("模型已经保存！")

e_time = time.time()
print("Whole time:{}".format(e_time - s_time))
writter.close()
