import os

import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

image_path = "../test_dataset/img.png"
# image = image.convert('RGB')
# print(image)
# image.show()
# transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
#                                             torchvision.transforms.ToTensor()])
#
# image = transform(image)
#
# model = torch.load("../model_save/model_gpu20.pth")
#
# image = torch.reshape(image, (1, 3, 32, 32))
# image = image.to('mps')
# model.eval()
# with torch.no_grad():
#     output = model(image)
#
# print("种类：",classes[output.argmax(1).item()])

# 对单个图片进行预测
def prediect(img_path):
    img = Image.open(img_path)
    img = img.convert("RGB") # 有些图片为RGBA格式,所以需要转换成3通道
    # print(img) # mode=RGB size=437x357 at 0x102F42130> (437, 357)

    # 对输入图片进行调整
    trans = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor()
    ])
    img = trans(img)
    # print(img.shape) # torch.Size([3, 32, 32])

    # 维度添加，下面给出两种方法: 1.reshape() 2.unsqueeze()
    img = torch.reshape(img,(1,3,32,32))
    img = img.to('mps')
    # img = torch.unsqueeze(img, 0)  # 对图片数据维度进行扩充,给指定位置加上维数, 参数0:在0这个位置上增加维度1
    # print(img.shape) #torch.Size([1, 3, 32, 32])

    #读取模型
    net = torch.load("../model_save/model_gpu50.pth")
    net.eval()
    with torch.no_grad():
        outputs = net(img).to('mps')
        result = classes[outputs.argmax(1).item()]
        return result

# 对文件夹下对图片进行预测

def predict_pkg(pkg_path):
    img_name_list = os.listdir(pkg_path)
    for i in range(len(img_name_list)):
        img_path = pkg_path + '/' + img_name_list[i]
        img = Image.open(img_path)
        plt.subplot(6, 6, i + 1)
        plt.tight_layout()
        plt.imshow(img)
        plt.title("predict: {}".format(prediect(img_path)))
        plt.xticks([])
        plt.yticks([])
    plt.show()






if __name__ == '__main__':
    # image_path = "../test_dataset/dogs/img.png"
    # p = prediect(image_path)
    # print(p)

    pkg_path = "../test_dataset/dogs"
    predict_pkg(pkg_path)
