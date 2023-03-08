import torch
import torchvision
from PIL import Image

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

image_path = "image/img_5.png"
image = Image.open(image_path)
image = image.convert('RGB')
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)

model = torch.load("model_save/model_gpu20.pth")
print(model)

image = torch.reshape(image, (1, 3, 32, 32))
image = image.to('mps')
model.eval()
with torch.no_grad():
    output = model(image)

print(output)
print("种类：",classes[output.argmax(1).item()])



