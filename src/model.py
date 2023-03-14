import torch
from torch import nn
from torch.nn import functional as F


class Qiao(nn.Module):
    def __init__(self):
        super(Qiao, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),nn.Softmax(),
            nn.Linear(64, 10)
        )
        self.model1 =nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=20 * 4 * 4, out_features=64),
            torch.nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x




if __name__ == '__main__':
    qiao = Qiao()
    test = torch.ones((64, 3, 32, 32))
    print(qiao)
    print(qiao(test).shape)

