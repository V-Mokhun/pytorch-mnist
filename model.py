import torch
from torch import nn
from torchmetrics.classification import Accuracy
from dataset import NUM_CLASSES


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.cnn_layer(x)
        return self.linear_layer(x)


model = MNISTModel()

accuracy_fn = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
