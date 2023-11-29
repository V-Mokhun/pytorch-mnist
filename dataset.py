import torch
from torchvision import datasets, transforms

BATCH_SIZE = 32
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

train_set = datasets.MNIST('data', download=True, train=True, transform=transform)
validation_set = datasets.MNIST('data', download=True, train=False, transform=transform)
classnames = train_set.classes
NUM_CLASSES = len(classnames)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE)

