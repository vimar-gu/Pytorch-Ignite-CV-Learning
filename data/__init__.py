from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
train_set = MNIST(download=True, root='.', transform=data_transform, train=True)
test_set = MNIST(download=True, root='.', transform=data_transform, train=False)
