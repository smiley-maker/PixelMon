from src.utils.dependencies import *
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


class MNIST_DATASET(Dataset):
    def __init__(self) -> None:
        super().__init__()
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_data = MNIST(
            root="~/Data/",
            transform=transform,
            download=True
        )

        self.test_data = MNIST(
            root="~/Data/",
            transform=transform,
            download=True
        )
    

    