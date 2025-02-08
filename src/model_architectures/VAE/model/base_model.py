from src.utils.dependencies import *
from abc import abstractmethod

class BaseVAE(nn.Module):

    def __init__(self, *args, **kwargs):
        super(BaseVAE, self).__init__(*args, **kwargs)
    
    def encode(self, input : torch.tensor) -> list[torch.tensor]:
        raise NotImplementedError
    
    def sample(self, batch_size : int, current_device : int, **kwargs) -> torch.tensor:
        raise NotImplementedError
    
    def generate(self, x : torch.tensor, **kwargs) -> torch.tensor:
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, *inputs : torch.tensor) -> torch.tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs) -> torch.tensor:
        pass