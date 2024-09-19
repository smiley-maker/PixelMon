from src.utils.dependencies import *

class GAN(nn):
    def __init__(self,
                 dataloader : DataLoader,
                 device : torch.device,
                 image_size : tuple[int] = 64,
                 num_channels : int = 3,
                 z_dim : int = 100,
                 ngf : int = 64,
                 ndf : int = 64
                ):
        super().__init__()
        self.data = dataloader
        self.device = device
        self.image_size = image_size
        self.num_channels = num_channels
        self.z_dim = z_dim
        self.ngf = ngf
        self.ndf = ndf
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    class Generator():
        def __init__(self, ngpu):
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d( self.nz, self.ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(self.ngf * 8),
                nn.ReLU(True),
                # state size. ``(ngf*8) x 4 x 4``
                nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf * 4),
                nn.ReLU(True),
                # state size. ``(ngf*4) x 8 x 8``
                nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf * 2),
                nn.ReLU(True),
                # state size. ``(ngf*2) x 16 x 16``
                nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf),
                nn.ReLU(True),
                # state size. ``(ngf) x 32 x 32``
                nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. ``(nc) x 64 x 64``
            )

        def forward(self, input):
            return self.main(input)