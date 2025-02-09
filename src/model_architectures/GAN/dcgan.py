from src.utils.dependencies import *

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_layers=4, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.layers = nn.ModuleList()  # Use ModuleList for proper parameter registration

        in_channels = channels_img
        for i in range(num_layers):
            out_channels = features_d * (2**i)  # Double features at each layer
            kernel_size = 4
            stride = 2
            padding = 1 if i < num_layers - 1 else 0  # Adjust padding for last layer
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                    nn.BatchNorm2d(out_channels) if i > 0 else nn.Identity(), # No batchnorm in first layer.
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = out_channels

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.adaptive_pool(x)
        x = x.view(-1, 1)  # Flatten for sigmoid
        x = nn.Sigmoid()(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, num_layers=4, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        self.layers = nn.ModuleList()

        in_channels = z_dim
        out_channels = features_g * (2**(num_layers-1)) * 4 # Start large.
        self.layers.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        in_channels = out_channels

        for i in range(num_layers - 1):
            out_channels = features_g * (2**(num_layers-2-i)) # Decrease features
            kernel_size = 4
            stride = 2
            padding = 1
            output_padding = 1
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            )
            in_channels = out_channels

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, channels_img, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))

    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)

    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print("Success")


if __name__ == "__main__":
    test()