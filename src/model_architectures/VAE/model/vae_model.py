#!/usr/bin/env python3


from src.utils.dependencies import *
from src.model_architectures.VAE.model.base_model import BaseVAE
from torchvision.models.resnet import resnet152, resnet18, resnet50
from torch.autograd import Variable


class VanillaVAE(BaseVAE):
    def __init__(self, 
                 in_channels : int,
                 latent_dim : int,
                 hidden_dims : list = None,
                 *args, 
                 **kwargs):
        super(VanillaVAE, self).__init__(*args, **kwargs)

        self.latent_dim = latent_dim
        self.in_channels = in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.hidden_dims = hidden_dims
        
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )

            in_channels = h_dim
        

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        
        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)
        decoder_hidden_dims = hidden_dims[::-1]

        for i in range(len(decoder_hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_hidden_dims[i], decoder_hidden_dims[i+1], kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.BatchNorm2d(decoder_hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_hidden_dims[-1], decoder_hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(decoder_hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                decoder_hidden_dims[-1], 3, kernel_size=3, padding=1
            ),
            nn.Tanh()
        )
    

    def encode(self, input : torch.tensor) -> list[torch.tensor]:
        """
        Encodes the input image using the encoder network. 
        Returns the latent vector representative of the input.

        Args:
            input (torch.tensor): Input tensor/image to the encoder [B x C x H x W]

        Returns:
            list[torch.tensor]: List of latent codes for the batch
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split into mu and var components
        mu = self.fc_mu(result)
        logvar = self.fc_var(result)

        return [mu, logvar]


    def decode(self, z : torch.tensor) -> torch.tensor:
        """
        Maps the given latent vector into the image space. 

        Args:
            z (torch.tensor): Latent vector [B x D]

        Returns:
            torch.tensor: Generated image [B x C x H x W]
        """

        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu : torch.tensor, logvar : torch.tensor) -> torch.tensor:
        """
        Reparameterization trick to align encoder output to Gaussian distribution. 

        Args:
            mu (torch.tensor): mean [B x D]
            logvar (torch.tensor): Standard deviation [B x D]

        Returns:
            torch.tensor: latent vector [B x D]
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input : torch.tensor, **kwargs) -> list[torch.tensor]:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return [self.decode(z), mu, logvar]



class ResNet_VAE(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(ResNet_VAE, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.latent_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.latent_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.latent_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.latent_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )


    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)

        return x_reconst, mu, logvar