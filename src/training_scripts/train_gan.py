from src.utils.dependencies import *
from src.data_loaders.mnist_handler import MNIST_DATASET
from src.model_architectures.GAN.dcgan import Generator, Discriminator, initialize_weights

class TrainGAN:
    def __init__(
            self,
            discriminator,
            generator,
            opt_disc,
            opt_gen, 
            epochs : int, 
            batch_size : int,
            data, 
            zdim : int,
            device, 
            write_results : str, 
            save_images = True,
            extra_losses = None
        ) -> None:
        
        self.disc = discriminator
        self.gen = generator
        self.opt_disc = opt_disc
        self.opt_gen = opt_gen
        self.criterion = nn.BCELoss()
        self.epochs = epochs
        self.batch_size = batch_size
        self.data = data
        self.zdim = zdim
        self.device = device
        self.extra_losses = extra_losses
        self.save_images = save_images
        self.writer = SummaryWriter(log_dir=write_results)  # Create writer


    def train_model(self):
        step = 0
        fixed_noise = torch.randn((32, self.zdim, 1, 1)).to(self.device)
        for epoch in range(self.epochs):
            for batch_idx, (real, _) in enumerate(self.data):
                real = real.to(self.device)
                noise = torch.randn((self.batch_size, self.zdim, 1, 1)).to(self.device)
                fake = self.gen(noise)

                ## Train Discriminator max log(D(x)) + log(1 - D(G(z)))
                disc_real = self.disc(real).reshape(-1) 
                loss_disc_real = self.criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = self.disc(fake).reshape(-1)
                loss_disc_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
                loss_disc = (loss_disc_real + loss_disc_fake) / 2

                self.disc.zero_grad()
                loss_disc.backward(retain_graph=True)
                self.opt_disc.step()

                ## Train Generator min log(1 - D(G(z))) <-> max log(D(G(z)))
                output = self.disc(fake).reshape(-1)
                loss_gen = self.criterion(output, torch.ones_like(output))
                self.gen.zero_grad()
                loss_gen.backward()
                self.opt_gen.step()

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch [{epoch}/{self.epochs}] Batch {batch_idx}/{len(self.data)} Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
                    )

                    with torch.no_grad():
                        fake = self.gen(fixed_noise)
                        img_grid_real = make_grid(
                            real[:32], normalize=True
                        )
                        img_grid_fake = make_grid(
                            fake[:32], normalize=True
                        )

#                        self.writer.add_image("Real", img_grid_real, global_step=step)
                        self.writer.add_image("Fake", img_grid_fake, global_step=step)

                    step += 1
        
        self.writer.close()