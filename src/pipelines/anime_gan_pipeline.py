#!/usr/bin/env python3

from src.utils.dependencies import *
from src.data_loaders.pixelart_handler import PixelArtDataset
from src.model_architectures.GAN.dcgan import Generator, Discriminator, initialize_weights
from src.training_scripts.train_gan import TrainGAN
from src.data_loaders.anime_handler import AnimeFacesDataset



if __name__ == "__main__":
    device = torch.device("mps")
    print(f"Using {device}")
    LR = 2e-4
    BATCH_SIZE = 64
    IMAGE_SIZE = 64
    CHANNELS_IMG = 3
    Z_DIM = 100
    NUM_EPOCHS = 3
    FEATURES_DISC = 64
    FEATURES_GEN = 64

    trans = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
            )
        ]
    )

    dataset = ImageFolder(root="/Users/jordan/Data/anime_dataset/", transform=trans)
#    dataset = AnimeFacesDataset(img_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
    initialize_weights(gen)
    initialize_weights(disc)
    gen.train()
    disc.train()

    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))

    trainer = TrainGAN(
        disc, gen, opt_disc, opt_gen, NUM_EPOCHS, BATCH_SIZE, dataloader, Z_DIM, device, write_results="src/results/training/animeGANTest_3epochs/"
    )

    trainer.train_model()