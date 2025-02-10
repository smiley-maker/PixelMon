from src.utils.dependencies import *
from src.model_architectures.VAE.model.vae_model import VanillaVAE
from src.training_scripts.train_vae import TrainVAE
from src.data_loaders.anime_handler import AnimeFacesDataset
from src.data_loaders.pokemon_handler import PokemonDataset
from src.data_loaders.landscapes_handler import LandscapeDataset


if __name__ == "__main__":
    # Find available device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")  # Optional: Print to confirm device selection
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device") # Optional: Print to confirm device selection
    else:
        device = torch.device("cpu")
        print("Using CPU device") # Optional: Print to confirm device selection

    # Load model
    model = VanillaVAE(
       in_channels=3,
       latent_dim=32,
       hidden_dims=[32, 64, 128, 256, 512]
    ).to(device)

    model.train() # Put in training mode

    # Load data
    dataset = LandscapeDataset()
    dataloader = DataLoader(dataset, batch_size=128, drop_last=True)
    print(len(dataloader))

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Train model
    trainer = TrainVAE(
        model=model,
        optimizer=optimizer,
        epochs=80,
        batch_size=128,
        data=dataloader, 
        xdim=(3,64,64),
        device=device,
        write_results="src/results/training/80epochspokemonVAEBig",
        save_images=True,
    )

    trainer.train_model()