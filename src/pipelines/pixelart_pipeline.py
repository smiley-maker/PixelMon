#!/usr/bin/env python3

from src.utils.dependencies import *
from src.data_loaders.pokemon_handler import PokemonDataset
from src.data_loaders.landscapes_handler import LandscapeDataset
from src.data_loaders.pixelart_handler import PixelArtDataset
from src.model_architectures.VAE.model.vae_model import VanillaVAE
from src.training_scripts.train_vae import TrainVAE

def visualize_latent_space(model, data_loader, device, num_samples=100):
    model.eval()
    latents = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
          if len(latents) > num_samples:
            break
          data = data.to(device)#.view(data.size(0), -1)
          mu, _ = model.encode(data)
          latents.append(mu)

    latents = torch.cat(latents, dim=0).cpu().numpy()
    tsne = TSNE(n_components=3, verbose=1)
    tsne_results = tsne.fit_transform(latents)
    fig = px.scatter_3d(tsne_results, x=0, y=1, z=2)
    fig.update_layout(title='VAE Latent Space with TSNE',
                        width=600,
                        height=600)

    fig.show()


def scale_image(image_data):
    """
    Scales image data with values in the range of -1 to 1 to the range of 0 to 255.

    Args:
        image_data: A NumPy array containing the image data.

    Returns:
        A NumPy array containing the scaled image data.
    """

    min_val = -1
    max_val = 1
    new_min_val = 0
    new_max_val = 255

    # Calculate scaling factor
    scaling_factor = (new_max_val - new_min_val) / (max_val - min_val)

    # Scale the image data
    scaled_data = (image_data - min_val) * scaling_factor

    # Clip values to the range 0-255
    scaled_data = np.clip(scaled_data, new_min_val, new_max_val)

    # Convert to integers (optional)
    scaled_data = scaled_data.astype(np.uint8)

    return scaled_data.transpose((1, 2, 0))


def generate_sample(model):
    """
    Generates a sample image by sampling from the latent space and decoding.
    """
    # Sample random noise from standard normal distribution
    z = torch.randn(1, model.latent_dim)

    # Decode the noise to get the generated image
    generated_image = model.decode(z.to("mps"))

    # Move the image tensor to CPU and convert to numpy array for visualization
    generated_image = generated_image.detach().cpu().numpy()

    # Process the image for display (remove extra dimension if present)
    generated_image = generated_image.squeeze() if generated_image.shape[0] == 1 else generated_image

    # Ensure pixel values are in range [0, 255] for visualization
    generated_image = (scale_image(generated_image)).astype(np.uint8)

    # Display the generated image using Matplotlib
    plt.figure(figsize=(5, 5))
    plt.imshow(generated_image)
    plt.axis('off')  # Hide axes for cleaner visualization
    plt.show()

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
       latent_dim=16,
       hidden_dims=[32, 64, 128]
    ).to(device)

    model.train() # Put in training mode

    # Load data
    dataset = PixelArtDataset()
    dataloader = DataLoader(dataset, batch_size=50, drop_last=True)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Train model
    trainer = TrainVAE(
        model=model,
        optimizer=optimizer,
        epochs=150,
        batch_size=50,
        data=dataloader, 
        xdim=(3,16,16),
        device=device,
        write_results="src/results/training/150epochspixelart",
        save_images=True,
    )

    trainer.train_model()

#    torch.save(model, "src/results/models/pixelart150epochs.pt")

    model.eval()

    generate_sample(model)

    visualize_latent_space(model, dataloader, device, num_samples=800)