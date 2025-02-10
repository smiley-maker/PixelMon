#!/usr/bin/env python3

from src.utils.dependencies import *

PIXELART_IMAGES = "/Users/jordan/Data/anime_dataset/anime_faces/"

class AnimeFacesDataset(Dataset):
    def __init__(self, img_size=64) -> None:
        self.img_size = img_size
        self.image_dir = PIXELART_IMAGES
        self.image_paths = sorted(self._find_files_(self.image_dir))[:10000]
        print(f"The Anime Faces Dataset has {len(self.image_paths)} images.")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index) -> torch.tensor:
        image_path = self.image_paths[index]
        x = io.imread(image_path)
        if x.ndim == 2: # Convert grayscale to RGB
            x = np.stack([x] * 3, axis=-1)

        x = transform.resize(x, (self.img_size, self.img_size, 3), anti_aliasing=True)        
        x = torch.tensor(x).permute(2, 0, 1).float() # Channels first for PyTorch
        x = (x - x.min()) / (x.max() - x.min()) # Ensure pixel values are in [0, 1]

        return x
    
    def _find_files_(self, image_dir, pattern="*.jpg"):
        img_path_list = []
        for root, dirnames, filenames in os.walk(image_dir):
            for filename in fnmatch.filter(filenames, pattern):
                img_path_list.append(os.path.join(root, filename))
        
        return img_path_list


if __name__ == "__main__":
    print("Data Handler was run directly, initiating example...")
    dataset = AnimeFacesDataset()

    print("Selecting a random image from the dataset...")
    x = dataset[random.randrange(0, len(dataset))]

    plt.figure(figsize=(5, 5))
    plt.imshow(x.permute(1, 2, 0).numpy())
    plt.show()