#!/usr/bin/env python3

from src.utils.dependencies import *
#from ..utils.dependencies import *

#POKEMON_IMAGE_PATH = "/Users/jordan/Data/pokemon_dataset/images/"
POKEMON_IMAGE_PATH = "/Users/jordan/Data/pokemon_images/data_ready/"
POKEMON_DATA_PATH = "/Users/jordan/Data/pokemon_dataset/pokemon.csv"

class PokemonDataset(Dataset):
    def __init__(self):
        self.image_dir = POKEMON_IMAGE_PATH
        self.image_paths = sorted(self._find_files_(self.image_dir))
        self.pokemon_df = pd.read_csv(POKEMON_DATA_PATH)
        self.pokemon_df.set_index("Name", inplace=True)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
#        pokemon_name = image_path.split("/")[-1].replace('.png', '')
#        pokemon_type = self.pokemon_df.loc[pokemon_name]["Type1"]

        x = io.imread(image_path)
        x.resize((120, 120, 3))
        x = torch.tensor(x).type(torch.IntTensor)
#        x = x[:, :, :3]
        xmin, xmax = torch.min(x), torch.max(x)
        x_norm = (x - xmin) / (xmax - xmin)

#        x = torch.reshape(x, (3, 120, 120))

        return x#, pokemon_type, pokemon_name
    
    def _find_files_(self, image_dir, pattern="*.JPG"):
        img_path_list = []
        for root, dirnames, filenames in os.walk(image_dir):
            for filename in fnmatch.filter(filenames, pattern):
                img_path_list.append(os.path.join(root, filename))
        
        return img_path_list


if __name__ == "__main__":
    print("Data Handler was run directly, initiating example...")
    dataset = PokemonDataset()
    print("CSV data has a format of:")
    print(dataset.pokemon_df.head())

    print("Selecting a random Pokemon from the dataset...")
    x = dataset[random.randrange(0, len(dataset))]

    plt.figure(figsize=(5, 5))
    plt.imshow(x)
    plt.show()