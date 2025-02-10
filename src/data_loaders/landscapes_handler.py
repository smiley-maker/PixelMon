#!/usr/bin/env python3

from src.utils.dependencies import *


LANDSCAPE_IMAGE_PATH = "/Users/jordan/Data/pokemon/dataset/train/"
class LandscapeDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.image_dir = LANDSCAPE_IMAGE_PATH
        self.image_paths = sorted(self._find_files_(self.image_dir))
        print(len(self.image_paths))
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index : int) -> torch.tensor:
        image_path = self.image_paths[index]
        x = io.imread(image_path)
        x.resize((3, 64, 64))
        x = torch.tensor(x).float()  #normalizes to be between 0 and 1.
        x = (x - x.min()) / (x.max() - x.min())
        return x
    
    def _find_files_(self, image_dir, pattern="*.jpg"):
        """
        Finds all image files matching the specified pattern within the given directory and its subdirectories.

        Args:
            image_dir (str): The root directory to search for image files.
            pattern (str, optional): The filename pattern to match (default: "*.jpeg").

        Returns:
            list: A list of paths to the found image files.
        """

        img_path_list = []

        # Iterate over the root directory and its subdirectories
        for root, _, filenames in os.walk(image_dir):
            # Filter filenames based on the pattern
            filtered_files = fnmatch.filter(filenames, pattern)

            # Append full paths to the list
            img_path_list.extend([os.path.join(root, file) for file in filtered_files])

        return img_path_list
