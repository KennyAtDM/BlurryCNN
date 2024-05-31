import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import imgaug.augmenters as iaa

class BlurDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Populate image paths and labels
        for subdir, label in [('sharp', 0),("sharp_backblur",0),('defocused_blurred', 1), ('motion_blurred', 1),('output_blurry', 1),('blur_frontblur', 1)]:
            subdir_path = os.path.join(root_dir, subdir)
            for file_name in os.listdir(subdir_path):
                if file_name.endswith(('jpg', 'jpeg', 'png')):
                    self.image_paths.append(os.path.join(subdir_path, file_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
