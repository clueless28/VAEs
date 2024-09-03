import os
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        self.images = [img for img in self.images if self.is_rgb_image(os.path.join(root_dir, img))]

    def __len__(self):
        return len(self.images)
    
    def is_rgb_image(self, img_path):
        with Image.open(img_path) as img:
            return img.mode == 'RGB'

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# Initialize dataset
def loader(path, batch_size):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = CustomImageDataset(root_dir=path, transform=transform)
    # Train-validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # DataLoaders for train and validation
    train_loader = DataLoader(train_dataset, batch_size =batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)
    return train_loader, val_loader