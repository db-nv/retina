import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import config
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm





def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])

        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std

# def mean_std(loader):
#   images, lebels = next(iter(loader))
#   # shape of images = [b,c,w,h]
#   mean, std = images.mean([0,2,3]), images.std([0,2,3])
#   return mean, std



class DRDataset(Dataset):
    def __init__(self, images_folder, path_to_csv, train=True, transform=None):
        super().__init__()
        self.data = pd.read_csv(path_to_csv)
        self.images_folder = images_folder
        self.image_files = os.listdir(images_folder)
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.data.shape[0] if self.train else len(self.image_files)

    def __getitem__(self, index):
        if self.train:
            image_file, label = self.data.iloc[index]
        else:
            # if test simply return -1 for label, I do this in order to
            # re-use same dataset class for test set submission later on
            image_file, label = self.image_files[index], -1
            image_file = image_file.replace(".jpg", "")

        image = np.array(Image.open(os.path.join(self.images_folder, image_file+".jpg")))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


if __name__ == "__main__":
    """
    Test if everything works ok
    """
    dataset = DRDataset(
        images_folder="/home/ec2-user/SageMaker/SolutionAge/data/training/",
        path_to_csv="/home/ec2-user/SageMaker/SolutionAge/data/trainLabels.csv",
        transform=config.val_transforms,
    )
    loader = DataLoader(
        dataset=dataset, batch_size=32, num_workers=2, shuffle=True, pin_memory=True
    )

#     for x, label in tqdm(loader):
#         print(x.shape)
#         print(label.shape)
#         import sys
#         sys.exit()
        
    mean, std = get_mean_std(loader)
    print(" This the mean & std", mean, std)


