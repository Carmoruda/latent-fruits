import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Configuration

"""Device in which to run the model."""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Batch size for training.
The batch size is the number of samples that will be propagated through the network at once.
"""
BATCH_SIZE = 64

"""
Number of epochs to train the model.
An epoch is a full pass through the training dataset.
"""
EPOCHS = 10

"""
Dimension of the latent space.
The latent space is a lower-dimensional representation of the input data.
"""
LATENT_DIM = 128


class ChineseMNSIT(Dataset):
    def __init__(self, csv_file, root_dir, transform=None) -> None:
        """Dataset for Chinese MNIST.

        Args:
            csv_file (string): Path to the CSV file containing annotations (labels, image paths...).
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        """

        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.annotations)

    from typing import Union

    def __getitem__(self, index) -> Image.Image:
        """Get a sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Image.Image: The image.
        """

        # Handle case where index is a tensor
        # (A tensor is a multi-dimensional array used in PyTorch for data representation)
        if torch.is_tensor(index):
            index = index.tolist()

        # Get the image filename from the CSV file
        img_filename = str(self.annotations.iloc[index, 0])

        # Load the image
        img_name = os.path.join(self.root_dir, img_filename)

        # Open the image using PIL
        image = Image.open(img_name)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Our VAE only needs the image, not the label
        # So we return only the image
        return image


if __name__ == "__main__":
    # Define the transformations for the dataset
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # Ensure image is grayscale
            transforms.ToTensor(),  # Ensure image is converted to tensor
            transforms.Normalize((0.5,), (0.5,)),  # Normalize the image to [-1, 1]
        ]
    )

    # Create the dataset
    dataset = ChineseMNSIT(csv_file="chinese_mnist.csv", root_dir="data/", transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
