import os
import shutil
import zipfile

import requests
import torch
from PIL import Image
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    """Dataset of images for VAE.

    Args:
        Dataset (torch.utils.data.Dataset): PyTorch Dataset class.
    """

    def __init__(self, url, dataset_path, transform=None) -> None:
        """Initialize the dataset. Assigns URL, dataset path, optional
        transformations, downloads the dataset and lists image files.


        Args:
            url (string): URL to the Kaggle dataset download.
            dataset_path (string): Directory for the dataset.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        """

        self.url = url
        self.dataset_path = dataset_path
        self.transform = transform

        self.download()

        # List all image files in the dataset directory and store their names if they are images
        self.images = [
            file
            for file in os.listdir(self.dataset_path)
            if os.path.isfile(os.path.join(self.dataset_path, file))
            and file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, index) -> Image.Image:
        """Get a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Image.Image: The requested image.
        """
        # Handle case where index is a tensor
        # (A tensor is a multi-dimensional array used in PyTorch for data representation)
        if torch.is_tensor(index):
            index = index.item()

        # Ensure index is an integer
        index = int(index)

        # Get the image name for the sample
        img_name = self.images[index]
        img_path = os.path.join(self.dataset_path, img_name)

        # Load the image
        image = Image.open(img_path)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image

    def download(self) -> None:
        """Download the dataset from the Kaggle URL and extract it."""

        output_zip = "./dataset.zip"
        extract_to = "."

        # Download the zip file
        response = requests.get(self.url)
        with open(output_zip, "wb") as f:
            f.write(response.content)

        # Extract the zip file
        with zipfile.ZipFile(output_zip, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        # Move dataset to ./data
        data_src = os.path.join(extract_to, "extracted", self.dataset_path)
        data_dst = os.path.join(extract_to, self.dataset_path)
        shutil.move(data_src, data_dst)

        # (Optional) Remove the ./extracted folder and downloaded zip
        shutil.rmtree(os.path.join(extract_to, "extracted"))
        os.remove(output_zip)
