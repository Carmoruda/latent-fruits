import os
import shutil
import zipfile
from typing import Optional

import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CVAEDataset(Dataset):
    """Dataset of images for VAE.

    Args:
        Dataset (torch.utils.data.Dataset): PyTorch Dataset class.
    """

    def __init__(
        self,
        url: str,
        dataset_path: str,
        transform: Optional[transforms.Compose] = None,
        label: int = 0,
        extracted_folder: str = "default",
        download: bool = False,
    ) -> None:
        """Initialize the dataset. Assigns URL, dataset path, optional
        transformations, downloads the dataset and lists image files.


        Args:
            url (string): URL to the Kaggle dataset download.
            dataset_path (string): Directory for the dataset.
            transform (transforms.Compose, optional): Optional transform to be applied on a sample. Defaults to None.
            label (int, optional): The label to assign to all images in the dataset. Defaults to 0.
            extracted_folder (str, optional): The folder name inside the extracted zip where images are located. Defaults to "default".
            download (bool, optional): Whether to download the dataset. Defaults to False.
        """

        self.url = url
        self.dataset_path = dataset_path
        self.transform = transform
        self.label = label

        if download:
            self.download(extracted_folder)

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
        number = len(self.images)

        if number <= 0:
            raise ValueError(
                f"No se encontraron imágenes en el directorio '{self.dataset_path}'. Verifica que la descarga y extracción fue exitosa y que hay imágenes en la carpeta."
            )

        return number

    def __getitem__(self, index) -> tuple[Image.Image, int]:
        """Get a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple[Image.Image, int]: The requested image and its label.
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

        # For CVAE, we return the image and its label
        return image, self.label

    def download(self, extracted_folder: str) -> None:
        """Download the dataset from the Kaggle URL and extract it.

        Args:
            extracted_folder (str): Path to the folder where the dataset will be extracted.

        Raises:
            RuntimeError: If the dataset cannot be downloaded or extracted.
        """

        print("Downloading and extracting dataset...")

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
        extracted_root = os.path.join(extract_to, extracted_folder)
        data_dst = os.path.join(extract_to, self.dataset_path)

        # Search for the folder containing images inside 'extracted'
        data_src = None
        for root, dirs, files in os.walk(extracted_root):
            if any(
                file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
                for file in files
            ):
                data_src = root
                break

        if data_src is None:
            raise RuntimeError(
                f"No se encontró ninguna carpeta con imágenes dentro de '{extracted_root}'."
            )

        # Create destination directory if it doesn't exist
        if not os.path.exists(data_dst):
            os.makedirs(data_dst)

        # Move all images to data_dst
        for file in os.listdir(data_src):
            src_file = os.path.join(data_src, file)
            dst_file = os.path.join(data_dst, file)
            if os.path.isfile(src_file):
                shutil.move(src_file, dst_file)

        # Delete the extracted folder and zip file to clean up
        shutil.rmtree(extracted_root)
        os.remove(output_zip)

        print(f"Dataset downloaded and extracted to '{data_dst}'.")
