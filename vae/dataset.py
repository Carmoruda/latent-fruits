import os
import shutil
import zipfile
from pathlib import Path
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
        delete_extracted: bool = True,
    ) -> None:
        """Initialize the dataset. Assigns URL, dataset path, optional
        transformations, downloads the dataset and lists image files.


        Args:
            url (string): URL to the Kaggle dataset download.
            dataset_path (Path): Directory for the dataset.
            transform (transforms.Compose, optional): Optional transform to be applied on a sample. Defaults to None.
            label (int, optional): The label to assign to all images in the dataset. Defaults to 0.
            extracted_folder (str, optional): The folder name inside the extracted zip where images are located. Defaults to "default".
            download (bool, optional): Whether to download the dataset. Defaults to False.
        """

        self.url = url
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.label = label

        if download:
            self.download(extracted_folder, delete_extracted)

        self.dataset_path.mkdir(parents=True, exist_ok=True)

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        self.images = sorted(
            file.name
            for file in self.dataset_path.iterdir()
            if file.is_file() and file.suffix.lower() in image_extensions
        )

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
        img_path = self.dataset_path / img_name

        # Load the image and ensure the file handle is closed promptly
        with Image.open(img_path) as image:
            # Apply transformations if any
            if self.transform:
                image = self.transform(image)
            else:
                image = image.copy()  # detach from the context manager

        # For CVAE, we return the image and its label
        return image, self.label

    def download(self, extracted_folder: str, delete_extracted: bool = True) -> None:
        """Download the dataset from the Kaggle URL and extract it.

        Args:
            extracted_folder (str): Path to the folder where the dataset will be extracted.
            delete_extracted (bool, optional): Whether to delete the extracted folder after moving images. Defaults to True.

        Raises:
            RuntimeError: If the dataset cannot be downloaded or extracted.
        """

        print("Downloading and extracting dataset...")

        output_zip = Path("dataset.zip")
        extract_to = Path(".")

        # Download the zip file
        response = requests.get(self.url, stream=True, timeout=30)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"Error downloading dataset from URL: {self.url}") from exc

        # Write the zip file to disk
        with output_zip.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)

        # Extract the zip file
        with zipfile.ZipFile(output_zip, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        # Move dataset to final location
        extracted_root = (extract_to / extracted_folder).resolve()
        data_dst = self.dataset_path.resolve()

        # Search for the folder containing images inside 'extracted'
        image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        data_src: Optional[Path] = None
        for root, _, files in os.walk(extracted_root):
            if any(file.lower().endswith(image_extensions) for file in files):
                data_src = Path(root)
                break

        if data_src is None:
            raise RuntimeError(
                f"No se encontró ninguna carpeta con imágenes dentro de '{extracted_root}'."
            )

        # Create destination directory if it doesn't exist
        data_dst.mkdir(parents=True, exist_ok=True)

        # Move all images to data_dst, de-duplicating names when needed
        for src_file in data_src.iterdir():
            if not src_file.is_file():
                continue

            dst_file = data_dst / src_file.name
            if dst_file.exists():
                dst_file = self._resolve_duplicate_path(dst_file)
                print(
                    f"Archivo duplicado detectado; renombrado '{src_file.name}' a '{dst_file.name}'."
                )

            shutil.move(str(src_file), str(dst_file))

        # Delete the extracted folder and zip file to clean up
        if delete_extracted:
            shutil.rmtree(extracted_root, ignore_errors=True)
            output_zip.unlink(missing_ok=True)

        print(f"Dataset downloaded and extracted to '{data_dst}'.")

    @staticmethod
    def _resolve_duplicate_path(destination: Path) -> Path:
        """Generate a non-conflicting path by appending a numeric suffix."""

        counter = 1
        stem = destination.stem
        suffix = destination.suffix
        parent = destination.parent

        candidate = destination
        while candidate.exists():
            candidate = parent / f"{stem}_{counter}{suffix}"
            counter += 1

        return candidate
