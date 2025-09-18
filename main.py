import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Configuration

"""Device in which to run the model."""
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Windows

# MacOS with Apple Silicon
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Apple Silicon GPU via Metal
else:
    DEVICE = torch.device("cpu")  # Fallback

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

        #  Get the annotations for the sample
        suite_id = self.annotations.iloc[index, 0]
        sample_id = self.annotations.iloc[index, 1]
        code = self.annotations.iloc[index, 2]

        # Load the image
        img_name = f"input_{suite_id}_{sample_id}_{code}.jpg"
        img_path = os.path.join(self.root_dir, img_name)

        # Open the image using PIL
        image = Image.open(img_path)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Our VAE only needs the image, not the label
        # So we return only the image
        return image


class VAE(torch.nn.Module):
    """Variational Autoencoder (VAE) model.

    Args:
        torch (Module): PyTorch module.
    """

    def __init__(self, latent_dim: int) -> None:
        """Initialize the VAE model.

        Args:
            latent_dim (int): Dimension of the latent space.
        """

        super(VAE, self).__init__()

        # Encoder: Convolutional layers that compress the image
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.Flatten(),
        )

        # Layers for the mean and log-variance of the latent space
        self.fc_mu = torch.nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = torch.nn.Linear(128 * 8 * 8, latent_dim)

        # Decoder
        self.decoder_input = torch.nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = torch.nn.Sequential(
            torch.nn.Unflatten(1, (128, 8, 8)),
            torch.nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.Sigmoid(),  # Output values between [0, 1]
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (torch.Tensor): Mean of the latent space.
            logvar (torch.Tensor): Log-variance of the latent space.

        Returns:
            torch.Tensor: Sampled latent vector.
        """

        # Standard deviation
        std = torch.exp(0.5 * logvar)

        # Sample epsilon from a standard normal distribution
        eps = torch.randn_like(std)

        # Sampled latent vector
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed image, mean, and log-variance.
        """
        # Encode the input
        x = self.encoder(x)

        # Get the mean and log-variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterize to get the latent vector
        z = self.reparameterize(mu, logvar)

        # Decode the latent vector
        # (First expand it to match the decoder input size)
        z_expanded = self.decoder_input(z)
        reconstructed = self.decoder(z_expanded)

        return reconstructed, mu, logvar

    def generate_images(self, num_images: int) -> torch.Tensor:
        """Generate new images by sampling from the latent space.

        Args:
            num_images (int): Number of images to generate.

        Returns:
            torch.Tensor: Generated images.
        """

        # Set the model to evaluation mode
        self.eval()

        # Deactivate gradients for inference
        with torch.no_grad():
            # Generate random latent vectors
            z = torch.randn(num_images, LATENT_DIM).to(DEVICE)

            # Decode the latent vectors to generate images
            generated_images = self.decoder(z)

        return generated_images

    def plot_reconstructions(self, dataloader: DataLoader, num_images: int = 8) -> None:
        """Plot original and reconstructed images.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.
            num_images (int, optional): Number of images to plot. Defaults to 8.
        """

        # Set the model to evaluation mode
        self.eval()

        with torch.no_grad():
            # Get a batch of images from the dataloader
            images = next(iter(dataloader)).to(DEVICE)
            images = images[:num_images]

            # Reconstruct the images using the VAE
            reconstructed_images, _, _ = model(images)

            # Plot the original and reconstructed images
            fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

            for i in range(num_images):
                # Original images
                ax = axes[0, i]
                ax.imshow(images[i].cpu().squeeze(), cmap="gray")
                ax.axis("off")
                ax.set_title("Original")

                # Reconstructed images
                ax = axes[1, i]
                ax.imshow(reconstructed_images[i].cpu().squeeze(), cmap="gray")
                ax.axis("off")
                ax.set_title("Reconstructed")

            plt.tight_layout()
            plt.show()

    def plot_generated_images(self, num_images: int = 8) -> None:
        """Plot generated images.

        Args:
            num_images (int, optional): Number of images to plot. Defaults to 8.
        """

        # Generate new images
        generated_images = self.generate_images(num_images)

        # Plot the generated images
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))

        for i in range(num_images):
            ax = axes[i]
            ax.imshow(generated_images[i].cpu().squeeze(), cmap="gray")
            ax.axis("off")
            ax.set_title("Generated")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Define the transformations for the dataset
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # Ensure image is grayscale
            transforms.ToTensor(),  # Ensure image is converted to tensor
            # transforms.Normalize((0.5,), (0.5,)),  # Normalize the image to [-1, 1]
        ]
    )

    # Create the dataset
    dataset = ChineseMNSIT(
        csv_file="data/chinese_mnist.csv", root_dir="data/images/", transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initizialize the VAE model
    model = VAE(latent_dim=LATENT_DIM).to(DEVICE)

    # Define the optimizer (Adam for better convergence)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training on device: {DEVICE}")
    print(f"Number of samples in dataset: {len(dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of epochs: {EPOCHS}")
    print(f"Latent dimension: {LATENT_DIM}")

    print("\n")
    print("-" * 50)
    print(" " * 15 + "Starting Training")
    print("-" * 50)
    print("\n")

    # Training loop
    for epoch in range(EPOCHS):
        # Set the model to training mode
        model.train()

        # Track the training loss
        train_loss = 0

        # Iterate over the data loader
        #    i = batch index
        #    (images) = batch of images
        for i, (images) in enumerate(dataloader):
            # Move the images to the device (GPU or CPU)
            images = images.to(DEVICE)

            # Forward pass
            reconstructed_images, mu, logvar = model(images)

            # Calculate the reconstruction loss (Binary Cross Entropy)
            reconstructuion_loss = F.binary_cross_entropy(
                reconstructed_images, images, reduction="sum"
            )

            # Calculate the KL divergence loss
            kl_divergenece_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Total loss (the sum of reconstruction and KL divergence losses)
            loss = reconstructuion_loss + kl_divergenece_loss

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Backpropagation (Calculate gradients)
            optimizer.step()  # Update weights

            train_loss += loss.item()

        avg_loss = train_loss / len(dataset)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # Plot some reconstructions
    model.plot_reconstructions(dataloader, num_images=8)
