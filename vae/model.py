import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

DEVICE = None
"""Device in which to run the model."""

# Choose the device to run the model: CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


BATCH_SIZE = 64
"""
Batch size for training.
The batch size is the number of samples that will be propagated through the network at once.
"""


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

        self.latent_dim = latent_dim

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

    def train_model(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        num_epochs: int = 10,
    ) -> tuple[list[float], list[float]]:
        """Train the VAE model.
        Args:
            train_dataloader (DataLoader): DataLoader for the training dataset.
            val_dataloader (DataLoader): DataLoader for the validation dataset.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler.
            num_epochs (int, optional): Number of training epochs. Defaults to 10.
        Returns:
            tuple[list[float], list[float]]: Histories of training and validation losses.
        """

        train_loss_history = []
        val_loss_history = []

        for epoch in range(num_epochs):
            # --- Training Phase ---
            self.train()
            total_train_loss = 0
            for images in train_dataloader:
                images = images.to(DEVICE)

                # Forward pass
                reconstructed_images, mu, logvar = self(images)

                # Calculate loss
                reconstruction_loss = F.binary_cross_entropy(
                    reconstructed_images, images, reduction="sum"
                )
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (reconstruction_loss / len(images)) + (kl_divergence_loss * 1e-4) / len(
                    images
                )

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() / len(images)

            avg_train_loss = total_train_loss / len(train_dataloader)
            train_loss_history.append(avg_train_loss)

            # --- Validation Phase ---
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for images in val_dataloader:
                    images = images.to(DEVICE)
                    reconstructed_images, mu, logvar = self(images)
                    reconstruction_loss = F.binary_cross_entropy(
                        reconstructed_images, images, reduction="sum"
                    )
                    kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    val_loss = (reconstruction_loss / len(images)) + (
                        kl_divergence_loss * 1e-4
                    ) / len(images)
                    total_val_loss += val_loss.item() / len(images)

            avg_val_loss = total_val_loss / len(val_dataloader)
            val_loss_history.append(avg_val_loss)

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}"
            )

            # Update the learning rate scheduler
            scheduler.step(avg_val_loss)

        return train_loss_history, val_loss_history

    def generate_images(
        self,
        root_path: str,
        num_images: int = 5,
        file_name: str = "default.png",
    ) -> torch.Tensor:
        """Generate new images by sampling from the latent space.

        Args:
            num_images (int): Number of images to generate.
            root_path (str): Root path to save the generated images.
            file_name (str, optional): File name to save the generated images. Defaults to "default.png".

        Returns:
            torch.Tensor: Generated images.
        """

        # Set the model to evaluation mode
        self.eval()

        # Deactivate gradients for inference
        with torch.no_grad():
            # Generate random latent vectors
            z = torch.randn(num_images, self.latent_dim).to(DEVICE)

            # Decode the latent vectors to generate images
            z_expanded = self.decoder_input(z)
            generated_images = self.decoder(z_expanded)

            # Plot the original and reconstructed images
            _, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

            for i in range(num_images):
                # Original images
                ax = axes[0, i]
                ax.imshow(generated_images[i].cpu().squeeze(), cmap="gray")
                ax.axis("off")
                ax.set_title("Original")

                # Reconstructed images
                ax = axes[1, i]
                ax.imshow(generated_images[i].cpu().squeeze(), cmap="gray")
                ax.axis("off")
                ax.set_title("Reconstructed")

            plt.tight_layout()

            # Create directory to save reconstructed images if it doesn't exist
            os.makedirs(root_path, exist_ok=True)

            # Save the reconstructed images
            plt.savefig(file_name)
            plt.close()

        return generated_images

    def plot_reconstructions(
        self,
        dataloader: DataLoader,
        root_path: str,
        num_images: int = 8,
        file_name: str = "default.png",
    ) -> None:
        """Plot original and reconstructed images.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.
            num_images (int, optional): Number of images to plot. Defaults to 8.
            root_path (str): Root path to save the reconstructed images.
            file_name (str, optional): File name to save the reconstructed images. Defaults to "
        """

        # Set the model to evaluation mode
        self.eval()

        with torch.no_grad():
            # Get a batch of images from the dataloader
            images = next(iter(dataloader)).to(DEVICE)
            images = images[:num_images]

            # Reconstruct the images using the VAE
            reconstructed_images, _, _ = self(images)

            # Plot the original and reconstructed images
            _, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

            for i in range(num_images):
                # Original images
                ax = axes[0, i]
                ax.imshow(images[i].cpu().squeeze(), cmap="gray")
                ax.axis("off")
                ax.set_title(f"Generated {i + 1}")

            plt.tight_layout()

            # Create directory to save reconstructed images if it doesn't exist
            os.makedirs(root_path, exist_ok=True)

            # Save the reconstructed images
            plt.savefig(file_name)
            plt.close()
