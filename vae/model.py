import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
        dataloader: DataLoader,
        dataset: Dataset,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        num_epochs: int = 10,
    ) -> list[float]:
        """Train the VAE model.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.
            dataset (Dataset): Dataset object.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler.
            num_epochs (int, optional): Number of training epochs. Defaults to EPOCHS.

        Returns:
            list[float]: History of training losses.
        """

        loss_history = []

        for epoch in range(num_epochs):
            # Set the model to training mode
            self.train()

            # Track the training loss
            train_loss = 0

            # Iterate over the data loader
            #    i = batch index
            #    (images) = batch of images
            for i, (images) in enumerate(dataloader):
                # Move the images to the device (GPU or CPU)
                images = images.to(DEVICE)

                # Forward pass
                reconstructed_images, mu, logvar = self(images)

                # Calculate the reconstruction loss (Binary Cross Entropy)
                reconstruction_loss = F.binary_cross_entropy(
                    reconstructed_images, images, reduction="sum"
                )

                # print(f"Reconstruction Loss: {reconstruction_loss.item() / len(images)}")

                # Calculate the KL divergence loss
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # print(f"KL Divergence Loss: {kl_divergence_loss.item() / len(images)}")

                # Total loss (the sum of reconstruction and KL divergence losses)
                loss = (reconstruction_loss / len(images)) + (kl_divergence_loss * 1e-4) / len(
                    images
                )

                # print(f"Total Loss: {loss.item()}")

                # Backward pass and optimization
                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()  # Backpropagation (Calculate gradients)
                optimizer.step()  # Update weights

                train_loss += loss.item() / len(images)
                # print(f"train_loss after batch {i}: {train_loss}")

            avg_loss = train_loss / len(dataloader)
            loss_history.append(avg_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

            # Update the learning rate
            scheduler.step(avg_loss)

        return loss_history

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
            z = torch.randn(num_images, self.latent_dim).to(DEVICE)

            # Decode the latent vectors to generate images
            generated_images = self.decoder(z)

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

            # Create directory to save reconstructed images if it doesn't exist
            os.makedirs(root_path, exist_ok=True)

            # Save the reconstructed images
            plt.savefig(file_name)
            plt.close()
