import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
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


class CVAE(torch.nn.Module):
    """Variational Autoencoder (VAE) model.

    Args:
        torch (Module): PyTorch module.
    """

    def __init__(self, latent_dim: int, n_classes: int = 1, n_components: int = 10, beta: float = 1e-3) -> None:
        """Initialize the VAE model.

        Args:
            latent_dim (int): Dimension of the latent space.
            n_classes (int): Number of classes for the classification head. Defaults to 1.
            n_components (int): Number of components for the Gaussian Mixture Model. Defaults to 10.
            beta (float): Weight for the KL divergence term in the loss function. Defaults to 1e-3.
        """

        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.n_components = n_components
        self.gmm: dict[int, GaussianMixture] = {}
        self.lr_history: list[float] = []
        self.current_lr: Optional[float] = None
        self.beta = beta

        super(CVAE, self).__init__()

        # Encoder: Convolutional layers that compress the image
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        encoder_flattened_dim = 256 * 12 * 12

        # Layers for the mean and log-variance of the latent space
        self.fc_mu = torch.nn.Linear(encoder_flattened_dim + n_classes, latent_dim)
        self.fc_logvar = torch.nn.Linear(encoder_flattened_dim + n_classes, latent_dim)

        # Decoder
        self.decoder_input = torch.nn.Linear(latent_dim + n_classes, encoder_flattened_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Unflatten(1, (256, 12, 12)),
            torch.nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, output_padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1, output_padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1, output_padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                32, 1, kernel_size=9, stride=1, padding=2, output_padding=0
            ),
            # Sigmoid keeps pixel intensities inside [0, 1] so they match the dataset after ToTensor().
            torch.nn.Sigmoid(),
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

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): One-hot encoded class labels.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed image, mean, and log-variance.
        """
        # Encode the input
        x_encoded = self.encoder(x)

        # Concatenate the class labels with the encoded input
        x_combined = torch.cat([x_encoded, y], dim=1)

        # Get the mean and log-variance
        mu = self.fc_mu(x_combined)
        logvar = self.fc_logvar(x_combined)

        # Reparameterize to get the latent vector
        z = self.reparameterize(mu, logvar)

        # Concatenate the class labels with the latent vector
        z_combined = torch.cat([z, y], dim=1)

        # Decode the latent vector
        # (First expand it to match the decoder input size)
        z_expanded = self.decoder_input(z_combined)
        reconstructed = self.decoder(z_expanded)

        return reconstructed, mu, logvar

    def fit_gmm(self, train_dataloader: DataLoader) -> None:
        """Fit a Gaussian Mixture Model to the latent space of the training data.

        Args:
            train_dataloader (DataLoader): DataLoader for the training dataset.
        """
        # Evaluate the model to get latent representations
        self.eval()

        # Collect latent vectors by class
        latent_by_label: dict[int, list[np.ndarray]] = defaultdict(list)

        # Disable gradient computation
        with torch.no_grad():
            # Iterate over the training data to collect latent vectors
            for images, labels in train_dataloader:
                # Put images and labels on the correct device
                images = images.to(DEVICE)

                # One-hot encode the labels
                y_onehot = F.one_hot(labels, num_classes=self.n_classes).float().to(DEVICE)

                # Get the latent vectors (mu)
                _, mu, _ = self(images, y_onehot)

                # Convert to numpy for GMM fitting
                mu_np = mu.cpu().numpy()

                # Get the labels
                labels_np = labels.cpu().numpy()

                # Group latent vectors by their labels
                for class_idx in range(self.n_classes):
                    # Create a mask for the current class
                    class_mask = labels_np == class_idx

                    # Append the latent vectors for the current class
                    if np.any(class_mask):
                        latent_by_label[class_idx].append(mu_np[class_mask])

        # Fit a GMM for each class
        self.gmm = {}

        # Fit a GMM for each class
        print("Fitting class-conditional GMMs...")
        for class_idx in range(self.n_classes):
            class_latents = latent_by_label.get(class_idx, [])
            if not class_latents:
                warnings.warn(
                    f"No latent samples collected for class {class_idx}; falling back to standard normal sampling.",
                    RuntimeWarning,
                )
                continue

            samples = np.concatenate(class_latents, axis=0)
            class_gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type="diag",
                reg_covar=1e-6,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="covariance is not symmetric positive-semidefinite.",
                    category=RuntimeWarning,
                )
                class_gmm.fit(samples)

            self.gmm[class_idx] = class_gmm
            print(f"  · Fitted GMM for class {class_idx} with {samples.shape[0]} samples.")

        if not self.gmm:
            warnings.warn(
                "No class GMMs were fitted successfully; generation will use standard normal sampling.",
                RuntimeWarning,
            )
        else:
            print("GMMs fitted.")

    def train_model(
        self,
        loss_function,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        epoch: int = 1,
        num_epochs: int = 10,
    ) -> tuple[list[float], list[float]]:
        """Train the VAE model.
        Args:
            train_dataloader (DataLoader): DataLoader for the training dataset.
            val_dataloader (DataLoader): DataLoader for the validation dataset.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler.
            epoch (int, optional): Number of epoch in training.
            num_epochs (int, optional): Total number of epochs for training. Defaults to 10.
            loss_function (object, optional): Loss function to use. Defaults to F.mse_loss.
        Returns:
            tuple[list[float], list[float]]: Histories of training and validation losses.
        """

        train_loss_history = []
        val_loss_history = []

        if epoch == 0 and self.lr_history:
            # Reset between independent training sessions
            self.lr_history.clear()

        if loss_function is None:
            loss_function = F.mse_loss

        # --- Training Phase ---
        self.train()
        total_train_loss = 0
        for images, labels in train_dataloader:
            images = images.to(DEVICE)

            # One-hot encode the labels
            # (Convert labels to matrix of size [batch_size, n_classes])
            y_onehot = F.one_hot(labels, num_classes=self.n_classes).float().to(DEVICE)

            # Forward pass
            reconstructed_images, mu, logvar = self(images, y_onehot)

            # Calculate loss
            reconstruction_loss = loss_function(reconstructed_images, images, reduction="sum")

            kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (reconstruction_loss / len(images)) + (kl_divergence_loss * self.beta) / len(
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
            for images, labels in val_dataloader:
                images = images.to(DEVICE)
                y_onehot = F.one_hot(labels, num_classes=self.n_classes).float().to(DEVICE)
                reconstructed_images, mu, logvar = self(images, y_onehot)
                reconstruction_loss = loss_function(
                    reconstructed_images, images, reduction="sum"
                )
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                val_loss = (reconstruction_loss / len(images)) + (
                    kl_divergence_loss * self.beta
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

        current_lr = optimizer.param_groups[0]["lr"]
        self.lr_history.append(current_lr)
        self.current_lr = current_lr

        return train_loss_history, val_loss_history

    def generate_images(
        self,
        root_path: str,
        labels: list[int],
        file_name: str = "default.png",
    ) -> torch.Tensor:
        """Generate new images by sampling from the latent space.

        Args:
            root_path (str): Root path to save the generated images.
            labels (list[int]): List of labels for the images to generate.
            file_name (str, optional): File name to save the generated images. Defaults to "default.png".

        Returns:
            torch.Tensor: Generated images.
        """

        num_images = len(labels)

        # Set the model to evaluation mode
        self.eval()

        # Deactivate gradients for inference
        with torch.no_grad():
            samples = []

            # Sample from the GMM if available,
            # otherwise sample from standard normal
            if self.gmm:
                # Sample from the GMM for each label
                for label in labels:
                    # Ensure label is an integer
                    label_idx = int(label)
                    # Get the GMM for the current class
                    class_gmm = self.gmm.get(label_idx)

                    # If no GMM was fitted for this class, fall back to standard normal
                    if class_gmm is None:
                        samples.append(torch.randn(1, self.latent_dim))
                        continue

                    # Sample a latent vector from the class-specific GMM
                    sample, _ = class_gmm.sample(1)
                    tensor_sample = torch.from_numpy(sample).to(device=DEVICE, dtype=torch.float32)
                    samples.append(tensor_sample)

                z = torch.cat(samples, dim=0).to(DEVICE)
            else:
                # Generate random latent vectors from a standard normal distribution
                z = torch.randn(num_images, self.latent_dim).to(DEVICE)

            # One-hot encode the labels
            y_onehot = (
                F.one_hot(torch.tensor(labels), num_classes=self.n_classes).float().to(DEVICE)
            )

            # Concatenate the class labels with the latent vector
            z_combined = torch.cat([z, y_onehot], dim=1)

            # Decode the latent vectors to generate images
            z_expanded = self.decoder_input(z_combined)
            generated_images = self.decoder(z_expanded)

            # Plot the original and reconstructed images
            _, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))

            if num_images == 1:
                axes = [axes]

            for i in range(num_images):
                # Original images
                ax = axes[i]
                ax.imshow(generated_images[i].cpu().squeeze(), cmap="gray")
                ax.axis("off")
                ax.set_title(f"Generated {i + 1}")

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
            images, labels = next(iter(dataloader))
            images = images.to(DEVICE)
            images = images[:num_images]
            labels = labels[:num_images]

            # One-hot encode the labels
            y_onehot = F.one_hot(labels, num_classes=self.n_classes).float().to(DEVICE)

            # Reconstruct the images using the VAE
            reconstructed_images, _, _ = self(images, y_onehot)

            # Plot the original and reconstructed images
            _, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

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

    def plot_learning_rate(self, file_name: str = "lr_schedule.png") -> None:
        """Plot and save the learning rate schedule observed during training."""

        if not self.lr_history:
            warnings.warn("Learning rate history is empty; skipping LR plot.", RuntimeWarning)
            return

        directory = os.path.dirname(file_name)
        if directory:
            os.makedirs(directory, exist_ok=True)

        epochs = range(1, len(self.lr_history) + 1)

        plt.figure(figsize=(6, 4))
        plt.plot(epochs, self.lr_history, marker="o")
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        latent_dim: int,
        n_classes: int,
        *,
        n_components: int = 10,
        device: torch.device = DEVICE,
        map_location: Optional[Union[torch.device, str]] = None,
    ) -> "CVAE":
        """Instantiate a CVAE and load the weights stored in ``checkpoint_path``.

        Args:
            checkpoint_path: Location of the ``.pth`` file produced with ``torch.save``.
            latent_dim: Latent dimensionality used when the model was trained.
            n_classes: Number of classes the model was conditioned on during training.
            n_components: Number of components for the class-conditional GMM. Defaults to 10.
            device: Device where the model should be placed after loading.
            map_location: Optional override for ``torch.load``'s ``map_location``.

        Raises:
            FileNotFoundError: If ``checkpoint_path`` does not exist.

        Returns:
            CVAE: Model with the restored parameters, ready for inference.
        """

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if map_location is None:
            map_location = device

        state_dict = torch.load(checkpoint_path, map_location=map_location)
        model = cls(latent_dim=latent_dim, n_classes=n_classes, n_components=n_components)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
