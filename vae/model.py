from pathlib import Path
from typing import Optional, Union

import torch
from sklearn.mixture import GaussianMixture

DEVICE = None
"""Device in which to run the model."""

# Choose the device to run the model: CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class CVAE(torch.nn.Module):
    """Variational Autoencoder (VAE) model.

    Args:
        torch (Module): PyTorch module.
    """

    def __init__(
        self,
        latent_dim: int,
        n_classes: int = 1,
        n_components: int = 10,
        beta: float = 1e-3,
    ) -> None:
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
