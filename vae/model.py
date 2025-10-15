from pathlib import Path
from typing import Optional, Sequence, Union

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
        *,
        extra_latent_dims: Optional[Sequence[int]] = None,
    ) -> None:
        """Initialize the VAE model.

        Args:
            latent_dim (int): Dimension of the latent space.
            n_classes (int): Number of classes for the classification head. Defaults to 1.
            n_components (int): Number of components for the Gaussian Mixture Model. Defaults to 10.
            beta (float): Weight for the KL divergence term in the loss function. Defaults to 1e-3.
            extra_latent_dims (Sequence[int] | None): Sizes for additional latent levels (hierarchical VAE).
            latent_dims (list[int]): List of all latent dimensions including the primary one (latent_dim + extra_latent_dims).
            latent_levels (int): Total number of latent levels.
            total_latent_dim (int): Sum of all latent dimensions.
            logvar_min (float): Minimum value for log-variance to ensure numerical stability.
            logvar_max (float): Maximum value for log-variance to ensure numerical stability.
        """

        self.latent_dim: int = latent_dim
        self.n_classes: int = n_classes
        self.n_components: int = n_components
        self.gmm: dict[int, GaussianMixture] = {}
        self.lr_history: list[float] = []
        self.current_lr: Optional[float] = None
        self.beta: float = beta
        self.extra_latent_dims: list[int] = list(extra_latent_dims or [])
        self.latent_dims: list[int] = [latent_dim] + self.extra_latent_dims
        self.latent_levels: int = len(self.latent_dims)
        self.total_latent_dim: int = sum(self.latent_dims)
        self.logvar_min: float = -10.0
        self.logvar_max: float = 10.0

        super(CVAE, self).__init__()

        # Encoder: individual blocks let us capture multi-scale activations later on.
        self.encoder_convs = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=1),
                    torch.nn.ReLU(),
                ),
                torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
                    torch.nn.ReLU(),
                ),
                torch.nn.Sequential(
                    torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                ),
                torch.nn.Sequential(
                    torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                ),
            ]
        )

        # Output dimension after the final convolutional layer
        self.encoder_output_dim = 256 * 12 * 12

        # Channels at each encoder stage (for selecting feature maps for extra latent levels)
        self.encoder_channels = [
            block[0].out_channels  # type: ignore[misc]
            for block in self.encoder_convs  # type: ignore[misc]
        ]

        # Layers for the mean and log-variance of the latent space
        self.fc_mu = torch.nn.Linear(self.encoder_output_dim + n_classes, latent_dim)
        self.fc_logvar = torch.nn.Linear(self.encoder_output_dim + n_classes, latent_dim)

        # Layer to expand the latent vector to match the decoder input size
        self.decoder_input = torch.nn.Linear(
            self.total_latent_dim + n_classes, self.encoder_output_dim
        )

        # Decoder: series of transposed convolutions to reconstruct the input
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

        # Additional latent levels for hierarchical VAE
        self.extra_posterior_mu = torch.nn.ModuleList()  # Posterior mean networks
        self.extra_posterior_logvar = torch.nn.ModuleList()  # Posterior log-variance networks
        self.extra_prior_mu = torch.nn.ModuleList()  # Prior mean networks
        self.extra_prior_logvar = torch.nn.ModuleList()  # Prior log-variance networks
        self.extra_feature_indices: list[int] = []  # Indices of encoder features used

        if self.extra_latent_dims:
            # Select which encoder feature maps to use for the additional latent levels
            self.extra_feature_indices = self._select_feature_indices(
                len(self.extra_latent_dims)
            )

            # Dimensions of the previous latent level (starting with the primary latent_dim)
            prev_latent_dim = self.latent_dim

            for idx, latent_size in enumerate(self.extra_latent_dims):
                # Index of the encoder feature map and feature channels to use for this latent level
                feature_idx = self.extra_feature_indices[idx]
                feature_channels = self.encoder_channels[feature_idx]

                # Input dimension for the posterior networks
                posterior_input_dim = feature_channels + self.n_classes + prev_latent_dim

                # Create the posterior and prior networks for this latent level
                self.extra_posterior_mu.append(self._make_head(posterior_input_dim, latent_size))
                self.extra_posterior_logvar.append(
                    self._make_head(posterior_input_dim, latent_size)
                )

                # Input dimension for the prior networks
                prior_input_dim = prev_latent_dim + self.n_classes

                # Create the prior networks for this latent level
                self.extra_prior_mu.append(self._make_head(prior_input_dim, latent_size))
                self.extra_prior_logvar.append(self._make_head(prior_input_dim, latent_size))

                # Update the previous latent dimension for the next level
                prev_latent_dim = latent_size

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the convolutional encoder and collect intermediate feature maps.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: Encoded representation and feature maps.
        """

        features: list[torch.Tensor] = []

        h = x

        # Pass through each encoder block and collect features
        for block in self.encoder_convs:
            h = block(h)
            features.append(h)

        # Flatten the final feature map for the latent space layers
        flattened = torch.flatten(h, start_dim=1)

        return flattened, features

    def _make_head(self, in_dim: int, out_dim: int) -> torch.nn.Module:
        """Small MLP to map pooled features into latent statistics.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.

        Returns:
            torch.nn.Module: MLP model.
        """
        # Hidden layer dimension (at least 256, at most double the output dimension)
        hidden_dim = max(out_dim * 2, min(in_dim, 256))

        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def _select_feature_indices(self, num_levels: int) -> list[int]:
        """Pick encoder feature maps to drive additional latent levels (deep to shallow)

        Args:
            num_levels (int): Number of additional latent levels.

        Raises:
            ValueError: If the number of levels exceeds available encoder stages.

        Returns:
            list[int]: Indices of selected encoder feature maps.
        """

        # Available encoder stages (0 = shallowest, -1 = deepest)
        available = list(range(len(self.encoder_channels)))

        # Ensure we don't exceed available stages
        if num_levels > len(available):
            raise ValueError(
                "Number of additional latent levels exceeds available encoder stages."
            )

        # Select the deepest available stages for the additional latent levels
        selected = list(reversed(available[-num_levels:])) if num_levels else []

        return selected

    @staticmethod
    def _pool_feature(feature: torch.Tensor) -> torch.Tensor:
        """Global average pool a feature map to a vector representation.

        Args:
            feature (torch.Tensor): Feature map tensor.

        Returns:
            torch.Tensor: Pooled feature vector.
        """

        pooled = torch.nn.functional.adaptive_avg_pool2d(feature, output_size=1)
        return torch.flatten(pooled, start_dim=1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (torch.Tensor): Mean of the latent space.
            logvar (torch.Tensor): Log-variance of the latent space.

        Returns:
            torch.Tensor: Sampled latent vector.
        """

        # Clamp log-variance for numerical stability
        # (we enforce min/max values)
        if hasattr(self, "logvar_min"):
            logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)

        # Standard deviation
        std = torch.exp(0.5 * logvar)

        # Sample epsilon from a standard normal distribution
        eps = torch.randn_like(std)

        # Sampled latent vector
        return mu + eps * std

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        """Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): One-hot encoded class labels.

        Returns:
            tuple: Reconstructed image and hierarchical latent statistics.
        """
        # Encode the input
        x_encoded, features = self.encode(x)

        # Concatenate the class labels with the encoded input
        x_combined = torch.cat([x_encoded, y], dim=1)

        # Get the mean and log-variance
        mu = self.fc_mu(x_combined)
        logvar = self.fc_logvar(x_combined)
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)

        # Reparameterize to get the latent vector
        z = self.reparameterize(mu, logvar)

        # Store latent statistics for all levels
        latent_stats: dict[str, dict[str, torch.Tensor]] = {
            "level_0": {
                "mu": mu,
                "logvar": logvar,
                "prior_mu": torch.zeros_like(mu),
                "prior_logvar": torch.zeros_like(logvar),
                "z": z,
            }
        }

        # For hierarchical VAE, compute additional latent levels
        latent_samples = [z]
        parent_z = z

        if self.extra_latent_dims:
            for level_idx, _ in enumerate(self.extra_latent_dims, start=1):
                # Get the feature map for this latent level
                feature_idx = self.extra_feature_indices[level_idx - 1]
                feature_vec = self._pool_feature(features[feature_idx])

                # Posterior network input: pooled feature + class label + previous latent
                posterior_input = torch.cat([feature_vec, y, parent_z], dim=1)

                # Compute posterior parameters and sample
                mu_extra = self.extra_posterior_mu[level_idx - 1](posterior_input)
                logvar_extra = self.extra_posterior_logvar[level_idx - 1](posterior_input)
                logvar_extra = torch.clamp(
                    logvar_extra, min=self.logvar_min, max=self.logvar_max
                )
                z_extra = self.reparameterize(mu_extra, logvar_extra)

                # Prior network input: previous latent + class label
                prior_input = torch.cat([parent_z, y], dim=1)
                prior_mu = self.extra_prior_mu[level_idx - 1](prior_input)
                prior_logvar = self.extra_prior_logvar[level_idx - 1](prior_input)
                prior_logvar = torch.clamp(
                    prior_logvar, min=self.logvar_min, max=self.logvar_max
                )

                # Store the statistics for this latent level
                latent_stats[f"level_{level_idx}"] = {
                    "mu": mu_extra,
                    "logvar": logvar_extra,
                    "prior_mu": prior_mu,
                    "prior_logvar": prior_logvar,
                    "z": z_extra,
                }

                # Append the sampled latent vector
                latent_samples.append(z_extra)
                parent_z = z_extra

        # Concatenate all latent samples and class labels
        combined_latent = torch.cat(latent_samples, dim=1)
        z_combined = torch.cat([combined_latent, y], dim=1)

        # Decode the latent vector
        # (First expand it to match the decoder input size)
        z_expanded = self.decoder_input(z_combined)
        reconstructed = self.decoder(z_expanded)

        return reconstructed, latent_stats

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        latent_dim: int,
        n_classes: int,
        *,
        n_components: int = 10,
        extra_latent_dims: Optional[Sequence[int]] = None,
        device: torch.device = DEVICE,
        map_location: Optional[Union[torch.device, str]] = None,
    ) -> "CVAE":
        """Instantiate a CVAE and load the weights stored in ``checkpoint_path``.

        Args:
            checkpoint_path: Location of the ``.pth`` file produced with ``torch.save``.
            latent_dim: Latent dimensionality used when the model was trained.
            n_classes: Number of classes the model was conditioned on during training.
            n_components: Number of components for the class-conditional GMM. Defaults to 10.
            extra_latent_dims: Additional latent dimensionalities if the model was hierarchical.
            device: Device where the model should be placed after loading.
            map_location: Optional override for ``torch.load``'s ``map_location``.

        Raises:
            FileNotFoundError: If ``checkpoint_path`` does not exist.

        Returns:
            CVAE: Model with the restored parameters, ready for inference.
        """

        # Ensure checkpoint_path is a Path object
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Default map_location to the target device if not provided
        if map_location is None:
            map_location = device

        # Load the state dictionary from the checkpoint
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        model = cls(
            latent_dim=latent_dim,
            n_classes=n_classes,
            n_components=n_components,
            extra_latent_dims=extra_latent_dims,
        )

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model
