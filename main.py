import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Encoder(nn.Module):
    """Encoder network for VAE.

    Args:
        nn (Module): PyTorch neural network module.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        """Initialize the encoder network.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layer.
            latent_dim (int): Dimension of the latent space.
        """
        # Initialize the parent class
        super(Encoder, self).__init__()

        # Define the layers
        # Fully connected layer from input to hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Fully connected layers for mean and log variance of latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # Fully connected layer for log variance of latent space
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """Forward pass through the encoder.

        Args:
            x (Tensor): Input tensor.

        Returns:
            mu (Tensor): Mean of the latent space.
            logvar (Tensor): Log variance of the latent space.
        """
        # Apply ReLU activation to the hidden layer
        h = torch.relu(self.fc1(x))
        # Compute mean and log variance
        mu = self.fc_mu(h)
        # Compute log variance
        logvar = self.fc_logvar(h)

        # Return the mean and log variance
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network for VAE.

    Args:
        nn (Module): PyTorch neural network module.
    """

    def __init__(self, latent_dim, hidden_dim, output_dim):
        """Initialize the decoder network.

        Args:
            latent_dim (int): Dimension of the latent space.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output data.
        """
        # Initialize the parent class
        super(Decoder, self).__init__()

        # Define the layers
        # Fully connected layer from latent to hidden
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        # Fully connected layer from hidden to output
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        """Forward pass through the decoder.

        Args:
            z (Tensor): Latent space representation.

        Returns:
            Tensor: Reconstructed output.
        """
        # Apply ReLU activation to the hidden layer
        h = torch.relu(self.fc1(z))
        # Apply sigmoid activation to the output layer
        x_hat = torch.sigmoid(self.fc2(h))
        # Return the reconstructed output
        return x_hat


class VAE(nn.Module):
    """Variational Autoencoder (VAE) model.

    Args:
        nn (Module): PyTorch neural network module.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        """Initialize the VAE model.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layer.
            latent_dim (int): Dimension of the latent space.
        """
        # Initialize the parent class
        super(VAE, self).__init__()
        # Initialize the encoder and decoder
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        """Forward pass through the VAE.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Reconstructed output.
        """
        # Encode the input to get mean and log variance
        mu, logvar = self.encoder(x)
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        # Sample from the standard normal distribution
        eps = torch.randn_like(std)
        # Generate latent variable
        z = mu + eps * std
        # Decode the latent variable to get the reconstructed output
        x_hat = self.decoder(z)
        # Return the reconstructed output, mean, and log variance
        return x_hat, mu, logvar


def loss_function(x, x_hat, mu, logvar):
    """Compute the VAE loss function.

    Args:
        x (Tensor): Input tensor.
        x_hat (Tensor): Reconstructed output.
        mu (Tensor): Mean of the latent space.
        logvar (Tensor): Log variance of the latent space.

    Returns:
        Tensor: Total VAE loss.
    """
    # Binary Cross Entropy loss (for reconstruction)
    BCE = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    # Kullback-Leibler Divergence loss (for regularization)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Return the total loss
    return BCE + KLD


if __name__ == "__main__":
    # Hyperparameters

    # Input dimensions (size of flattened MNIST images)
    input_dim = 784
    # Hidden layer dimension (number of neurons in hidden layer)
    hidden_dim = 400
    # Latent space dimension (size of the latent vector)
    latent_dim = 20

    # Learning rate for the optimizer
    lr = 1e-3
    # Batch size for training
    batch_size = 128
    # Number of epochs to train
    epochs = 100

    # Data loader
    # Transform to convert images to tensors and flatten them
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
    )

    # Training dataset and data loader
    train_dataset = datasets.MNIST(
        root="./data/numbers", train=True, transform=transform, download=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model and optimizer (Adam)
    vae = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Training loop
    vae.train()

    for epoch in range(epochs):
        train_loss = 0
        for x, _ in train_loader:
            x = x.view(-1, input_dim)
            optimizer.zero_grad()
            x_hat, mu, logvar = vae(x)
            loss = loss_function(x, x_hat, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}")

    # visualizing reconstructed outputs
    vae.eval()
    with torch.no_grad():
        x, _ = next(iter(train_loader))
        x = x.view(-1, input_dim)
        x_hat, _, _ = vae(x)
        x = x.view(-1, 28, 28)
        x_hat = x_hat.view(-1, 28, 28)

        fig, axs = plt.subplots(2, 10, figsize=(15, 3))
        for i in range(10):
            axs[0, i].imshow(x[i].cpu().numpy(), cmap="gray")
            axs[1, i].imshow(x_hat[i].cpu().numpy(), cmap="gray")
            axs[0, i].axis("off")
            axs[1, i].axis("off")
        plt.show()
    # visualizing generated samples
    with torch.no_grad():
        z = torch.randn(10, latent_dim)
        sample = vae.decoder(z)
        sample = sample.view(-1, 28, 28)

        fig, axs = plt.subplots(1, 10, figsize=(15, 3))
        for i in range(10):
            axs[i].imshow(sample[i].cpu().numpy(), cmap="gray")
            axs[i].axis("off")
        plt.show()
