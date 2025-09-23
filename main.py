import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from vae import VAE, VAEDataset
from vae.model import BATCH_SIZE, DEVICE

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
    dataset = VAEDataset(
        "https://www.kaggle.com/api/v1/datasets/download/borhanitrash/cat-dataset",
        "data",
        download=False,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    epochs_to_test = [10]
    latent_dims_to_test = [128]
    learning_rates_to_test = [1e-3]

    # Dictionary for storing loss histories
    loss_history = {}

    for epoch in epochs_to_test:
        for latent_dim in latent_dims_to_test:
            for learning_rate in learning_rates_to_test:
                # Create a unique file_name for this run's results
                file_name = f"output/epochs_{epoch}_latent_{latent_dim}_lr_{learning_rate}.png"

                # Initialize the VAE model
                model = VAE(latent_dim=latent_dim).to(DEVICE)

                # Define the optimizer (Adam for better convergence)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                print("\n")
                print("-" * 50)
                print(" " * 15 + "Starting Training")
                print(f" * Training on device: {DEVICE}")
                print(f" * Number of samples in dataset: {len(dataset)}")
                print(f" * Batch size: {BATCH_SIZE}")
                print(f" * Number of epochs: {epoch}")
                print(f" * Learning rate: {learning_rate}")
                print(f" * Latent dimension: {latent_dim}")
                print("-" * 50)
                print("\n")

                # Training loop
                loss = model.train_model(dataloader, dataset, optimizer, num_epochs=epoch)

                # Save loss history to a dictonary
                key = f"Epochs: {epoch}, Latent_dim: {latent_dim}, LR: {learning_rate}"
                loss_history[key] = loss

                # Plot some reconstructions
                model.plot_reconstructions(
                    dataloader, num_images=8, root_path="output", file_name=file_name
                )
