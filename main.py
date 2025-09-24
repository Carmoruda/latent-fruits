import torch
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from vae import VAE, VAEDataset
from vae.model import BATCH_SIZE, DEVICE


def train_vae(config):
    # Create the dataset
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
    dataset = VAEDataset(
        "https://www.kaggle.com/api/v1/datasets/download/borhanitrash/cat-dataset",
        "data",
        download=False,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, Optimizer
    model = VAE(latent_dim=config["latent_dim"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5)

    for epoch in range(config["epochs"]):
        loss = model.train_model(
            dataloader, dataset, optimizer, scheduler, num_epochs=1
        )  # Train for 1 epoch

        # Report metrics to Ray Tune
        train.report({"loss": loss[-1]})


if __name__ == "__main__":
    # Define the search space for hyperparameters
    search_space = {
        "lr": tune.grid_search([1e-3, 1e-4, 1e-5]),
        "latent_dim": tune.grid_search([64, 128, 256]),
        "epochs": tune.grid_search([10, 50, 100]),
    }

    # Scheduler to stop bad trials early
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=1,
        reduction_factor=2,
    )

    # Set up the Tuner
    tuner = tune.Tuner(
        train_vae,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=1,  # Will run 27 trials due to grid search
        ),
        param_space=search_space,
    )

    # Run the hyperparameter tuning
    results = tuner.fit()

    # Get the best result
    best_result = results.get_best_result(metric="loss", mode="min")

    print("\n" * 2)
    print("-" * 50)
    print(" " * 15 + "Best Hyperparameters")
    print("-" * 50)
    print(f" * Best trial config: {best_result.config}")
    if best_result.metrics is not None and "loss" in best_result.metrics:
        print(f" * Best trial final validation loss: {best_result.metrics['loss']}")
    else:
        print(" * Best trial final validation loss: Not available")

    # You can now retrain the model with the best hyperparameters or load the best checkpoint
    # For example:
    # best_model = VAE(latent_dim=best_result.config["latent_dim"])
    # best_checkpoint = torch.load(best_result.checkpoint.to_air_checkpoint().path + "/checkpoint.pt")
    # best_model.load_state_dict(best_checkpoint["model_state"])
    print("-" * 50)
    print("\n")
