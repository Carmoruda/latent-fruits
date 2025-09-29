from pathlib import Path

import torch
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import transforms

from vae import CVAE, CVAEDataset
from vae.model import BATCH_SIZE, DEVICE


def train_vae(config, report=True):
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "output"

    # Create the dataset
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    anime_dataset = CVAEDataset(
        "https://www.kaggle.com/api/v1/datasets/download/soumikrakshit/anime-faces",
        str(data_dir) + "/anime",
        download=False,
        transform=transform,
        label=1,
        extracted_folder="data",
        delete_extracted=False,
    )

    cat_dataset = CVAEDataset(
        "https://www.kaggle.com/api/v1/datasets/download/borhanitrash/cat-dataset",
        str(data_dir) + "/cats",
        download=False,
        transform=transform,
        label=0,
        extracted_folder="cats",
        delete_extracted=True,
    )

    dataset = ConcatDataset([cat_dataset, anime_dataset])

    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = int((len(dataset) - train_size) / 2)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Optimizer
    model = CVAE(latent_dim=config["latent_dim"], n_classes=config["n_classes"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5)

    for current_epoch in range(config["epochs"]):
        _, val_loss = model.train_model(
            train_dataloader,
            val_dataloader,
            optimizer,
            scheduler,
            epoch=current_epoch,
            num_epochs=config["epochs"],
        )  # Train for 1 epoch

        # Report metrics to Ray Tune
        if report:
            train.report({"loss": val_loss[-1]})

    model.plot_reconstructions(
        dataloader=DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
        root_path=str(data_dir),
        file_name=f"{output_dir}/reconstructions.png",
    )

    return model, test_dataloader, train_dataloader


def hyperparameter_tuning():
    # Define the search space for hyperparameters
    search_space = {
        "lr": tune.grid_search([1e-3, 1e-4, 1e-5]),
        "latent_dim": tune.grid_search([64, 128, 256]),
        "epochs": tune.grid_search([10, 50, 100]),
        "n_classes": 3,
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


if __name__ == "__main__":
    # hyperparameter_tuning()
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "output"

    model, test_dataloader, train_dataloader = train_vae(
        {
            "lr": 1e-3,
            "latent_dim": 128,
            "epochs": 10,
            "n_classes": 2,
        },
        False,
    )

    # Fit the GMM model
    model.fit_gmm(train_dataloader)

    model.generate_images(
        root_path=str(data_dir), labels=[0, 1, 1, 0], file_name=f"{output_dir}/generated.png"
    )

    model_save_path = output_dir / "trained_model.pth"

    torch.save(model.state_dict(), model_save_path)
    print(f"Model Saved: {model_save_path}")
