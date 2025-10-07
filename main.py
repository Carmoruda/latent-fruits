from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import transforms

from vae import CVAE, CVAEDataset
from vae.model import BATCH_SIZE, DEVICE


def train_vae(config, report=True):
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the dataset
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    anime_dataset = CVAEDataset(
        "https://www.kaggle.com/api/v1/datasets/download/soumikrakshit/anime-faces",
        str(data_dir) + "/apple",
        download=False,
        transform=transform,
        label=1,
        extracted_folder="data",
        delete_extracted=False,
    )

    cat_dataset = CVAEDataset(
        "https://www.kaggle.com/api/v1/datasets/download/borhanitrash/cat-dataset",
        str(data_dir) + "/banana",
        download=False,
        transform=transform,
        label=0,
        extracted_folder="cats",
        delete_extracted=True,
    )

    # Downsample each domain so both have the same number of images
    def balance_dataset(dataset: CVAEDataset, target_length: int) -> Dataset:
        if len(dataset) <= target_length:
            return dataset

        generator = torch.Generator()
        indices = torch.randperm(len(dataset), generator=generator)[:target_length].tolist()
        return Subset(dataset, indices)

    balanced_length = min(len(cat_dataset), len(anime_dataset))
    cat_dataset = balance_dataset(cat_dataset, balanced_length)
    anime_dataset = balance_dataset(anime_dataset, balanced_length)

    def split_dataset(dataset, train_fraction=0.8, val_fraction=0.1):
        train_size = int(train_fraction * len(dataset))
        val_size = int(val_fraction * len(dataset))
        test_size = len(dataset) - train_size - val_size
        return random_split(dataset, [train_size, val_size, test_size])

    cat_train, cat_val, cat_test = split_dataset(cat_dataset)
    anime_train, anime_val, anime_test = split_dataset(anime_dataset)

    train_dataset = ConcatDataset([cat_train, anime_train])
    val_dataset = ConcatDataset([cat_val, anime_val])
    test_dataset = ConcatDataset([cat_test, anime_test])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Optimizer
    model = CVAE(latent_dim=config["latent_dim"], n_classes=config["n_classes"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    loss_function = config.get("loss_function", F.mse_loss)

    for current_epoch in range(config["epochs"]):
        _, val_loss = model.train_model(
            loss_function,
            train_dataloader,
            val_dataloader,
            optimizer,
            scheduler,
            epoch=current_epoch,
            num_epochs=config["epochs"],
        )  # Train for 1 epoch

        # Report metrics to Ray Tune
        if report:
            metrics = {"loss": val_loss[-1]}
            if model.current_lr is not None:
                metrics["lr"] = model.current_lr
            train.report(metrics)

    model.plot_reconstructions(
        dataloader=DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
        root_path=str(data_dir),
        file_name=f"{output_dir}/reconstructions.png",
    )

    if model.lr_history:
        model.plot_learning_rate(file_name=f"{output_dir}/lr_schedule.png")

    return model, test_dataloader, train_dataloader


def load_model_and_generate(
    checkpoint_path: Union[str, Path],
    labels: Sequence[int],
    output_path: Union[str, Path],
    *,
    latent_dim: int,
    n_classes: int,
    n_components: int = 10,
    gmm_dataloader: Optional[DataLoader] = None,
    map_location: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """Load a trained CVAE checkpoint and produce new samples.

    Args:
        checkpoint_path: Path to the ``.pth`` file with the model weights.
        labels: Sequence of conditional labels to control each generated image.
        output_path: Destination PNG path for the generated grid.
        latent_dim: Latent dimensionality used when the model was trained.
        n_classes: Number of conditional classes used during training.
        n_components: Number of Gaussian components per class when fitting the GMM.
        gmm_dataloader: Optional dataloader; if provided the latent GMM is refitted before sampling.
        map_location: Optional device override for ``torch.load``.

    Returns:
        torch.Tensor: Tensor containing the generated images.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = CVAE.load_from_checkpoint(
        checkpoint_path,
        latent_dim=latent_dim,
        n_classes=n_classes,
        n_components=n_components,
        map_location=map_location,
    )

    if gmm_dataloader is not None:
        model.fit_gmm(gmm_dataloader)

    return model.generate_images(
        root_path=str(output_path.parent),
        labels=list(labels),
        file_name=str(output_path),
    )


def hyperparameter_tuning():
    # Define the search space for hyperparameters
    search_space = {
        "lr": tune.grid_search([1e-3, 1e-4, 1e-5]),
        "latent_dim": tune.grid_search([32, 64, 128]),
        "epochs": tune.grid_search([10, 20]),
        "n_classes": 2,
        "loss_function": tune.grid_search([F.mse_loss, F.cross_entropy]),
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
            "epochs": 20,
            "n_classes": 2,
            "loss_function": F.mse_loss,
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
    load_model_and_generate(
        checkpoint_path=model_save_path,
        labels=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        output_path=output_dir / "generated_from_checkpoint.png",
        latent_dim=128,
        n_classes=2,
        gmm_dataloader=train_dataloader,
    )
