from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader

from latent_fruits import load_config, seed_everything
from latent_fruits import training as training_utils
from vae import CVAE
from vae.model import DEVICE


def train_vae(config, report=True):
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_callback = train.report if report else None

    model, train_dataloader, _, test_dataloader = training_utils.run_training_pipeline(
        config,
        data_dir=data_dir,
        output_dir=output_dir,
        report_callback=report_callback,
        device=DEVICE,
    )

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
        training_utils.fit_gmm(
            model,
            gmm_dataloader,
            n_components=model.n_components,
        )

    return training_utils.generate_images(
        model,
        labels=list(labels),
        root_path=output_path.parent,
        file_name=output_path,
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
    config_dir = Path(__file__).parent / "configs/local.yaml"
    config = load_config(config_dir)

    # Adjust paths to be absolute
    data_dir = Path(__file__).parent / config.data_dir
    output_dir = Path(__file__).parent / config.output_dir
    config = config.with_updates(data_dir=data_dir, output_dir=output_dir)

    config.ensure_directories()
    seed_everything(config.seed)

    model, test_loader, train_loader = train_vae(
        {
            "lr": config.learning_rate,
            "latent_dim": config.latent_dim,
            "epochs": config.epochs,
            "n_classes": config.n_classes,
            "loss_function": F.mse_loss,
            "beta": config.beta,
            "batch_size": config.batch_size,
            "seed": config.seed,
        },
        False,
    )

    # hyperparameter_tuning()

    # Fit the GMM model
    training_utils.fit_gmm(model, train_loader, n_components=model.n_components)
    training_utils.generate_images(
        model,
        labels=[0, 1, 1, 0],
        root_path=data_dir,
        file_name=output_dir / "generated.png",
    )
    model_save_path = output_dir / "trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model Saved: {model_save_path}")
    load_model_and_generate(
        checkpoint_path=model_save_path,
        labels=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        output_path=output_dir / "generated_from_checkpoint.png",
        latent_dim=config.latent_dim,
        n_classes=config.n_classes,
        gmm_dataloader=train_loader,
    )
