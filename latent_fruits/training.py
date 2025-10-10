from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, cast
from collections.abc import Sized

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import transforms

from vae import CVAE, CVAEDataset


def _get_device(model: torch.nn.Module, override: Optional[torch.device]) -> torch.device:
    if override is not None:
        return override
    try:
        return next(model.parameters()).device  # type: ignore[call-arg]
    except StopIteration:
        return torch.device("cpu")


def _balance_dataset(
    dataset: Dataset,
    target_length: int,
    *,
    seed: Optional[int],
) -> Dataset:
    dataset_size = len(cast(Sized, dataset))

    if dataset_size <= target_length:
        return dataset

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    indices = torch.randperm(dataset_size, generator=generator)[:target_length].tolist()
    return Subset(dataset, indices)


def prepare_dataloaders(
    data_dir: Path,
    *,
    batch_size: int,
    seed: Optional[int],
    download: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create balanced train/val/test dataloaders for the project datasets."""

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    apple_dataset = CVAEDataset(
        "",
        str(data_dir / "apple"),
        download=download,
        transform=transform,
        label=1,
        extracted_folder="data",
        delete_extracted=False,
    )

    banana_dataset = CVAEDataset(
        "",
        str(data_dir / "banana"),
        download=download,
        transform=transform,
        label=0,
        extracted_folder="data",
        delete_extracted=True,
    )

    balanced_length = min(len(banana_dataset), len(apple_dataset))
    banana_dataset = _balance_dataset(banana_dataset, balanced_length, seed=seed)
    apple_dataset = _balance_dataset(apple_dataset, balanced_length, seed=seed)

    def split_dataset(
        dataset: Dataset, *, generator: torch.Generator
    ) -> tuple[Dataset, Dataset, Dataset]:
        train_fraction = 0.8
        val_fraction = 0.1
        dataset_size = len(cast(Sized, dataset))
        train_size = int(train_fraction * dataset_size)
        val_size = int(val_fraction * dataset_size)
        test_size = dataset_size - train_size - val_size
        splits = random_split(dataset, [train_size, val_size, test_size], generator=generator)
        return tuple(splits)  # type: ignore[return-value]

    split_generator = torch.Generator()
    if seed is not None:
        split_generator.manual_seed(seed)

    banana_train, banana_val, banana_test = split_dataset(banana_dataset, generator=split_generator)
    apple_train, apple_val, apple_test = split_dataset(apple_dataset, generator=split_generator)

    train_dataset = ConcatDataset([banana_train, apple_train])
    val_dataset = ConcatDataset([banana_val, apple_val])
    test_dataset = ConcatDataset([banana_test, apple_test])

    loader_generator = torch.Generator()
    if seed is not None:
        loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def run_training_pipeline(
    config: Mapping[str, Any],
    *,
    data_dir: Path,
    output_dir: Path,
    report_callback: Optional[Callable[[dict[str, float]], None]] = None,
    device: Optional[torch.device] = None,
) -> tuple[CVAE, DataLoader, DataLoader, DataLoader]:
    """High-level helper that prepares data, trains the model, and logs metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(config)
    batch_size = int(cfg.get("batch_size", 64))
    seed = cfg.get("seed")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        data_dir,
        batch_size=batch_size,
        seed=seed,
        download=bool(cfg.get("download", False)),
    )

    model = CVAE(
        latent_dim=int(cfg["latent_dim"]),
        n_classes=int(cfg["n_classes"]),
        beta=float(cfg.get("beta", 1e-3)),
        n_components=int(cfg.get("n_components", 10)),
    )
    model_device = _get_device(model, device)
    model.to(model_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    loss_function = cfg.get("loss_function", F.mse_loss)
    epochs = int(cfg["epochs"])
    beta = float(cfg.get("beta", 1e-3))

    for current_epoch in range(epochs):
        _, val_history = train_model(
            model,
            loss_function,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            epoch=current_epoch,
            num_epochs=epochs,
            beta=beta,
            device=model_device,
        )

        if report_callback:
            metrics = {"loss": val_history[-1]}
            current_lr = getattr(model, "current_lr", None)
            if current_lr is not None:
                metrics["lr"] = current_lr
            report_callback(metrics)

    plot_reconstructions(
        model,
        val_loader,
        root_path=output_dir,
        file_name=output_dir / "reconstructions.png",
        device=model_device,
    )

    if getattr(model, "lr_history", None):
        plot_learning_rate(model, file_name=output_dir / "lr_schedule.png")

    return model, train_loader, val_loader, test_loader

def train_model(
    model: CVAE,
    loss_function: Optional[Callable[..., torch.Tensor]],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    *,
    epoch: int,
    num_epochs: int,
    beta: float,
    device: Optional[torch.device] = None,
) -> tuple[list[float], list[float]]:
    """Run one training epoch followed by validation.

    Returns lists containing a single element each to keep backward compatibility
    with the previous API (train/val loss histories).
    """

    model_device = _get_device(model, device)
    model.to(model_device)

    if loss_function is None:
        loss_function = F.mse_loss

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    if epoch == 0 and getattr(model, "lr_history", None):
        model.lr_history.clear()

    # --- Training Phase ---
    model.train()
    total_train_loss = 0.0
    total_train_samples = 0

    for images, labels in train_dataloader:
        images = images.to(model_device)
        labels = labels.to(model_device)

        y_onehot = F.one_hot(labels, num_classes=model.n_classes).float()

        reconstructed_images, mu, logvar = model(images, y_onehot)

        try:
            reconstruction_loss = loss_function(
                reconstructed_images, images, reduction="sum"
            )  # type: ignore[misc]
            reconstruction_loss = reconstruction_loss / images.size(0)
        except TypeError:
            reconstruction_loss = loss_function(reconstructed_images, images)
            if reconstruction_loss.ndim > 0:
                reconstruction_loss = reconstruction_loss.mean()

        kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_divergence_loss = kl_divergence_loss / images.size(0)

        loss = (reconstruction_loss / len(images)) + (beta * kl_divergence_loss) / len(images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_train_loss += loss.item() * batch_size
        total_train_samples += batch_size

    avg_train_loss = total_train_loss / max(total_train_samples, 1)
    train_loss_history.append(avg_train_loss)

    # --- Validation Phase ---
    model.eval()
    total_val_loss = 0.0
    total_val_samples = 0

    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(model_device)
            labels = labels.to(model_device)
            y_onehot = F.one_hot(labels, num_classes=model.n_classes).float()

            reconstructed_images, mu, logvar = model(images, y_onehot)

            try:
                reconstruction_loss = loss_function(
                    reconstructed_images, images, reduction="sum"
                )  # type: ignore[misc]
                reconstruction_loss = reconstruction_loss / images.size(0)
            except TypeError:
                reconstruction_loss = loss_function(reconstructed_images, images)
                if reconstruction_loss.ndim > 0:
                    reconstruction_loss = reconstruction_loss.mean()

            kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_divergence_loss = kl_divergence_loss / images.size(0)

            loss = (reconstruction_loss / len(images)) + (beta * kl_divergence_loss) / len(images)

            batch_size = images.size(0)
            total_val_loss += loss.item() * batch_size
            total_val_samples += batch_size

    avg_val_loss = total_val_loss / max(total_val_samples, 1)
    val_loss_history.append(avg_val_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}"
    )

    scheduler.step(avg_val_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    if hasattr(model, "lr_history"):
        model.lr_history.append(current_lr)
    if hasattr(model, "current_lr"):
        model.current_lr = current_lr

    return train_loss_history, val_loss_history


def fit_gmm(
    model: CVAE,
    train_dataloader: DataLoader,
    *,
    n_components: int,
    device: Optional[torch.device] = None,
) -> dict[int, GaussianMixture]:
    """Fit a Gaussian Mixture Model to the latent space of the training data."""

    model_device = _get_device(model, device)
    model.eval()

    latent_by_label: dict[int, list[np.ndarray]] = {}

    with torch.no_grad():
        for images, labels in train_dataloader:
            images = images.to(model_device)
            labels = labels.to(model_device)
            y_onehot = F.one_hot(labels, num_classes=model.n_classes).float()

            _, mu, _ = model(images, y_onehot)

            mu_np = mu.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for class_idx in range(model.n_classes):
                class_mask = labels_np == class_idx
                if np.any(class_mask):
                    latent_by_label.setdefault(class_idx, []).append(mu_np[class_mask])

    fitted_gmms: dict[int, GaussianMixture] = {}

    if not latent_by_label:
        return fitted_gmms

    for class_idx in range(model.n_classes):
        class_latents = latent_by_label.get(class_idx)
        if not class_latents:
            continue

        samples = np.concatenate(class_latents, axis=0)
        class_gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            reg_covar=1e-6,
        )

        class_gmm.fit(samples)
        fitted_gmms[class_idx] = class_gmm

    model.gmm = fitted_gmms
    return fitted_gmms


def generate_images(
    model: CVAE,
    *,
    labels: Sequence[int],
    root_path: Path,
    file_name: Path,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate new images by sampling from the latent space."""

    model_device = _get_device(model, device)
    model.eval()

    root_path.mkdir(parents=True, exist_ok=True)
    file_name.parent.mkdir(parents=True, exist_ok=True)

    labels_tensor = torch.tensor(labels, device=model_device)
    y_onehot = F.one_hot(labels_tensor, num_classes=model.n_classes).float()

    num_images = len(labels)

    with torch.no_grad():
        if getattr(model, "gmm", None):
            samples = []
            for label in labels:
                class_gmm = model.gmm.get(label) if hasattr(model, "gmm") else None
                if class_gmm is None:
                    samples.append(torch.randn(1, model.latent_dim, device=model_device))
                    continue

                sample, _ = class_gmm.sample(1)
                tensor_sample = torch.from_numpy(sample).to(
                    device=model_device, dtype=torch.float32
                )
                samples.append(tensor_sample)

            z = torch.cat(samples, dim=0).to(model_device)
        else:
            z = torch.randn(num_images, model.latent_dim, device=model_device)

        z_combined = torch.cat([z, y_onehot], dim=1)
        z_expanded = model.decoder_input(z_combined)
        generated_images = model.decoder(z_expanded)

    _plot_generated_images(generated_images, root_path, file_name)
    return generated_images


def plot_reconstructions(
    model: CVAE,
    dataloader: DataLoader,
    *,
    root_path: Path,
    num_images: int = 8,
    file_name: Path,
    device: Optional[torch.device] = None,
) -> None:
    """Plot original and reconstructed images."""

    model_device = _get_device(model, device)
    model.eval()

    root_path.mkdir(parents=True, exist_ok=True)
    file_name.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        images, labels = next(iter(dataloader))
        images = images.to(model_device)[:num_images]
        labels = labels.to(model_device)[:num_images]

        y_onehot = F.one_hot(labels, num_classes=model.n_classes).float()
        reconstructed_images, _, _ = model(images, y_onehot)

    _plot_reconstructions(images, reconstructed_images, root_path, file_name)


def plot_learning_rate(model: CVAE, *, file_name: Path) -> None:
    """Plot and save the learning rate schedule observed during training."""

    lr_history = getattr(model, "lr_history", None)
    if not lr_history:
        return

    file_name.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(lr_history) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, lr_history, marker="o")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def _plot_generated_images(
    generated_images: torch.Tensor,
    root_path: Path,
    file_name: Path,
) -> None:
    num_images = generated_images.size(0)

    _, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(generated_images[i].cpu().squeeze(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"Generated {i + 1}")

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def _plot_reconstructions(
    images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    root_path: Path,
    file_name: Path,
) -> None:
    num_images = images.size(0)

    _, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

    for i in range(num_images):
        ax = axes[0, i]
        ax.imshow(images[i].cpu().squeeze(), cmap="gray")
        ax.axis("off")
        ax.set_title("Original")

        ax = axes[1, i]
        ax.imshow(reconstructed_images[i].cpu().squeeze(), cmap="gray")
        ax.axis("off")
        ax.set_title("Reconstructed")

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
