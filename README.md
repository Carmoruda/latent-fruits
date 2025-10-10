# Conditional VAE for Fruit Images

This project trains a convolutional Variational Autoencoder (CVAE) on fruit imagery (apples, bananas, and any additional classes you plug in), then augments its latent space with Gaussian Mixture Models (GMMs) to generate new samples. The codebase exposes a reusable training pipeline so you can embed the workflow in other scripts or run large-scale experiments with Ray Tune.

## Features
- Convolutional CVAE with configurable latent dimensionality and conditional labels.
- Automatic device selection (CUDA, Apple MPS, or CPU) plus reproducible seeding helpers.
- Balanced train/validation/test splits with reconstruction plots written to disk.
- Optional class-conditional Gaussian Mixture Models for guided sampling.
- Ray Tune integration via a lightweight pipeline function exposed in `latent_fruits.training`.
- YAML-driven configuration (`latent_fruits.config.ProjectConfig`) so hyperparameters stay versionable.

## Project Structure
- `main.py`: CLI entry point that loads a config, seeds everything, and invokes the training pipeline.
- `latent_fruits/config.py`: Dataclass definition + YAML loader for project-wide configuration.
- `latent_fruits/training.py`: End-to-end training utilities (data prep, epoch loop, plotting, GMM fitting).
- `latent_fruits/utils.py`: Shared helpers (e.g., deterministic seeding).
- `vae/model.py`: Pure CVAE module (network definition + checkpoint loader).
- `vae/dataset.py`: Dataset helper capable of downloading Kaggle archives (for example fruit datasets) and de-duplicating files.
- `configs/`: Example YAML configs (e.g., `configs/local.yaml`).
- `data/`: Staging area for images (Git-ignored).
- `output/`: Generated figures, checkpoints, and logs.

## Getting Started
### Prerequisites
- Python 3.10+ is recommended.
- A GPU is optional but supported (CUDA or Apple Silicon MPS).
- Kaggle credentials if you want the script to download the dataset for you.

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
The pinned versions cover PyTorch, TorchVision, Ray Tune, scikit-learn, matplotlib, Pillow, and their runtime dependencies. If you need a wheel tailored for CUDA or MPS, edit `requirements.txt` (or reinstall PyTorch) following the [official instructions](https://pytorch.org/get-started/locally/).

### Dataset Setup
By default, `main.py` expects images to already exist under `data/`. You can point each class label at any folder (e.g., `data/apple`, `data/banana`), swap in your own fruit photos, or adapt the paths to other domains.

1. **Manual download:** Obtain the fruit image sets you need (for example, Kaggle’s [Fresh and Rotten Fruits](https://www.kaggle.com/datasets/maysee/fresh-and-rotten-fruits) or any other collection) and extract the images into the appropriate subdirectories beneath `data/`.
2. **Automatic download:** Set `download: true` in your config (or pass it via CLI overrides) and make sure your Kaggle API credentials are available as environment variables (`KAGGLE_USERNAME`, `KAGGLE_KEY`). The dataset helper streams the zip to disk, extracts it, deduplicates files, and cleans up temporary artifacts.

Ensure `data/` only contains image files (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`). The dataset loader raises a descriptive error if no images are found.

## Usage
### Configuration Basics
Hyperparameters live in YAML under `configs/`. The default `configs/local.yaml` looks like:

```yaml
data_dir: data
output_dir: output
batch_size: 128
learning_rate: 1e-3
latent_dim: 128
beta: 1e-4
epochs: 20
n_classes: 2
seed: 2025
download: false
```

Every field maps to `ProjectConfig`, so you can add overrides directly in YAML or programmatically via `ProjectConfig.with_updates(...)`.

### Train the VAE from the CLI
```bash
python main.py
```
The script will:
- load `configs/local.yaml`, ensure the data/output directories exist, and seed RNGs,
- build balanced dataset splits (80/10/10),
- train for the configured number of epochs (`epochs` in the config),
- write `output/reconstructions.png` with original vs. reconstructed samples,
- fit a GMM on the latent space using the training split,
- and save `output/generated.png` with images sampled from the latent space (GMM-driven when available).

If you want to reuse the training pipeline from another script or a notebook:

```python
from pathlib import Path
from latent_fruits import training, load_config

config = load_config(Path("configs/local.yaml"))
model, train_loader, _, test_loader = training.run_training_pipeline(
    config.__dict__,
    data_dir=Path(config.data_dir),
    output_dir=Path(config.output_dir),
)
```

### Hyperparameter Tuning
Uncomment the `hyperparameter_tuning()` call in `main.py` (or invoke it manually) to launch a Ray Tune sweep across learning rate, latent dimension, loss functions, and epoch counts. Each trial calls the same pipeline under the hood, and metrics are reported via Ray’s callback. Feel free to extend `search_space` with additional parameters (e.g., `batch_size`, `beta`).

### Generating Images
After training, the GMM is fitted on demand and you can export new samples either by running `main.py` (see the bottom of the file) or programmatically via `latent_fruits.training.generate_images`. The helper saves a grid to disk and returns the tensor so you can embed it in notebooks.

## Notes & Tips
- Training on CPU is feasible for quick experiments but significantly slower than using a GPU or Apple Silicon.
- The learning-rate scheduler (`ReduceLROnPlateau`) reacts to validation loss; expect longer runs to stabilize training quality.
- For reproducibility, consider seeding NumPy, PyTorch, and Ray before training or tuning.
- Monitor available disk space when running Ray Tune—each trial can maintain checkpoints and logs.

## Troubleshooting
- **"No images found" error:** Verify the `data/` directory contains images, not subfolders. Adjust the download flag or path as needed.
- **Kaggle download failures:** Confirm your Kaggle API credentials are set and that the dataset is accessible. Alternatively, download manually and set `download=False`.
- **Matplotlib backend issues in headless environments:** Configure a non-interactive backend (e.g., `matplotlib.use("Agg")`) before imports if needed.

Happy training! Tag us if you generate something fun 🍎🍌
