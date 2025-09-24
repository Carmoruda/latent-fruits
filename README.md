# Variational Autoencoder for Cat Images

This project trains a convolutional Variational Autoencoder (VAE) on cat images and augments its latent space with a Gaussian Mixture Model (GMM) to synthesize new samples. It includes utilities to split the dataset, monitor validation loss, and optionally run large-scale hyperparameter sweeps with Ray Tune.

## Features
- Convolutional VAE with configurable latent dimensionality.
- Automatic device selection (CUDA, Apple MPS, or CPU).
- Train/validation/test split with reconstruction visualizations saved to disk.
- Optional Gaussian Mixture Model fitting on the latent space for guided sampling.
- Ray Tune integration for hyperparameter exploration using an ASHA scheduler.

## Project Structure
- `main.py`: Entry point for training, evaluation, and Ray Tune integration.
- `vae/dataset.py`: Dataset helper that downloads (optional) and loads images from a Kaggle cat dataset.
- `vae/model.py`: VAE definition, training loop, latent-space GMM utilities, and plotting helpers.
- `data/`: Expected location of raw input images (ignored by Git).
- `output/`: Generated figures such as reconstructions and synthetic samples.

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
By default, `main.py` instantiates `VAEDataset` with `download=False`, so it expects images to already be present under `data/`.

1. **Manual download:** Obtain the [Kaggle Cats dataset](https://www.kaggle.com/borhanitrash/cat-dataset) manually and extract the images into the `data/` directory.
2. **Automatic download:** Set `download=True` when constructing `VAEDataset` (see `main.py`) and make sure your Kaggle API credentials are available as environment variables (`KAGGLE_USERNAME`, `KAGGLE_KEY`). The helper will download the archive, extract images, and clean up temporary files.

Ensure `data/` only contains image files (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`). The dataset loader raises a descriptive error if no images are found.

## Usage
### Train the VAE
```bash
python main.py
```
The script will:
- create train/validation/test splits (80/10/10),
- train for the configured number of epochs (`epochs` in `train_vae`),
- write `output/reconstructions.png` with original vs. reconstructed samples,
- fit a GMM on the latent space using the training split,
- and save `output/generated.png` with images sampled from the latent space (GMM-driven when available).

Key hyperparameters live in `main.py`:
- `lr`: Adam learning rate.
- `latent_dim`: Latent space dimensionality.
- `epochs`: Number of training epochs.
Adjust them directly or wire in a configuration loader if needed.

### Hyperparameter Tuning
Uncomment the `hyperparameter_tuning()` call in `main.py` to launch a Ray Tune sweep across learning rate, latent dimension, and epoch counts. Results are reported to the Ray dashboard (if running) and the best configuration is printed on completion. You can refine the search space or scheduler strategy in `hyperparameter_tuning()`.

### Generating Images
After training, `model.generate_images()` samples from the latent distribution (using the fitted GMM when available) and writes a grid of generated images to the `output/` directory. Modify the `num_images` argument to control how many samples are produced.

## Notes & Tips
- Training on CPU is feasible for quick experiments but significantly slower than using a GPU or Apple Silicon.
- The learning-rate scheduler (`ReduceLROnPlateau`) reacts to validation loss; expect longer runs to stabilize training quality.
- For reproducibility, consider seeding NumPy, PyTorch, and Ray before training or tuning.
- Monitor available disk space when running Ray Tune—each trial can maintain checkpoints and logs.

## Troubleshooting
- **"No images found" error:** Verify the `data/` directory contains images, not subfolders. Adjust the download flag or path as needed.
- **Kaggle download failures:** Confirm your Kaggle API credentials are set and that the dataset is accessible. Alternatively, download manually and set `download=False`.
- **Matplotlib backend issues in headless environments:** Configure a non-interactive backend (e.g., `matplotlib.use("Agg")`) before imports if needed.

Happy training!
