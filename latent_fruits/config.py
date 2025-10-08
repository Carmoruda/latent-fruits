from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Configuration for the project.

    Attributes:
        data_dir (Path): Directory for the dataset.
        output_dir (Path): Directory for output files.
        batch_size (int): Batch size for training.
        beta (float): Weight for the KL divergence term in the loss function.
        image_size (int): Size to which images are resized.
        seed (int): Random seed for reproducibility.
    """

    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    batch_size: int = 64
    beta: float = 1e-3
    image_size: int = 100
    seed: int = 42

    def with_updates(self, **overrides: Any) -> "ProjectConfig":
        """Return a new ProjectConfig with updated fields.

        Returns:
            ProjectConfig: A new instance of ProjectConfig with updated fields.
        """

        # Build a dict of overrides but ignore any keys explicitly set to None.
        valid_overrides: Dict[str, Any] = {}

        for key, value in overrides.items():
            # skip values that are None (treat them as "no override")
            if value is None:
                continue

            valid_overrides[key] = value

        # If no valid overrides, return self (this instance)
        if not valid_overrides:
            return self

        # If there are valid overrides, build a dict of converted values
        # for fields that need conversion
        converted: Dict[str, Any] = {}

        for key, value in valid_overrides.items():
            # If the field is a Path, convert the string to a Path
            if key in {"data_dir", "output_dir"} and value is not None:
                converted[key] = Path(value)
            else:
                converted[key] = value

        # Return a new instance with the updated fields
        return replace(self, **converted)

    def ensure_directories(self) -> None:
        """
        Ensure expected directories exist on disk
        (if they don't already, create them).
        """

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_config(
    config_path: Optional[Path] = None,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> ProjectConfig:
    """Load a ProjectConfig from YAML with optional runtime overrides.

    Args:
        config_path (Optional[Path], optional): Path to the config YAML file. Defaults to None.
        overrides (Optional[Dict[str, Any]], optional): Overrides for config values. Defaults to None.

    Raises:
        FileNotFoundError: If the config file is not found.
        ValueError: If the config file is invalid.

    Returns:
        ProjectConfig: The loaded and possibly overridden config.
    """

    # Build a dict of config data from the file and overrides
    config_data: Dict[str, Any] = {}

    # Load from the config file if provided
    if config_path:
        # Resolve the path and check it exists
        resolved = Path(config_path).expanduser()

        # If the file doesn't exist, raise an error
        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {resolved}")

        # Load the YAML file
        with resolved.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}

            # Ensure the loaded data is a dict
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file must define a mapping, got {type(loaded)!r}")

            # Update the config data with the loaded values
            config_data.update(loaded)

    # Apply any overrides
    if overrides:
        config_data.update({k: v for k, v in overrides.items() if v is not None})

    base = ProjectConfig()

    return base.with_updates(**config_data)
