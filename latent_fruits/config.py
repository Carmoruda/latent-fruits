from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Configuration for the project.

    Attributes:
        data_dir (Path): Directory for the dataset.
        output_dir (Path): Directory for output files.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        latent_dim (int): Dimensionality of the latent space.s
        beta (float): Weight for the KL divergence term in the loss function.
        epochs (int): Number of training epochs.
        image_size (int): Size to which images are resized.
        n_classes (int): Number of conditional classes.
        seed (int): Random seed for reproducibility.
    """

    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    batch_size: int = 64
    learning_rate: float = 1e-3
    latent_dim: int = 128
    beta: float = 1e-3
    epochs: int = 20
    image_size: int = 100
    n_classes: int = 2
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
        field_types: Dict[str, Any] = get_type_hints(ProjectConfig)

        for key, value in valid_overrides.items():
            target_type = field_types.get(key)
            converted[key] = self._coerce_value(key, value, target_type)

        # Return a new instance with the updated fields
        return replace(self, **converted)

    @staticmethod
    def _coerce_value(field_name: str, value: Any, target_type: Any) -> Any:
        """Convert the override value to match the field type if needed."""

        if target_type is None or target_type is Any:
            return value

        origin = get_origin(target_type)

        if origin is Union:
            valid_args = [arg for arg in get_args(target_type) if arg is not type(None)]
            if len(valid_args) == 1:
                return ProjectConfig._coerce_value(field_name, value, valid_args[0])

        try:
            if target_type is Path:
                return value if isinstance(value, Path) else Path(value)

            if target_type is int:
                return value if isinstance(value, int) else int(value)

            if target_type is float:
                return value if isinstance(value, float) else float(value)

            if target_type is bool:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered in {"true", "1", "yes", "y", "on"}:
                        return True
                    if lowered in {"false", "0", "no", "n", "off"}:
                        return False
                return bool(value)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid value for '{field_name}': expected {target_type}, got {value!r}"
            ) from exc

        # If none of the above conversions applied, return the value as-is
        return value

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
