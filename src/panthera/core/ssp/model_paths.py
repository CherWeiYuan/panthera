"""
Model paths.

This module finds the path to the model frozen graphs (.pb files).
"""

from importlib import resources
from pathlib import Path
import os

# Define the package where models live
MODEL_PACKAGE = "panthera.models"


def get_model_path(model_name: str) -> Path:
    """
    Resolves the absolute path to a model file stored within the package.
    Allows override via environment variable for production flexibility.
    """
    # 1. Check for Environment Variable Override
    # This allows a DevOps engineer to point to a high-speed SSD or
    # shared volume without touching your code.
    env_path = os.getenv(f"PANTHERA_{model_name.upper()}_PATH")
    if env_path:
        return Path(env_path)

    # 2. Use importlib.resources to find the file inside the installed package
    # This works even if the code is running from a .zip or site-packages
    try:
        p = resources.files(MODEL_PACKAGE).joinpath(f"{model_name}.pb")
        return Path(str(p))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model {model_name}.pb not found in {MODEL_PACKAGE}. "
            "Ensure the model file exists and is included in your distribution."
        )


# Constants for easy access
SPLICEAI_MODEL_PATH = get_model_path("spliceai")
MODELP_MODEL_PATH = get_model_path("modelp")
