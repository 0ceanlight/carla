import os
import logging
import shutil
import re
import numpy as np
from pathlib import Path


def clear_directory(directory, noconfirm=False):
    """Creates a directory if it doesn't exist. If it does exist, ask for confirmation (with y or enter), before deleting all files and directories in it."""

    if os.path.exists(directory):
        logging.warning(
            f"Directory {directory} already exists. Previous data will be overwritten."
        )
        prompt = "Do you want to delete all files and directories in this directory? (y/n): "
        if noconfirm or input(prompt).lower() in ['y', 'yes', '']:
            shutil.rmtree(directory)
            logging.debug(f"Directory {directory} cleared.")
        else:
            logging.debug(f"Keeping existing directory {directory}.")
    os.makedirs(directory, exist_ok=True)


def read_jsonc_file(path: str) -> str:
    """
    Reads the contents of a JSONC file.

    Args:
        path (str): Path to the JSONC file.

    Returns:
        str: Raw content of the file.
    """
    return Path(path).read_text(encoding="utf-8")


def strip_jsonc_comments(text: str) -> str:
    """
    Removes both inline (//) and block (/* */) comments from a JSONC string.

    Args:
        text (str): JSONC content as a string.

    Returns:
        str: Cleaned JSON string without comments.
    """
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    return text


def simulate_gps_drift(poses, bias_std=0.15, noise_std=1.2, jump_prob=0.03, jump_std=3.5, seed=None):
    """
    Apply realistic GPS drift to ground truth poses.

    Args:
        poses: np.ndarray of shape (N, 7) where each row is (x, y, z, qx, qy, qz, qw)
        bias_std: Standard deviation for the random walk bias
        noise_std: Standard deviation for the Gaussian noise
        jump_prob: Probability of a jump occurring at each step
        jump_std: Standard deviation for the jump noise

    Returns:
        np.ndarray of shape (N, 7) with drifted poses
    """
    if seed is not None:
        np.random.seed(seed)
    drifted_poses = poses.copy()
    n = poses.shape[0]
    bias = np.zeros(3)
    for i in range(n):
        # Random walk for bias
        bias += np.random.normal(0, bias_std, 3)
        # Occasional jump
        if np.random.rand() < jump_prob:
            bias += np.random.normal(0, jump_std, 3)
        # Gaussian noise
        noise = np.random.normal(0, noise_std, 3)
        drifted_poses[i, :3] += bias + noise
    return drifted_poses
