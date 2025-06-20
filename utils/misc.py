import os
import logging
import shutil

def clear_directory(directory):
    """Creates a directory if it doesn't exist. If it does exist, ask for confirmation (with y or enter), before deleting all files and directories in it."""

    if os.path.exists(directory):
        logging.warning(
            f"Directory {directory} already exists. Previous data will be overwritten."
        )
        confirm = input(
            "Do you want to delete all files and directories in this directory? (y/n): "
        )
        if confirm.lower() in ['y', 'yes', '']:
            shutil.rmtree(directory)
            logging.debug(f"Directory {directory} cleared.")
        else:
            logging.debug(f"Keeping existing directory {directory}.")
    os.makedirs(directory, exist_ok=True)
