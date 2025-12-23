import os
import random
import numpy as np
from pathlib import Path
from typing import Any
import yaml
import json
import joblib
from box.exceptions import BoxValueError
from box import ConfigBox
from ensure import ensure_annotations
import base64
import tensorflow as tf
from cnn_classifier import logger

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file from the given path and returns its content as a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the YAML file that needs to be read.

    Returns:
        ConfigBox: A ConfigBox containing the YAML content as attributes.
    
    Raises:
        ValueError: If the YAML file is empty or cannot be parsed.
        Exception: If any unexpected error occurs during the reading of the file.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file '{yaml_file} loaded successfully!'")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_dirs: list, verbose=True):
    """
    Create list of directories

    Args:
        path_to_dirs (list): list of paths of directories
        verbose (bool, optional): Ignore if multiple directories is to be created. Default is True.
    """
    for path in path_to_dirs:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save data in json format

    Args:
        path (Path): path to save the json file
        data (dict): data to be saved as json
    """
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    logger.info(f"json file saved at {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load JSON file and return data as ConfigBox.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Data from JSON file as class attributes.
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded succesfully from {path}")

    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Save a Python object to a binary file.

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load a Python object from a binary file.

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get file size in kilobytes (KB).

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

def encodeImageIntoBase64(croppedImagePath):
    """Encode an image file into a Base64 string."""
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

def decodeImage(imgstring, fileName):
    """Decode a Base64 string and save it as an image file."""
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

def configure_tf_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.info("No GPU found. Running on CPU.")
        return

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    logger.info(f"Enabled memory growth for {len(gpus)} GPU(s).")

def set_global_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.
    Call this at the START of any stage that uses randomness.
    """    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    logger.info(f"Global random seed set to {seed}")