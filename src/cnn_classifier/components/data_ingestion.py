import os
from urllib import request
import zipfile
from pathlib import Path
from cnn_classifier import logger
from cnn_classifier.utils.utilities import get_size
from cnn_classifier.entity.pipeline_config import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """Initialize DataIngestion with the given configuration."""
        self.config = config
    
    def download_data(self):
        """
        Download the data file from the source URL if it does not already exist locally.

        Returns:
            str: Path to the downloaded (or existing) local data file.

        Raises:
            Exception: If an error occurs during download.
        """
        try:
            if not os.path.exists(self.config.local_data_file):
                filename, headers = request.urlretrieve(
                    url=self.config.source_URL,
                    filename=self.config.local_data_file
                )
                logger.info(f"Downloaded {filename} with following info:\n{headers}")
            else:
                logger.info(f"Data already exists at {self.config.local_data_file} ({get_size(Path(self.config.local_data_file))}), skipping download")
            
            return self.config.local_data_file
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    
    def extract_zip_file(self):
        """Extract the contents of the downloaded zip file into the configured directory."""
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_file_ref:
            zip_file_ref.extractall(unzip_path)
        
        # Rename top-level folder: train/ or Train/ -> images/
        base = Path(unzip_path)
        src = None
        for candidate in [base / "train", base / "Train"]:
            if candidate.exists() and candidate.is_dir():
                src = candidate
                break
        if src is None:
            return
        
        dst = base / "images"
        
        # If already renamed in a previous run, do nothing
        if dst.exists() and dst.is_dir():
            return
        
        # Avoid accidental overwrite
        if dst.exists():
            raise FileExistsError(f"Cannot rename to '{dst}': path exists.")
        
        src.rename(dst)