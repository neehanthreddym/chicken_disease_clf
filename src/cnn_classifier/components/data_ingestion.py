import os
import re
import shutil
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
        src = None
        for candidate in [unzip_path / "train", unzip_path / "Train"]:
            if candidate.exists() and candidate.is_dir():
                src = candidate
                break
        if src is None:
            return
        
        dst = unzip_path / "images"
        
        # If already renamed in a previous run, do nothing
        if dst.exists() and dst.is_dir():
            return
        
        # Avoid accidental overwrite
        if dst.exists():
            raise FileExistsError(f"Cannot rename to '{dst}': path exists.")
        
        src.rename(dst)
    
    def organize_images_into_class_folders(self) -> None:
        """
        Moves images from artifacts/.../data/images/*.jpg into class label folders.
        Label is inferred from filename prefix (before first dot).
        """
        self.images_dir = self.config.unzip_dir / "images"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"images folder not found at: {self.images_dir}")
        
        class_dirs = {
            "healthy": self.images_dir / "Healthy",
            "salmo": self.images_dir / "Salmonella",
            "cocci": self.images_dir / "Coccidiosis",
            "ncd": self.images_dir / "New Castle Disease",
        }
        
        # Ensure folders exist
        for cls in class_dirs.values():
            cls.mkdir(parents=True, exist_ok=True)
        
        def normalize_key(stem: str) -> str:
            s = stem.lower().strip()
            s = re.sub(r"[^a-z]", "", s)
            if s.startswith("pcr"):
                s = s[3:]
            return s
        
        moved, skipped, unknown = 0, 0, 0
        
        # Move images directly under images/
        for f in self.images_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            # filename like: pcrhealthy.123.jpg -> image filename prefix is "pcrhealthy"
            img_filename_prefix = f.name.split(".")[0]
            key = normalize_key(img_filename_prefix)

            target_dir = class_dirs.get(key)
            if target_dir is None:
                unknown += 1
                continue
            
            dest = target_dir / f.name
            if dest.exists():
                skipped += 1
                continue

            shutil.move(str(f), str(dest))
            moved += 1
        
        logger.info(f"Organize images: moved={moved}, skipped={skipped}, unknown={unknown}")
    
    def print_data_summary(self):
        img_exts = {".jpg", ".jpeg", ".png"}

        logger.info(f"Data Summary @ {self.images_dir}")
        logger.info("Classes: " + ", ".join(sorted([f.name for f in self.images_dir.iterdir() if f.is_dir()])))

        class_counts = {}
        total = 0

        for class_dir in sorted([f for f in self.images_dir.iterdir() if f.is_dir()]):
            n = sum(1 for f in class_dir.rglob("*") if f.suffix.lower() in img_exts)
            class_counts[class_dir.name] = n
            total += n
        
        class_summary = "\n".join([f"{k:20s} {v}" for k, v in class_counts.items()])
        logger.info(f"Data Summary\n{class_summary}\nTOTAL: {total}")