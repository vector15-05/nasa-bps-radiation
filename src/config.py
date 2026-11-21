from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
IMG_DIR = DATA_DIR / "images"
META_PATH = DATA_DIR / "meta.csv"
SUBSET_META_PATH = DATA_DIR / "subset_meta.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "resnet18_nasa_bps.pth"

# NASA Public S3 Bucket
S3_BUCKET = "nasa-bps-training-data"
S3_PREFIX = "Microscopy/"
S3_META_KEY = "Microscopy/train/meta.csv" 

# Metadata column mapping (verify against your downloaded meta.csv)
COL_FILENAME = "filename"
COL_PARTICLE = "particle_type"
COL_GROUP = "experiment_id"  # Fallback: "replicate" or "hr_post_exposure"
COL_DOSE = "dose_Gy"

# Training hyperparameters
SAMPLES_PER_CLASS = 500
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4
SEED = 42