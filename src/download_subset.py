import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

import config

def get_s3_client():
    return boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED))

def download_file(s3, key, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not dest_path.exists():
        s3.download_file(config.S3_BUCKET, key, str(dest_path))

def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)
    s3 = get_s3_client()

    print(f"Downloading metadata from s3://{config.S3_BUCKET}/{config.S3_META_KEY}...")
    download_file(s3, config.S3_META_KEY, config.META_PATH)

    df = pd.read_csv(config.META_PATH)
    print("Columns found:", df.columns.tolist())

    # Keep only Fe and X-ray classes and sample evenly
    df_filtered = df[df[config.COL_PARTICLE].isin(["Fe", "X-ray"])].copy()
    subset_df = (
        df_filtered.groupby(config.COL_PARTICLE, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), config.SAMPLES_PER_CLASS), random_state=config.SEED))
        .reset_index(drop=True)
    )
    subset_df.to_csv(config.SUBSET_META_PATH, index=False)
    print(f"Saved subset metadata ({len(subset_df)} images) to {config.SUBSET_META_PATH}")

    # Parallel download of TIFF images
    tasks = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for fname in subset_df[config.COL_FILENAME]:
            s3_key = f"{config.S3_PREFIX}train/{fname}"
            dest = config.IMG_DIR / fname
            tasks.append(executor.submit(download_file, s3, s3_key, dest))

        for _ in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading TIFFs"):
            pass

    print("Download complete.")

if __name__ == "__main__":
    main()