"""
Data Downloader Module for Sentinal AI.

Downloads the official benchmark dataset:
Source: airt-ml/twitter-human-bots (Hugging Face / Cresci et al. 2017)
37,438 labelled Twitter accounts (Human vs. Bot).
"""

import os
import sys
import urllib.request
from pathlib import Path


DATASET_URL = (
    "https://huggingface.co/datasets/airt-ml/twitter-human-bots"
    "/resolve/main/twitter_human_bots_dataset.csv"
)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
TARGET_FILE = RAW_DATA_DIR / "twitter_human_bots_dataset.csv"


def download_benchmark_data(force: bool = False) -> Path:
    """
    Downloads the benchmark dataset if not already present.
    
    Args:
        force (bool): If True, re-downloads even if the file exists.
        
    Returns:
        Path: Path to the downloaded CSV file.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_FILE.exists() and not force:
        file_size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
        print(f"[OK] Dataset already exists: {TARGET_FILE} ({file_size_mb:.2f} MB)")
        return TARGET_FILE

    print(f"[*] Downloading benchmark dataset from {DATASET_URL}...")
    
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r--> Progress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Sentinal-AI-ML-Benchmark)")]
    urllib.request.install_opener(opener)

    urllib.request.urlretrieve(DATASET_URL, TARGET_FILE, reporthook=reporthook)
    print("\n[OK] Download completed successfully!")
    print(f"[OK] Saved to: {TARGET_FILE} ({TARGET_FILE.stat().st_size / (1024 * 1024):.2f} MB)")

    return TARGET_FILE


if __name__ == "__main__":
    download_benchmark_data()
