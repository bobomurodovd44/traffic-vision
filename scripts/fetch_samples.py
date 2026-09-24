"""One-off helper: pull the 4 sample videos (links in Videos.pdf) into samples/.

Not part of the submission interface - just a convenience script for the team.
Run once, with internet: python scripts/fetch_samples.py
"""
from __future__ import annotations

import pathlib

import gdown

FILE_IDS = {
    "sample_001.mp4": "1kR9jODA2Wotw4gwkvpRKdqFADNJNc1nS",
    "sample_002.mp4": "1hp8DYeqtYHSwfM6qAo9FPSRHlpMFrIN_",
    "sample_003.mp4": "10cHEReCWzO3u-Vk1CnNgHAx6egGy5MwJ",
    "sample_004.mp4": "1aJ-QsAZVYJtLKHiRvKKeBq1D3GWNobRd",
}

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "samples"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for name, file_id in FILE_IDS.items():
        dest = OUT_DIR / name
        if dest.exists():
            print(f"skip {name} (already downloaded)")
            continue
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"fetching {name} ...")
        gdown.download(url, str(dest), quiet=False)


if __name__ == "__main__":
    main()
