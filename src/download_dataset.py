import argparse
import shutil
import sys
from pathlib import Path

try:
    import kagglehub
except ImportError as e:
    kagglehub = None
    _import_error = e
else:
    _import_error = None


def copy_tree(src: Path, dst: Path):
    # Copy the downloaded dataset cache into the working directory.
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        s = item
        d = dst / item.name
        if item.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def main():
    # Download dataset and copy into the working directory.
    parser = argparse.ArgumentParser(description="Download Kaggle dataset via kagglehub.")
    parser.add_argument("--dataset", type=str, required=True, help="Kaggle dataset slug, e.g., owner/dataset-name.")
    parser.add_argument("--out", type=str, default="data", help="Destination directory.")
    args = parser.parse_args()

    if kagglehub is None:
        print(f"[ERROR] kagglehub not installed: {_import_error}. Install with `pip install kagglehub`.")
        sys.exit(1)

    print(f"Downloading {args.dataset} via kagglehub...")
    cached_path = Path(kagglehub.dataset_download(args.dataset))
    print(f"Cached at: {cached_path}")

    dest = Path(args.out)
    print(f"Copying to {dest} ...")
    copy_tree(cached_path, dest)

    print("Done. Verify the folder structure: images/{split}/ and labels/{split}/")


if __name__ == "__main__":
    main()
