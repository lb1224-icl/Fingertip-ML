import argparse
import sys
import zipfile
from pathlib import Path

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError as e:
    KaggleApi = None  # type: ignore
    _import_error = e
else:
    _import_error = None


def download_kaggle_dataset(dataset: str, dest_zip: Path) -> Path:
    if KaggleApi is None:
        raise ImportError(
            f"kaggle package not installed. Install with `pip install kaggle`. Original error: {_import_error}"
        )

    api = KaggleApi()
    api.authenticate()

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset} to {dest_zip} ...")
    api.dataset_download_files(dataset, path=str(dest_zip.parent), quiet=False, unzip=False, force=True)

    # Kaggle names the file <dataset_slug>.zip; rename to the desired destination.
    slug_name = Path(dataset.split("/")[-1]).name + ".zip"
    downloaded = dest_zip.parent / slug_name
    if downloaded != dest_zip:
        downloaded.replace(dest_zip)
    return dest_zip


def extract(zip_path: Path, out_dir: Path):
    print(f"Extracting {zip_path} to {out_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle dataset (YOLO format).")
    parser.add_argument("--dataset", type=str, required=True, help="Kaggle dataset slug, e.g., 'owner/dataset-name'.")
    parser.add_argument("--out", type=str, default="data/hand_keypoint_dataset_26k", help="Destination directory.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the downloaded ZIP file.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir.parent / "dataset.zip"

    try:
        download_kaggle_dataset(args.dataset, zip_path)
    except Exception as e:
        print(f"[ERROR] Failed to download: {e}")
        sys.exit(1)

    extract(zip_path, out_dir)

    if not args.keep_zip:
        try:
            zip_path.unlink()
        except OSError:
            pass

    print("Done. Verify the folder structure: images/{split}/ and labels/{split}/")


if __name__ == "__main__":
    main()
