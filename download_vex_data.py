#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "requests>=2.25.0",
#     "zstandard>=0.17.0",
#     "tqdm>=4.61.0"
# ]
# ///
"""
Script to download the latest VEX dataset from Red Hat Security Data.
This script:
1. If no local data exists or data is older than 3 weeks:
   a. Fetches the latest archive filename from the archive_latest.txt file
   b. Downloads the archive file
   c. Extracts the contents to a local data directory
2. Otherwise:
   a. Checks changes.csv for individual file updates
   b. Downloads any individual files that have changed since the last sync
   c. Updates only the files that have been modified since the last sync
The script maintains a metadata.json file to track the last sync time.
"""

import csv
import json
import shutil
import sys
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import requests
import zstandard as zstd
from tqdm import tqdm

BASE_URL = "https://security.access.redhat.com/data/csaf/v2/vex"
ARCHIVE_LIST_URL = f"{BASE_URL}/archive_latest.txt"
CHANGES_CSV_URL = f"{BASE_URL}/changes.csv"
DATA_DIR = Path(__file__).parent / "data"
ARCHIVE_DIR = DATA_DIR / "archives"
EXTRACTED_DIR = DATA_DIR / "extracted"


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [DATA_DIR, ARCHIVE_DIR, EXTRACTED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_latest_archive_filename():
    try:
        response = requests.get(ARCHIVE_LIST_URL)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching latest archive information: {e}")
        sys.exit(1)


def get_changes_csv():
    print(f"Downloading changes.csv from {CHANGES_CSV_URL}")
    try:
        response = requests.get(CHANGES_CSV_URL)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading changes.csv: {e}")
        sys.exit(1)


def download_archive(filename):
    archive_url = f"{BASE_URL}/{filename}"
    local_path = ARCHIVE_DIR / filename

    if local_path.exists():
        print(f"Archive already exists at {local_path}")
        return local_path

    print(f"Downloading {archive_url} to {local_path}")
    try:
        response = requests.get(archive_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(local_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print("Download complete!")
        return local_path
    except requests.exceptions.RequestException as exc:
        print(f"Error downloading archive: {exc}")
        if local_path.exists():
            local_path.unlink()  # Delete partially downloaded file
        sys.exit(1)

def extract_archive(archive_path):
    if EXTRACTED_DIR.exists() and any(EXTRACTED_DIR.iterdir()):
        print(
            f"Extraction directory {EXTRACTED_DIR} already exists and contains files. Skipping extraction."
        )
        return

    print(f"Extracting {archive_path}")
    temp_tar = DATA_DIR / "temp.tar"
    with open(archive_path, "rb") as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with open(temp_tar, "wb") as tar_file:
            dctx.copy_stream(compressed_file, tar_file)

    file_count = 0
    with tarfile.open(temp_tar) as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Unpacking files"):
            tar.extract(member, path=EXTRACTED_DIR, filter="data")
            if member.isfile():
                file_count += 1

    temp_tar.unlink()
    print(f"Extraction complete to {EXTRACTED_DIR}: {file_count} files extracted")

def load_sync_metadata():
    """Read the metadata file if it exists."""
    metadata_path = DATA_DIR / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


def parse_changes_csv(csv_content, last_sync_dt: datetime):
    if not csv_content:
        print("No CSV content found in downloaded changes.csv")
        return []

    changed_files = []
    print(f"Only fetching files modified after {last_sync_dt}")

    csv_reader = csv.reader(csv_content.splitlines())

    for row in csv_reader:
        # Remove quotes if present in both fields
        file_path = row[0].strip('"')
        timestamp_str = row[1].strip('"')

        file_changed_dt = datetime.fromisoformat(timestamp_str)
        # Skip files that haven't changed since our last sync
        if last_sync_dt and file_changed_dt <= last_sync_dt:
            continue

        changed_files.append(file_path)

    print(f"Found {len(changed_files)} files that need to be updated")
    return changed_files


def download_individual_files(file_paths):
    success_count = 0
    download_errors = []

    for file_path in tqdm(file_paths, desc="Downloading files"):
        if not file_path:
            # Skip any empty lines
            continue

        file_url = urljoin(f"{BASE_URL}/", file_path)
        local_path = EXTRACTED_DIR / file_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(file_url)
            response.raise_for_status()

            try:
                response.json()
            except json.JSONDecodeError:
                msg = f"Received invalid JSON for {file_url}"
                download_errors.append(msg)
                continue

            with open(local_path, "wb") as f:
                f.write(response.content)
            success_count += 1

        except requests.exceptions.RequestException as e:
            msg = f"Error downloading {file_url}: {e}"
            download_errors.append(msg)
            # If the file exists but is incomplete due to error, remove it
            if local_path.exists() and local_path.stat().st_size == 0:
                local_path.unlink()

    print(
        f"Finished downloading individual files: "
        f"{success_count} successful, {len(download_errors)} failed"
    )
    if download_errors:
        print(f"Failed file downloads:\n{'\n'.join(download_errors)}")


def clear_data_dir():
    """Clear the entire data directory and recreate directory structure."""
    if DATA_DIR.exists():
        print(f"Clearing {DATA_DIR} for fresh data")
        shutil.rmtree(DATA_DIR)


def main():
    now = datetime.now(tz=UTC)
    print(f"Starting VEX data set download at {now}")

    metadata = load_sync_metadata()
    need_full_download = False

    if not metadata:
        print("No metadata found, performing full download")
        need_full_download = True

    elif not EXTRACTED_DIR.exists() or not any(EXTRACTED_DIR.iterdir()):
        print("No extracted data found, performing full download")
        need_full_download = True

    else:
        last_sync_at = metadata.get("last_sync")
        last_sync_at = datetime.fromisoformat(last_sync_at)
        three_weeks_ago = now - timedelta(weeks=3)
        age_days = (now - last_sync_at).days
        if last_sync_at < three_weeks_ago:
            print(
                f"Data is {age_days} days old (from {last_sync_at.date()}), performing full download"
            )
            need_full_download = True
        else:
            print(
                f"Data is {age_days} days old (from {last_sync_at.date()}), syncing individual files only"
            )

    if need_full_download:
        clear_data_dir()
        ensure_directories()
        filename = get_latest_archive_filename()
        print(f"Latest archive: {filename}")

        archive_path = download_archive(filename)
        extract_archive(archive_path)
        metadata = {
            "last_sync": str(now),
        }
        print(f"Download complete at {datetime.now(tz=UTC)}")
    else:
        last_sync_at = metadata.get("last_sync")
        last_sync_at = datetime.fromisoformat(last_sync_at)
        print(f"Last changes sync: {last_sync_at}")

        print("Downloading changes.csv to identify recent updates...")
        changes_csv = get_changes_csv()
        changed_files = parse_changes_csv(changes_csv, last_sync_at)
        if changed_files:
            download_individual_files(changed_files)

        metadata["last_sync"] = str(now)
        print(f"Sync complete at {datetime.now(tz=UTC)}")

    metadata = json.dumps(metadata, indent=2, sort_keys=True)
    with open(DATA_DIR / "metadata.json", "w") as f:
        f.write(metadata)


if __name__ == "__main__":
    main()
