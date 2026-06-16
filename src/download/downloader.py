from pathlib import Path
import shutil
import tarfile
import requests
from typing import Any

def get_missing_data_list(
    start_id: int,
    end_id: int,
    audio_dir: Path,
    transcript_dir: Path,
):
    """
    Check for missing or corrupted data files and return a list of IDs that need to be downloaded.
    Args:
        start_id (int): The starting ID of the data range to check.
        end_id (int): The ending ID of the data range to check.
        audio_dir (Path): The directory where audio files are stored.
        transcript_dir (Path): The directory where transcript files are stored.
    Returns:
        List[int]: A list of IDs that need to be downloaded.    
    """
    download_ids = []

    completed = 0
    corrupted = 0
    missing = 0

    for pid in range(start_id, end_id + 1):

        audio_file = audio_dir / f"{pid}_AUDIO.wav"
        transcript_file = transcript_dir / f"{pid}_TRANSCRIPT.csv"

        audio_exists = audio_file.exists()
        transcript_exists = transcript_file.exists()

        if audio_exists and transcript_exists:
            completed += 1
            continue

        if audio_exists != transcript_exists:
            corrupted += 1

            if audio_exists:
                audio_file.unlink()

            if transcript_exists:
                transcript_file.unlink()

            download_ids.append(pid)
            continue

        missing += 1
        download_ids.append(pid)

    print(f"Completed : {completed}")
    print(f"Corrupted : {corrupted}")
    print(f"Missing   : {missing}")
    print(f"Need Download : {len(download_ids)}")

    return download_ids

from pathlib import Path
from typing import Any
import requests


def download_data_archive(
    base_url: str,
    data_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Download a DAIC-WOZ data archive and save it locally.

    Args:
        base_url (str): Base URL of the dataset archive directory.
        data_id (int): data ID to download.
        output_dir (Path): Directory where the archive will be saved.

    Returns:
        dict[str, Any]: Download metadata containing:
            - file_path: Path to the downloaded archive
            - status_code: HTTP status code
            - size_mb: downloaded file size in MB
            - url: full download URL

    Raises:
        requests.HTTPError: If the server returns an error status.
        ValueError: If the downloaded file size does not match Content-Length.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{data_id}_P.tar.gz"
    save_path = output_dir / filename
    tmp_path = output_dir / f"{filename}.tmp"

    url = f"{base_url.rstrip('/')}/{filename}"

    if save_path.exists():
        size_mb = save_path.stat().st_size / 1024 / 1024
        print(f"[SKIP] {filename} already exists ({size_mb:.2f} MB)")
        return {
            "file_path": save_path,
            "status_code": 200,
            "size_mb": round(size_mb, 2),
            "url": url,
            "status": "skipped",
        }

    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[DOWNLOAD] {filename}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    expected_size = int(response.headers.get("Content-Length", 0))

    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    actual_size = tmp_path.stat().st_size

    if expected_size > 0 and actual_size != expected_size:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"Incomplete download for {filename}: "
            f"expected {expected_size} bytes, got {actual_size} bytes."
        )

    tmp_path.rename(save_path)

    size_mb = actual_size / 1024 / 1024

    print(f"[DONE] {filename} ({size_mb:.2f} MB)")

    return {
        "file_path": save_path,
        "status_code": response.status_code,
        "size_mb": round(size_mb, 2),
        "url": url,
        "status": "downloaded",
    }

def extract_data_archive(
    archive_path: Path,
    audio_dir: Path,
    transcript_dir: Path,
) -> dict:
    """
    Extract a DAIC-WOZ archive and move audio/transcript
    files into their target directories.
    Args:
        archive_path (Path): Path to the downloaded archive.
        audio_dir (Path): Directory where audio files should be stored.
        transcript_dir (Path): Directory where transcript files should be stored.
    Returns:
        dict: Metadata about the extracted files, including:
            - data_id: The ID of the extracted data
            - audio_path: Path to the extracted audio file
            - transcript_path: Path to the extracted transcript file
            Raises:
            FileNotFoundError: If the archive or expected files are not found.
        
    """

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Archive not found: {archive_path}"
        )

    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    data_id = archive_path.name.split("_")[0]

    temp_dir = archive_path.parent / f"{data_id}_P"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(archive_path.parent)

    source_audio = temp_dir / f"{data_id}_AUDIO.wav"
    source_transcript = temp_dir / f"{data_id}_TRANSCRIPT.csv"

    if not source_audio.exists():
        raise FileNotFoundError(source_audio)

    if not source_transcript.exists():
        raise FileNotFoundError(source_transcript)

    target_audio = audio_dir / source_audio.name
    target_transcript = transcript_dir / source_transcript.name

    if target_audio.exists():
        target_audio.unlink()

    if target_transcript.exists():
        target_transcript.unlink()

    shutil.move(str(source_audio), str(target_audio))
    shutil.move(str(source_transcript), str(target_transcript))

    shutil.rmtree(temp_dir)

    return {
        "data_id": int(data_id),
        "audio_path": target_audio,
        "transcript_path": target_transcript,
    }