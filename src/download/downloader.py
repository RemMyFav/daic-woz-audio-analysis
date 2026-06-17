from pathlib import Path
from typing import Any
from datetime import datetime

import shutil
import tarfile
import requests
import pandas as pd
from tqdm.auto import tqdm


def append_log(row: dict[str, Any], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([row]).to_csv(
        log_path,
        mode="a",
        header=not log_path.exists(),
        index=False,
    )


def get_missing_data_list(
    data_ids: list[int],
    audio_dir: Path,
    transcript_dir: Path,
) -> list[int]:
    """
    Check for missing or corrupted audio/transcript pairs.

    Args:
        data_ids: Participant/data IDs to check.
        audio_dir: Directory where audio files are stored.
        transcript_dir: Directory where transcript files are stored.

    Returns:
        List of IDs that need to be downloaded.
    """
    download_ids = []

    completed = 0
    corrupted = 0
    missing = 0

    for data_id in data_ids:
        audio_file = audio_dir / f"{data_id}_AUDIO.wav"
        transcript_file = transcript_dir / f"{data_id}_TRANSCRIPT.csv"

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

            download_ids.append(data_id)
            continue

        missing += 1
        download_ids.append(data_id)

    print(f"Completed     : {completed}")
    print(f"Corrupted     : {corrupted}")
    print(f"Missing       : {missing}")
    print(f"Need Download : {len(download_ids)}")

    return download_ids


def download_data_archive(
    base_url: str,
    data_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Download a DAIC-WOZ archive.

    Returns:
        Metadata including file_path, size_mb, url, and status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{data_id}_P.tar.gz"
    save_path = output_dir / filename
    tmp_path = output_dir / f"{filename}.tmp"

    url = f"{base_url.rstrip('/')}/{filename}"

    if save_path.exists():
        size_mb = save_path.stat().st_size / 1024 / 1024
        return {
            "data_id": data_id,
            "file_path": save_path,
            "status_code": 200,
            "size_mb": round(size_mb, 2),
            "url": url,
            "status": "skipped",
        }

    if tmp_path.exists():
        tmp_path.unlink()

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

    return {
        "data_id": data_id,
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
) -> dict[str, Any]:
    """
    Extract a DAIC-WOZ archive and move audio/transcript files.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

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


def run_download_pipeline(
    data_ids: list[int],
    base_url: str,
    raw_dir: Path,
    audio_dir: Path,
    transcript_dir: Path,
    success_log: Path,
    failure_log: Path,
) -> dict[str, int]:
    """
    Download and extract DAIC-WOZ archives for a list of IDs.
    Logs success/failure internally.
    """
    success_count = 0
    failed_count = 0

    raw_dir.mkdir(parents=True, exist_ok=True)

    for data_id in tqdm(
        data_ids,
        desc="Downloading DAIC-WOZ",
        unit="file",
    ):
        archive_path = None

        try:
            download_result = download_data_archive(
                base_url=base_url,
                data_id=data_id,
                output_dir=raw_dir,
            )

            archive_path = download_result["file_path"]

            extract_result = extract_data_archive(
                archive_path=archive_path,
                audio_dir=audio_dir,
                transcript_dir=transcript_dir,
            )

            if archive_path.exists():
                archive_path.unlink()

            append_log(
                {
                    "data_id": data_id,
                    "status": download_result["status"],
                    "size_mb": download_result["size_mb"],
                    "audio_path": str(extract_result["audio_path"]),
                    "transcript_path": str(extract_result["transcript_path"]),
                    "timestamp": datetime.now().isoformat(),
                },
                success_log,
            )

            success_count += 1
            tqdm.write(f"[SUCCESS] {data_id}")

        except Exception as e:
            if archive_path is not None and archive_path.exists():
                archive_path.unlink()

            tmp_path = raw_dir / f"{data_id}_P.tar.gz.tmp"
            if tmp_path.exists():
                tmp_path.unlink()

            append_log(
                {
                    "data_id": data_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
                failure_log,
            )

            failed_count += 1
            tqdm.write(f"[FAILED] {data_id} | {type(e).__name__}: {e}")

    print("\n====================")
    print(f"Success : {success_count}")
    print(f"Failed  : {failed_count}")
    print("====================")

    return {
        "success": success_count,
        "failed": failed_count,
    }