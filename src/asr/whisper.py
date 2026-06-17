from pathlib import Path
from typing import Any
from datetime import datetime

import pandas as pd
from tqdm.auto import tqdm


def append_log(
    row: dict[str, Any],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([row]).to_csv(
        log_path,
        mode="a",
        header=not log_path.exists(),
        index=False,
    )


def transcribe_single_audio(
    model,
    audio_path: Path,
    output_dir: Path,
    language: str = "en",
) -> dict[str, Any]:
    """
    Transcribe one audio file using Whisper and save segment-level
    results as a CSV file.
    """

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    data_id = int(audio_path.stem.split("_")[0])

    result = model.transcribe(
        str(audio_path),
        language=language,
        verbose=False,
    )

    segments = result.get("segments")

    if segments is None:
        raise RuntimeError(
            f"No segments returned for {audio_path.name}"
        )

    if len(segments) == 0:
        raise RuntimeError(
            f"Empty transcription for {audio_path.name}"
        )

    rows = []

    for segment in segments:
        rows.append(
            {
                "Start_Time": segment["start"],
                "End_Time": segment["end"],
                "Text": segment["text"].strip(),
                "Confidence": round(
                    1 - segment.get("no_speech_prob", 0),
                    4,
                ),
            }
        )

    df = pd.DataFrame(rows)

    csv_path = output_dir / f"{data_id}_whisper.csv"

    df.to_csv(
        csv_path,
        index=False,
    )

    return {
        "data_id": data_id,
        "csv_path": csv_path,
        "num_segments": len(rows),
        "num_words": len(result.get("text", "").split()),
    }


def run_whisper_pipeline(
    model,
    audio_paths: list[Path],
    output_dir: Path,
    success_log: Path,
    failure_log: Path,
    language: str = "en",
    overwrite: bool = False,
) -> dict[str, int]:
    """
    Run Whisper ASR on a list of audio files.
    Logs success/failure internally.
    """

    success_count = 0
    failed_count = 0
    skipped_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    success_log.parent.mkdir(parents=True, exist_ok=True)
    failure_log.parent.mkdir(parents=True, exist_ok=True)

    for audio_path in tqdm(
        audio_paths,
        desc="Running Whisper",
        unit="audio",
    ):
        data_id = int(audio_path.stem.split("_")[0])
        output_csv = output_dir / f"{data_id}_whisper.csv"

        if output_csv.exists() and not overwrite:
            skipped_count += 1
            tqdm.write(f"[SKIP] {data_id}")
            continue

        try:
            result = transcribe_single_audio(
                model=model,
                audio_path=audio_path,
                output_dir=output_dir,
                language=language,
            )

            append_log(
                {
                    "data_id": result["data_id"],
                    "csv_path": str(result["csv_path"]),
                    "num_segments": result["num_segments"],
                    "num_words": result["num_words"],
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                },
                success_log,
            )

            success_count += 1
            tqdm.write(f"[SUCCESS] {result['data_id']}")

        except Exception as e:
            append_log(
                {
                    "data_id": data_id,
                    "audio_file": str(audio_path),
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "status": "failed",
                    "timestamp": datetime.now().isoformat(),
                },
                failure_log,
            )

            failed_count += 1
            tqdm.write(
                f"[FAILED] {audio_path.name} | "
                f"{type(e).__name__}: {e}"
            )

    print("\n====================")
    print(f"Success : {success_count}")
    print(f"Skipped : {skipped_count}")
    print(f"Failed  : {failed_count}")
    print("====================")

    return {
        "success": success_count,
        "skipped": skipped_count,
        "failed": failed_count,
    }