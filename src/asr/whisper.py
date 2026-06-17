from pathlib import Path
from typing import Any

import pandas as pd
import whisper


def transcribe_audio(
    model,
    audio_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Transcribe a single audio file using Whisper and save
    segment-level results as a CSV file.

    Args:
        model:
            Loaded Whisper model.

        audio_path:
            Path to input audio file.

        output_dir:
            Directory for saving outputs.

    Returns:
        dict[str, Any]:
            {
                "data_id": int,
                "csv_path": Path,
            }

    Raises:
        FileNotFoundError:
            If audio file does not exist.

        RuntimeError:
            If Whisper returns no segments.
    """

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_id = int(
        audio_path.stem.split("_")[0]
    )

    result = model.transcribe(
        str(audio_path),
        language="en",
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
                    1 - segment.get(
                        "no_speech_prob",
                        0,
                    ),
                    4,
                ),
            }
        )

    df = pd.DataFrame(rows)

    csv_path = (
        output_dir
        / f"{data_id}_whisper.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    return {
        "data_id": data_id,
        "csv_path": csv_path,
    }