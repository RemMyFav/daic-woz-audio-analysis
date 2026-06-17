from pathlib import Path
from typing import Any

import torch
from pyannote.audio import Pipeline


def diarize_audio(
    pipeline,
    audio_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Diarize a single audio file using Pyannote and save
    speaker segments as an RTTM file.

    Args:
        pipeline:
            Loaded Pyannote diarization pipeline.

        audio_path:
            Path to input audio file.

        output_dir:
            Directory for saving outputs.

    Returns:
        dict[str, Any]:
            {
                "data_id": int,
                "rttm_path": Path,
            }

    Raises:
        FileNotFoundError:
            If audio file does not exist.

        RuntimeError:
            If diarization returns no speaker segments.
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

    diarization = pipeline(
        str(audio_path)
    )

    segments = list(
        diarization.itertracks(
            yield_label=True
        )
    )

    if len(segments) == 0:
        raise RuntimeError(
            f"Empty diarization for {audio_path.name}"
        )

    rttm_path = (
        output_dir
        / f"{data_id}_diarization.rttm"
    )

    with open(
        rttm_path,
        "w",
        encoding="utf-8",
    ) as f:
        diarization.write_rttm(f)

    return {
        "data_id": data_id,
        "rttm_path": rttm_path,
    }