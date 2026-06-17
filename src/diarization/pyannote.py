from pathlib import Path
from typing import Any
from datetime import datetime

import torch
import soundfile as sf
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


def diarize_single_audio(
    pipeline,
    audio_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    data_id = int(audio_path.stem.split("_")[0])

    waveform_np, sample_rate = sf.read(
        str(audio_path),
        always_2d=True,
    )

    waveform = torch.tensor(
        waveform_np.T,
        dtype=torch.float32,
    )

    output = pipeline(
    {
        "waveform": waveform,
        "sample_rate": sample_rate,
        }
    )

    diarization = output.speaker_diarization

    segments = list(
        diarization.itertracks(
            yield_label=True
        )
    )

    if len(segments) == 0:
        raise RuntimeError(f"Empty diarization for {audio_path.name}")

    rows = []

    for turn, _, speaker in segments:
        rows.append(
            {
                "Start_Time": turn.start,
                "End_Time": turn.end,
                "Speaker": speaker,
            }
        )

    df = pd.DataFrame(rows)

    csv_path = output_dir / f"{data_id}_diarization.csv"
    df.to_csv(csv_path, index=False)

    return {
        "data_id": data_id,
        "csv_path": csv_path,
        "num_segments": len(rows),
        "num_speakers": df["Speaker"].nunique(),
    }


def run_diarization_pipeline(
    pipeline,
    audio_paths: list[Path],
    output_dir: Path,
    success_log: Path,
    failure_log: Path,
    overwrite: bool = False,
) -> dict[str, int]:

    success_count = 0
    failed_count = 0
    skipped_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    success_log.parent.mkdir(parents=True, exist_ok=True)
    failure_log.parent.mkdir(parents=True, exist_ok=True)

    for audio_path in tqdm(
        audio_paths,
        desc="Running Diarization",
        unit="audio",
    ):
        data_id = int(audio_path.stem.split("_")[0])
        output_csv = output_dir / f"{data_id}_diarization.csv"

        if output_csv.exists() and not overwrite:
            skipped_count += 1
            tqdm.write(f"[SKIP] {data_id}")
            continue

        try:
            result = diarize_single_audio(
                pipeline=pipeline,
                audio_path=audio_path,
                output_dir=output_dir,
            )

            append_log(
                {
                    "data_id": result["data_id"],
                    "csv_path": str(result["csv_path"]),
                    "num_segments": result["num_segments"],
                    "num_speakers": result["num_speakers"],
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
                f"[FAILED] {audio_path.name} | {type(e).__name__}: {e}"
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