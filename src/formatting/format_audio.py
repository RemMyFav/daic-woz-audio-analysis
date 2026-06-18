from pathlib import Path
from typing import Any
import re
import wave
import shutil

import pandas as pd
from tqdm.auto import tqdm


ANCHOR_1_PATTERN = r"\bhi\b[\s\W]*\b(i['‘’]?m|i\s+am)\b"

ANCHOR_2_PATTERN = (
    r"\bthanks\b[\s\W]*\bfor\b[\s\W]*\bcoming\b"
    r"|"
    r"\b(are|how\s+are)\b[\s\W]*\byou\b[\s\W]*\bdoing\b[\s\W]*\btoday\b"
)

ANCHOR_PATTERNS = {
    "anchor_1": ANCHOR_1_PATTERN,
    "anchor_2": ANCHOR_2_PATTERN,
}


def _find_anchor(
    transcript_csv: Path,
    anchor_type: str | None = None,
    max_time: float = 200,
) -> dict[str, Any] | None:
    """Find an anchor and map its first matched word to its source row."""
    if anchor_type is not None and anchor_type not in ANCHOR_PATTERNS:
        raise ValueError(f"Unknown anchor_type: {anchor_type}")

    df = pd.read_csv(transcript_csv)
    df = df[df["Start_Time"] <= max_time]

    text_parts: list[str] = []
    row_spans: list[tuple[int, int, Any]] = []
    cursor = 0

    for index, value in df["Text"].items():
        text = "" if pd.isna(value) else str(value)

        if text_parts:
            text_parts.append(" ")
            cursor += 1

        start = cursor
        text_parts.append(text)
        cursor += len(text)
        row_spans.append((start, cursor, index))

    full_text = "".join(text_parts)
    anchor_types = [anchor_type] if anchor_type else list(ANCHOR_PATTERNS)

    for candidate_type in anchor_types:
        match = re.search(
            ANCHOR_PATTERNS[candidate_type],
            full_text,
            flags=re.IGNORECASE,
        )
        if match is None:
            continue

        match_start = match.start()
        for row_start, row_end, index in row_spans:
            if row_start <= match_start < row_end:
                row = df.loc[index]
                return {
                    "anchor_type": candidate_type,
                    "anchor_time": float(row["Start_Time"]),
                    "anchor_text": match.group(0),
                }

    return None


def detect_anchor_type(
    transcript_csv: Path,
    max_time: float = 200,
) -> str | None:
    """
    Return:
        "anchor_1" if hi + I'm anchor exists
        "anchor_2" if secondary anchor exists
        None otherwise
    """

    anchor = _find_anchor(
        transcript_csv=transcript_csv,
        max_time=max_time,
    )
    return None if anchor is None else anchor["anchor_type"]


def find_anchor_time(
    transcript_csv: Path,
    anchor_type: str,
    max_time: float = 200,
) -> dict[str, Any] | None:
    anchor = _find_anchor(
        transcript_csv=transcript_csv,
        anchor_type=anchor_type,
        max_time=max_time,
    )
    if anchor is None:
        return None

    return {
        "anchor_time": anchor["anchor_time"],
        "anchor_text": anchor["anchor_text"],
    }


def trim_wav_from_time(
    input_wav: Path,
    output_wav: Path,
    start_time: float,
    pre_roll: float = 0.0,
) -> None:
    output_wav.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cut_time = max(
        0.0,
        start_time - pre_roll,
    )

    with wave.open(str(input_wav), "rb") as reader:
        params = reader.getparams()
        frame_rate = reader.getframerate()
        total_frames = reader.getnframes()

        start_frame = int(
            cut_time * frame_rate
        )

        if start_frame >= total_frames:
            raise ValueError(
                f"Cut time exceeds audio length: {input_wav}"
            )

        reader.setpos(start_frame)

        frames = reader.readframes(
            total_frames - start_frame
        )

    with wave.open(str(output_wav), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)


def trim_transcript_from_time(
    input_csv: Path,
    output_csv: Path,
    start_time: float,
    pre_roll: float = 0.0,
) -> None:
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cut_time = max(
        0.0,
        start_time - pre_roll,
    )

    df = pd.read_csv(input_csv)

    df = df[
        df["Start_Time"] >= cut_time
    ].copy()

    df["Start_Time"] = (
        df["Start_Time"] - cut_time
    ).clip(lower=0)

    df["End_Time"] = (
        df["End_Time"] - cut_time
    ).clip(lower=0)

    df.to_csv(
        output_csv,
        index=False,
    )


def format_audio_database(
    asr_dir: Path,
    audio_dir: Path,
    output_root: Path,
    max_time: float = 200,
    pre_roll: float = 0.0,
) -> pd.DataFrame:

    formatted_audio_dir = output_root / "formatted" / "audio"
    formatted_transcript_dir = output_root / "formatted" / "transcript"

    unformatted_audio_dir = output_root / "unformatted" / "audio"
    unformatted_transcript_dir = output_root / "unformatted" / "transcript"

    log_dir = output_root / "logs"

    for directory in [
        formatted_audio_dir,
        formatted_transcript_dir,
        unformatted_audio_dir,
        unformatted_transcript_dir,
        log_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    rows = []

    transcript_paths = sorted(
        asr_dir.glob("*_whisper.csv")
    )

    for transcript_csv in tqdm(
        transcript_paths,
        desc="Formatting database",
        unit="file",
    ):
        data_id = int(
            transcript_csv.stem.split("_")[0]
        )

        input_wav = audio_dir / f"{data_id}_AUDIO.wav"

        if not input_wav.exists():
            rows.append(
                {
                    "data_id": data_id,
                    "status": "missing_wav",
                    "anchor_type": None,
                    "anchor_time": None,
                    "anchor_text": None,
                    "error": "wav file not found",
                }
            )
            continue

        anchor_type = detect_anchor_type(
            transcript_csv=transcript_csv,
            max_time=max_time,
        )

        if anchor_type is None:
            target_wav = unformatted_audio_dir / input_wav.name
            target_transcript = unformatted_transcript_dir / transcript_csv.name

            shutil.copy2(input_wav, target_wav)
            shutil.copy2(transcript_csv, target_transcript)

            rows.append(
                {
                    "data_id": data_id,
                    "status": "unformatted",
                    "anchor_type": None,
                    "anchor_time": None,
                    "anchor_text": None,
                    "output_wav": str(target_wav),
                    "output_transcript": str(target_transcript),
                    "error": "no anchor found",
                }
            )
            continue

        anchor = find_anchor_time(
            transcript_csv=transcript_csv,
            anchor_type=anchor_type,
            max_time=max_time,
        )

        if anchor is None:
            target_wav = unformatted_audio_dir / input_wav.name
            target_transcript = unformatted_transcript_dir / transcript_csv.name

            shutil.copy2(input_wav, target_wav)
            shutil.copy2(transcript_csv, target_transcript)

            rows.append(
                {
                    "data_id": data_id,
                    "status": "unformatted",
                    "anchor_type": anchor_type,
                    "anchor_time": None,
                    "anchor_text": None,
                    "output_wav": str(target_wav),
                    "output_transcript": str(target_transcript),
                    "error": "anchor type found but trigger time not found",
                }
            )
            continue

        output_wav = formatted_audio_dir / input_wav.name
        output_transcript = formatted_transcript_dir / transcript_csv.name

        try:
            trim_wav_from_time(
                input_wav=input_wav,
                output_wav=output_wav,
                start_time=anchor["anchor_time"],
                pre_roll=pre_roll,
            )

            trim_transcript_from_time(
                input_csv=transcript_csv,
                output_csv=output_transcript,
                start_time=anchor["anchor_time"],
                pre_roll=pre_roll,
            )

            rows.append(
                {
                    "data_id": data_id,
                    "status": "formatted",
                    "anchor_type": anchor_type,
                    "anchor_time": anchor["anchor_time"],
                    "anchor_text": anchor["anchor_text"],
                    "output_wav": str(output_wav),
                    "output_transcript": str(output_transcript),
                    "error": None,
                }
            )

        except Exception as e:
            target_wav = unformatted_audio_dir / input_wav.name
            target_transcript = unformatted_transcript_dir / transcript_csv.name

            shutil.copy2(input_wav, target_wav)
            shutil.copy2(transcript_csv, target_transcript)

            rows.append(
                {
                    "data_id": data_id,
                    "status": "failed_trim",
                    "anchor_type": anchor_type,
                    "anchor_time": anchor["anchor_time"],
                    "anchor_text": anchor["anchor_text"],
                    "output_wav": str(target_wav),
                    "output_transcript": str(target_transcript),
                    "error": str(e),
                }
            )

    log_df = pd.DataFrame(rows)

    log_df.to_csv(
        log_dir / "formatting_log.csv",
        index=False,
    )

    return log_df
