import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.formatting.format_audio import detect_anchor_type, find_anchor_time


class FormatAudioAnchorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.transcript = Path(self.temp_dir.name) / "transcript.csv"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_transcript(self, rows: list[dict]) -> None:
        pd.DataFrame(rows).to_csv(self.transcript, index=False)

    def test_split_anchor_uses_row_containing_first_anchor_word(self) -> None:
        self.write_transcript(
            [
                {"Start_Time": 2.0, "End_Time": 3.0, "Text": "Are you ready?"},
                {"Start_Time": 10.0, "End_Time": 11.0, "Text": "Thanks for"},
                {"Start_Time": 11.0, "End_Time": 12.0, "Text": "coming today."},
            ]
        )

        self.assertEqual(detect_anchor_type(self.transcript), "anchor_2")
        self.assertEqual(
            find_anchor_time(self.transcript, "anchor_2"),
            {"anchor_time": 10.0, "anchor_text": "Thanks for coming"},
        )

    def test_hi_im_anchor_can_span_rows(self) -> None:
        self.write_transcript(
            [
                {"Start_Time": 1.0, "End_Time": 2.0, "Text": "Hi,"},
                {"Start_Time": 2.0, "End_Time": 3.0, "Text": "I'm Ellie."},
            ]
        )

        self.assertEqual(
            find_anchor_time(self.transcript, "anchor_1"),
            {"anchor_time": 1.0, "anchor_text": "Hi, I'm"},
        )

    def test_unrelated_earlier_word_does_not_set_anchor_time(self) -> None:
        self.write_transcript(
            [
                {"Start_Time": 3.0, "End_Time": 4.0, "Text": "How was your trip?"},
                {"Start_Time": 20.0, "End_Time": 21.0, "Text": "How are you"},
                {"Start_Time": 21.0, "End_Time": 22.0, "Text": "doing today?"},
            ]
        )

        result = find_anchor_time(self.transcript, "anchor_2")
        self.assertIsNotNone(result)
        self.assertEqual(result["anchor_time"], 20.0)

    def test_earlier_hi_is_not_combined_with_later_im(self) -> None:
        self.write_transcript(
            [
                {"Start_Time": 1.0, "End_Time": 2.0, "Text": "Hi,"},
                {"Start_Time": 2.0, "End_Time": 3.0, "Text": "a"},
                {"Start_Time": 3.0, "End_Time": 4.0, "Text": "b"},
                {"Start_Time": 4.0, "End_Time": 5.0, "Text": "c"},
                {"Start_Time": 10.0, "End_Time": 11.0, "Text": "Hi,"},
                {"Start_Time": 11.0, "End_Time": 12.0, "Text": "I’m Ellie."},
            ]
        )

        self.assertEqual(
            find_anchor_time(self.transcript, "anchor_1"),
            {"anchor_time": 10.0, "anchor_text": "Hi, I’m"},
        )


if __name__ == "__main__":
    unittest.main()
