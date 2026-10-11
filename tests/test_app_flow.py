"""Streamlit flow tests that do not require a webcam or a browser."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def _app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=30).run()


def _click(at: AppTest, label: str) -> None:
    for button in at.button:
        if button.label == label:
            button.click()
            return
    raise AssertionError(f"Button not found: {label}")


def test_sample_photo_shows_one_face_and_download() -> None:
    at = _app()
    _click(at, "Load sample photo")
    at.run()
    assert any("portrait.jpg" in caption.value for caption in at.caption)
    _click(at, "Detect faces")
    at.run()

    assert any(item.value == "1 face detected." for item in at.success)
    assert at.metric[0].label == "Faces detected"
    assert at.metric[0].value == "1"
    assert any(button.label == "Download processed image" for button in at.download_button)
    assert not at.exception


def test_blank_upload_reports_no_face() -> None:
    blank = np.full((180, 240, 3), 240, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", blank)
    assert ok

    at = _app()
    at.file_uploader[0].upload("blank-face.png", encoded.tobytes(), "image/png").run()
    _click(at, "Detect faces")
    at.run()

    assert any(item.value.startswith("No face detected") for item in at.warning)
    assert at.metric[0].value == "0"
    assert not at.exception


def test_live_mode_reports_a_missing_camera() -> None:
    at = _app()
    # Index 99 is not a real camera, including on laptops that have a webcam.
    at.number_input[0].set_value(99)
    at.radio[0].set_value("Live webcam").run()
    assert any("Live detection is off" in item.value for item in at.info)

    _click(at, "Start live detection")
    at.run()
    assert any("Could not open webcam" in item.value for item in at.error)
    assert not at.exception
