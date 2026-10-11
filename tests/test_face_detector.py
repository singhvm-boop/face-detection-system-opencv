"""Tests for Haar Cascade face detection helpers.

These tests do not need a webcam. A real camera is optional and is covered
by the manual live-mode check described in the README.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

import cv2
import numpy as np
import pytest

from face_detector import (
    BOX_COLOR_BGR,
    CASCADE_FILENAME,
    FaceDetectionError,
    analyze_image,
    detection_message,
    detect_faces,
    draw_faces,
    encode_png,
    ensure_bgr,
    image_from_bytes,
    load_image,
    open_camera,
    cascade_path,
)


def test_cascade_file_ships_with_opencv() -> None:
    path = cascade_path()
    assert path.name == CASCADE_FILENAME
    assert path.is_file()
    assert path.stat().st_size > 0


def test_blank_image_has_no_faces() -> None:
    blank = np.full((240, 320, 3), 255, dtype=np.uint8)
    faces = detect_faces(blank)
    assert faces == []
    result = analyze_image(blank)
    assert result.count == 0
    assert result.message.startswith("No face detected")


def test_detection_messages() -> None:
    assert "No face detected" in detection_message(0)
    assert detection_message(1) == "1 face detected."
    assert detection_message(3) == "3 faces detected."
    with pytest.raises(FaceDetectionError):
        detection_message(-1)


def test_draw_faces_uses_green_boxes_and_keeps_original() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    original_pixel = image[30, 20].copy()
    annotated = draw_faces(image, [(20, 30, 40, 50)])

    assert annotated.shape == image.shape
    # The top-left corner of the rectangle is green in BGR order.
    assert tuple(int(v) for v in annotated[30, 20]) == BOX_COLOR_BGR
    assert tuple(int(v) for v in image[30, 20]) == tuple(int(v) for v in original_pixel)


def test_multiple_boxes_are_all_drawn() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    boxes = [(10, 20, 40, 40), (180, 60, 50, 50)]
    annotated = draw_faces(image, boxes)
    assert tuple(int(v) for v in annotated[20, 10]) == BOX_COLOR_BGR
    assert tuple(int(v) for v in annotated[60, 180]) == BOX_COLOR_BGR


def test_grayscale_and_bgra_are_accepted() -> None:
    gray = np.full((40, 50), 128, dtype=np.uint8)
    bgra = np.zeros((40, 50, 4), dtype=np.uint8)
    assert ensure_bgr(gray).shape == (40, 50, 3)
    assert ensure_bgr(bgra).shape == (40, 50, 3)
    assert detect_faces(gray) == []


def test_invalid_settings_are_rejected() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    with pytest.raises(FaceDetectionError):
        detect_faces(image, scale_factor=1.0)
    with pytest.raises(FaceDetectionError):
        detect_faces(image, min_neighbors=-1)
    with pytest.raises(FaceDetectionError):
        detect_faces(image, min_size=0)


def test_missing_and_invalid_image_paths(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"
    with pytest.raises(FaceDetectionError, match="not found"):
        load_image(missing)

    not_a_file = tmp_path / "folder"
    not_a_file.mkdir()
    with pytest.raises(FaceDetectionError, match="not a file"):
        load_image(not_a_file)

    text_file = tmp_path / "notes.jpg"
    text_file.write_text("this is not an image", encoding="utf-8")
    with pytest.raises(FaceDetectionError, match="Could not read"):
        load_image(text_file)


def test_image_bytes_round_trip_and_rejection() -> None:
    image = np.full((60, 80, 3), (10, 20, 30), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    decoded = image_from_bytes(buffer.tobytes())
    assert decoded.shape == image.shape

    with pytest.raises(FaceDetectionError, match="empty"):
        image_from_bytes(b"")
    with pytest.raises(FaceDetectionError, match="Could not read"):
        image_from_bytes(b"not-an-image")


def test_encode_png_of_annotated_image() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    annotated = draw_faces(image, [(5, 5, 30, 30)])
    payload = encode_png(annotated)
    assert payload.startswith(b"\x89PNG")


def test_large_blank_image_returns_quickly_and_finds_nothing() -> None:
    # Wider than the internal detection width, so the resize path runs.
    wide = np.full((400, 2000, 3), 180, dtype=np.uint8)
    assert detect_faces(wide) == []


def test_unavailable_camera_index_is_reported() -> None:
    # Index 99 should not exist. This must fail clearly and must not hang.
    with pytest.raises(FaceDetectionError, match="Could not open webcam"):
        open_camera(99)


def test_included_portrait_detects_one_face() -> None:
    portrait = Path(__file__).resolve().parents[1] / "samples" / "portrait.jpg"
    result = analyze_image(load_image(portrait))
    assert result.count == 1
    assert result.message == "1 face detected."
    x, y, width, height = result.faces[0]
    assert width > 100 and height > 100
    assert x > 0 and y > 0


def test_app_source_compiles() -> None:
    project = Path(__file__).resolve().parents[1]
    py_compile.compile(str(project / "app.py"), doraise=True)
    py_compile.compile(str(project / "webcam_live.py"), doraise=True)
