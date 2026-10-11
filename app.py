"""Streamlit interface for the Haar Cascade face detection project.

Run it from the project folder:

    streamlit run app.py
"""

from __future__ import annotations

import atexit
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from face_detector import (
    BOX_COLOR_BGR,
    CASCADE_FILENAME,
    FaceDetectionError,
    analyze_image,
    bgr_to_rgb,
    cascade_path,
    detection_message,
    draw_faces,
    encode_png,
    image_from_bytes,
    load_image,
    open_camera,
    read_frame,
)

SAMPLE_DIR = Path(__file__).resolve().parent / "samples"
# One camera handle for this local app. Streamlit reruns the script often,
# so the capture object lives here instead of being reopened every rerun.
_LIVE_CAMERA: cv2.VideoCapture | None = None
_LIVE_CAMERA_INDEX: int | None = None


def release_live_camera() -> None:
    """Close the webcam so another program can use it."""
    global _LIVE_CAMERA, _LIVE_CAMERA_INDEX
    if _LIVE_CAMERA is not None:
        _LIVE_CAMERA.release()
        _LIVE_CAMERA = None
        _LIVE_CAMERA_INDEX = None


atexit.register(release_live_camera)


def _live_camera(camera_index: int) -> cv2.VideoCapture:
    """Reuse an open camera, or open a new one when the index changes."""
    global _LIVE_CAMERA, _LIVE_CAMERA_INDEX
    if (
        _LIVE_CAMERA is None
        or not _LIVE_CAMERA.isOpened()
        or _LIVE_CAMERA_INDEX != camera_index
    ):
        release_live_camera()
        _LIVE_CAMERA = open_camera(camera_index)
        _LIVE_CAMERA_INDEX = camera_index
    return _LIVE_CAMERA


def _settings_from_sidebar() -> dict[str, float | int]:
    st.sidebar.header("Detection settings")
    st.sidebar.caption(
        "These are the Haar Cascade controls. The defaults work for most photos."
    )
    scale_factor = st.sidebar.slider(
        "Scale factor",
        min_value=1.05,
        max_value=1.50,
        value=1.10,
        step=0.05,
        help="How much the image shrinks between scans. Lower is slower and more thorough.",
    )
    min_neighbors = st.sidebar.slider(
        "Minimum neighbors",
        min_value=1,
        max_value=10,
        value=5,
        help="Overlapping detections required to keep a box. Higher rejects more false faces.",
    )
    min_size = st.sidebar.slider(
        "Minimum face size (pixels)",
        min_value=20,
        max_value=120,
        value=30,
        step=5,
        help="Boxes smaller than this are ignored.",
    )
    camera_index = st.sidebar.number_input(
        "Camera index",
        min_value=0,
        max_value=8,
        value=0,
        step=1,
        help="0 is usually the built-in webcam. Try 1 if you have an external camera.",
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        f"**Method:** Haar Cascade  \n**Model file:** `{CASCADE_FILENAME}`"
    )
    return {
        "scale_factor": float(scale_factor),
        "min_neighbors": int(min_neighbors),
        "min_size": int(min_size),
        "camera_index": int(camera_index),
    }


def _show_result(original_bgr: np.ndarray, annotated_bgr: np.ndarray, count: int, message: str) -> None:
    """Show the count, both images, and a PNG download of the annotated result."""
    if count == 0:
        st.warning(message)
    else:
        st.success(message)

    metric_col, _ = st.columns([1, 3])
    metric_col.metric("Faces detected", count)

    original_col, annotated_col = st.columns(2)
    with original_col:
        st.markdown("**Original**")
        st.image(bgr_to_rgb(original_bgr), width="stretch")
    with annotated_col:
        st.markdown("**Detected faces**")
        st.image(bgr_to_rgb(annotated_bgr), width="stretch")

    st.download_button(
        label="Download processed image",
        data=encode_png(annotated_bgr),
        file_name=f"faces-detected-{count}.png",
        mime="image/png",
        type="primary",
    )


def _remember_result(source_id: str, original_bgr: np.ndarray, settings: dict) -> None:
    result = analyze_image(
        original_bgr,
        scale_factor=float(settings["scale_factor"]),
        min_neighbors=int(settings["min_neighbors"]),
        min_size=int(settings["min_size"]),
    )
    st.session_state.detection = {
        "source_id": source_id,
        "original_bgr": result.image_bgr,
        "annotated_bgr": result.annotated_bgr,
        "count": result.count,
        "message": result.message,
    }


def _render_stored_result(source_id: str) -> None:
    stored = st.session_state.get("detection")
    if not stored or stored.get("source_id") != source_id:
        return
    _show_result(
        stored["original_bgr"],
        stored["annotated_bgr"],
        stored["count"],
        stored["message"],
    )


def _sample_images() -> list[Path]:
    if not SAMPLE_DIR.is_dir():
        return []
    files: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(SAMPLE_DIR.glob(pattern))
    return sorted(files)


def image_mode(settings: dict) -> None:
    """Upload a photo, detect on button click, and offer a download."""
    st.subheader("Detect faces in an image")
    st.write(
        "Upload a photo of one person or a group. The detector draws a green "
        "rectangle around each frontal face it finds."
    )

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        help="JPG and PNG photos work best.",
    )
    # A new upload replaces the sample. The same upload is kept across reruns
    # so that clicking Detect faces does not discard the file.
    if uploaded is not None:
        upload_id = f"upload:{getattr(uploaded, 'file_id', uploaded.name)}:{uploaded.size}"
        if st.session_state.get("seen_upload_id") != upload_id:
            st.session_state.seen_upload_id = upload_id
            try:
                st.session_state.loaded_image = {
                    "id": upload_id,
                    "bgr": image_from_bytes(uploaded.getvalue()),
                    "label": uploaded.name,
                }
            except FaceDetectionError as exc:
                st.error(str(exc))
                return

    # Applied after the upload check so this click wins even when a file is
    # already sitting in the uploader.
    sample_files = _sample_images()
    if sample_files and st.button("Load sample photo"):
        sample = sample_files[0]
        try:
            st.session_state.loaded_image = {
                "id": f"sample:{sample.name}",
                "bgr": load_image(sample),
                "label": sample.name,
            }
        except FaceDetectionError as exc:
            st.error(str(exc))
            return

    loaded = st.session_state.get("loaded_image")
    if not loaded:
        st.info("Upload an image or load the sample photo, then click Detect faces.")
        return

    st.caption(f"Ready: {loaded['label']}")
    if st.button("Detect faces", type="primary"):
        try:
            _remember_result(loaded["id"], loaded["bgr"], settings)
        except FaceDetectionError as exc:
            st.error(str(exc))
            return

    if st.session_state.get("detection", {}).get("source_id") != loaded["id"]:
        st.info("Image loaded. Click Detect faces to draw the boxes.")
    _render_stored_result(loaded["id"])


def camera_photo_mode(settings: dict) -> None:
    """Take one photo with the browser camera and detect faces in it."""
    st.subheader("Detect faces in a camera photo")
    st.write(
        "Your browser will ask for camera permission. This captures a single "
        "photo on this device, then runs the same detector used for uploads."
    )
    photo = st.camera_input("Take a photo")
    if photo is None:
        st.info("Take a photo, then click Detect faces.")
        return

    try:
        image_bgr = image_from_bytes(photo.getvalue())
    except FaceDetectionError as exc:
        st.error(str(exc))
        return

    source_id = f"camera-photo:{len(photo.getvalue())}:{photo.file_id if hasattr(photo, 'file_id') else hash(photo.getvalue())}"
    if st.button("Detect faces in this photo", type="primary"):
        try:
            _remember_result(source_id, image_bgr, settings)
        except FaceDetectionError as exc:
            st.error(str(exc))
            return
    _render_stored_result(source_id)


@st.fragment(run_every=timedelta(milliseconds=150))
def _live_frames(camera_index: int, settings: dict) -> None:
    """Grab one webcam frame, detect faces, and refresh about six times a second."""
    try:
        capture = _live_camera(camera_index)
        frame = read_frame(capture)
        faces = analyze_image(
            frame,
            scale_factor=float(settings["scale_factor"]),
            min_neighbors=int(settings["min_neighbors"]),
            min_size=int(settings["min_size"]),
        )
    except FaceDetectionError as exc:
        release_live_camera()
        st.session_state.live_on = False
        st.session_state.camera_error = str(exc)
        # Leave the fragment and rerun the whole page so the error stays visible
        # and the camera is not opened again on the next tick.
        st.rerun(scope="app")
        return

    annotated = draw_faces(faces.image_bgr, faces.faces, color=BOX_COLOR_BGR)
    st.image(bgr_to_rgb(annotated), width="stretch")
    if faces.count == 0:
        st.warning(detection_message(0))
    else:
        st.success(faces.message)


def live_mode(settings: dict) -> None:
    """Stream the computer's webcam until the user presses Stop."""
    st.subheader("Live webcam detection")
    st.write(
        "This uses the webcam on the computer that is running the app. "
        "Stand in front of that computer, then press Stop when you are finished "
        "so the camera is released. Q and Esc quit the separate desktop window "
        "(`python webcam_live.py`), not this page."
    )

    start_col, stop_col = st.columns(2)
    with start_col:
        if st.button("Start live detection", type="primary"):
            st.session_state.live_on = True
            st.session_state.camera_error = None
    with stop_col:
        if st.button("Stop and release camera"):
            release_live_camera()
            st.session_state.live_on = False
            st.session_state.camera_error = None
            st.rerun()

    if st.session_state.get("camera_error"):
        st.error(st.session_state.camera_error)
        return

    if not st.session_state.get("live_on"):
        st.info("Live detection is off. Press Start live detection when you are ready.")
        return

    _live_frames(int(settings["camera_index"]), settings)


def main() -> None:
    st.set_page_config(
        page_title="Face Detection System",
        page_icon=":large_green_square:",
        layout="wide",
    )
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.4rem; max-width: 1100px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Face Detection System")
    st.caption("OpenCV and a Haar Cascade classifier — image upload and live webcam.")

    try:
        cascade_file = cascade_path()
    except FaceDetectionError as exc:
        st.error(str(exc))
        st.stop()

    settings = _settings_from_sidebar()
    st.sidebar.caption(f"Cascade loaded from OpenCV: `{cascade_file.name}`")

    mode = st.radio(
        "Choose a mode",
        ["Upload an image", "Camera photo", "Live webcam"],
        horizontal=True,
        label_visibility="collapsed",
    )

    with st.expander("How detection works in this project"):
        st.markdown(
            """
            The app converts the photo to grayscale, equalizes contrast, and
            slides a window across several image sizes. Each window is scored
            by a cascade of simple light-and-dark patterns (Haar features)
            trained to keep regions that look like a frontal face.

            A green rectangle is drawn for every detection that survives the
            minimum-neighbor check. This identifies *where* a face is. It does
            not identify *who* the person is.
            """
        )

    if mode == "Upload an image":
        image_mode(settings)
    elif mode == "Camera photo":
        camera_photo_mode(settings)
    else:
        live_mode(settings)


main()
