"""Face detection with OpenCV's Haar Cascade classifier.

The detector used here is the classic Viola-Jones cascade that ships with
OpenCV (``haarcascade_frontalface_default.xml``). It is not a deep-learning
model. Images stay in OpenCV's BGR channel order until a caller converts
them for display.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Green rectangles, as BGR (OpenCV order), not RGB.
BOX_COLOR_BGR: tuple[int, int, int] = (0, 255, 0)
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

# Detecting on very large photos is slow and does not improve Haar results.
# Boxes found on the resized copy are mapped back to the original pixels.
MAX_DETECT_WIDTH = 1280

_cascade: cv2.CascadeClassifier | None = None


class FaceDetectionError(Exception):
    """A problem the user can fix: a missing file, a bad image, or no camera."""


def cascade_path() -> Path:
    """Return the Haar cascade XML path bundled with the installed OpenCV."""
    directory = Path(cv2.data.haarcascades)
    path = directory / CASCADE_FILENAME
    if not path.is_file():
        raise FaceDetectionError(
            f"Could not find {CASCADE_FILENAME} in {directory}. "
            "Reinstall opencv-python-headless so the cascade files are included."
        )
    return path


def get_cascade() -> cv2.CascadeClassifier:
    """Load the frontal-face cascade once and reuse it for every frame."""
    global _cascade
    if _cascade is None:
        path = cascade_path()
        classifier = cv2.CascadeClassifier(str(path))
        if classifier.empty():
            raise FaceDetectionError(f"OpenCV could not load the cascade file at {path}.")
        _cascade = classifier
    return _cascade


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Accept grayscale, BGR, or BGRA arrays and return a 3-channel BGR image."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise FaceDetectionError("The image is empty or could not be decoded.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise FaceDetectionError(
        "Unsupported image shape. Expected a grayscale or color image."
    )


def load_image(path: str | Path) -> np.ndarray:
    """Read an image from disk. Raises FaceDetectionError with a clear reason."""
    image_path = Path(path)
    if not image_path.exists():
        raise FaceDetectionError(f"Image file not found: {image_path}")
    if not image_path.is_file():
        raise FaceDetectionError(f"Path is not a file: {image_path}")

    # IMREAD_UNCHANGED keeps an alpha channel when present; ensure_bgr drops it.
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FaceDetectionError(
            f"Could not read '{image_path.name}'. Use a JPG, PNG, BMP, or TIFF image."
        )
    return ensure_bgr(image)


def image_from_bytes(data: bytes) -> np.ndarray:
    """Decode an uploaded image. Used by the Streamlit interface."""
    if not data:
        raise FaceDetectionError("The uploaded file is empty.")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FaceDetectionError(
            "Could not read that file. Use a JPG, PNG, BMP, or TIFF image."
        )
    return ensure_bgr(image)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB for libraries that display RGB, such as Streamlit."""
    return cv2.cvtColor(ensure_bgr(image_bgr), cv2.COLOR_BGR2RGB)


def encode_png(image_bgr: np.ndarray) -> bytes:
    """Encode a BGR image as PNG bytes for download."""
    ok, buffer = cv2.imencode(".png", ensure_bgr(image_bgr))
    if not ok:
        raise FaceDetectionError("Could not encode the processed image as PNG.")
    return buffer.tobytes()


def detection_message(count: int) -> str:
    """Human-readable summary. Zero faces get an explicit message."""
    if count < 0:
        raise FaceDetectionError("Face count cannot be negative.")
    if count == 0:
        return (
            "No face detected. Try a clearer, front-facing photo with better lighting, "
            "or lower the minimum face size."
        )
    if count == 1:
        return "1 face detected."
    return f"{count} faces detected."


def _validate_settings(
    scale_factor: float, min_neighbors: int, min_size: int
) -> None:
    # scaleFactor must be greater than 1 because each pass shrinks the image.
    if scale_factor <= 1.0:
        raise FaceDetectionError("scaleFactor must be greater than 1.")
    if min_neighbors < 0:
        raise FaceDetectionError("minNeighbors cannot be negative.")
    if min_size < 1:
        raise FaceDetectionError("Minimum face size must be at least 1 pixel.")


def detect_faces(
    image_bgr: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: int = 30,
) -> list[tuple[int, int, int, int]]:
    """Find frontal faces and return boxes as (x, y, width, height).

    ``scale_factor`` controls how much the image shrinks between scans.
    ``min_neighbors`` is how many overlapping hits are required before a box
    is kept. ``min_size`` ignores boxes smaller than that many pixels.
    """
    _validate_settings(scale_factor, min_neighbors, min_size)
    original = ensure_bgr(image_bgr)
    height, width = original.shape[:2]

    scale = 1.0
    working = original
    if width > MAX_DETECT_WIDTH:
        scale = MAX_DETECT_WIDTH / float(width)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        working = cv2.resize(
            original,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    # Haar features are computed on a single brightness channel.
    # Histogram equalization helps when one side of the face is in shadow.
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    detections = get_cascade().detectMultiScale(
        gray,
        scaleFactor=float(scale_factor),
        minNeighbors=int(min_neighbors),
        minSize=(int(min_size), int(min_size)),
    )

    if detections is None or len(detections) == 0:
        return []

    inverse = 1.0 / scale
    faces: list[tuple[int, int, int, int]] = []
    for x, y, w, h in detections:
        faces.append(
            (
                int(round(x * inverse)),
                int(round(y * inverse)),
                int(round(w * inverse)),
                int(round(h * inverse)),
            )
        )
    return faces


def draw_faces(
    image_bgr: np.ndarray,
    faces: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = BOX_COLOR_BGR,
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of the image with a green box and label around each face."""
    output = ensure_bgr(image_bgr).copy()
    for index, (x, y, w, h) in enumerate(faces, start=1):
        end_x = x + w
        end_y = y + h
        cv2.rectangle(output, (x, y), (end_x, end_y), color, thickness)

        label = f"Face {index}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        text_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )
        # Place the label above the box when there is room, otherwise inside it.
        if y - text_height - 10 >= 0:
            text_origin_y = y - 6
            band_top = y - text_height - 10
            band_bottom = y
        else:
            text_origin_y = min(output.shape[0] - 4, y + text_height + 8)
            band_top = y
            band_bottom = min(output.shape[0], text_origin_y + baseline + 4)

        band_right = min(output.shape[1], x + text_width + 8)
        cv2.rectangle(output, (x, band_top), (band_right, band_bottom), color, thickness=-1)
        cv2.putText(
            output,
            label,
            (x + 4, text_origin_y),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )
    return output


@dataclass(frozen=True)
class DetectionResult:
    """Original image, annotated image, and the boxes that were found."""

    image_bgr: np.ndarray
    annotated_bgr: np.ndarray
    faces: list[tuple[int, int, int, int]]

    @property
    def count(self) -> int:
        return len(self.faces)

    @property
    def message(self) -> str:
        return detection_message(self.count)


def analyze_image(
    image_bgr: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: int = 30,
) -> DetectionResult:
    """Detect faces and draw boxes without modifying the caller's array."""
    original = ensure_bgr(image_bgr)
    faces = detect_faces(
        original,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=min_size,
    )
    annotated = draw_faces(original, faces)
    return DetectionResult(image_bgr=original, annotated_bgr=annotated, faces=faces)


def open_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """Open a webcam. On Windows, retry with DirectShow if the default fails."""
    if camera_index < 0:
        raise FaceDetectionError("Camera index cannot be negative.")

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened() and sys.platform == "win32":
        capture.release()
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not capture.isOpened():
        capture.release()
        raise FaceDetectionError(
            f"Could not open webcam (camera index {camera_index}). "
            "Check that a camera is connected, that the index is correct, "
            "and that no other app is using it."
        )

    # A buffer of 1 frame keeps the preview closer to real time.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def read_frame(capture: cv2.VideoCapture) -> np.ndarray:
    """Read one BGR frame or raise if the camera stopped producing images."""
    ok, frame = capture.read()
    if not ok or frame is None:
        raise FaceDetectionError(
            "The webcam did not return a frame. It may be disconnected or busy."
        )
    return ensure_bgr(frame)


def run_webcam(
    camera_index: int = 0,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: int = 30,
    window_name: str = "Face Detection — press Q or Esc to quit",
) -> None:
    """Show live detection in a desktop window.

    Press Q, Esc, or close the window to release the camera and exit.
    This requires the full ``opencv-python`` package and a local display.
    The Streamlit app does not need that package for photo detection.
    """
    capture = open_camera(camera_index)
    try:
        while True:
            frame = read_frame(capture)
            faces = detect_faces(
                frame,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors,
                min_size=min_size,
            )
            annotated = draw_faces(frame, faces)
            count_label = detection_message(len(faces))
            cv2.putText(
                annotated,
                count_label,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                BOX_COLOR_BGR,
                2,
                cv2.LINE_AA,
            )
            try:
                cv2.imshow(window_name, annotated)
            except cv2.error as exc:
                raise FaceDetectionError(
                    "OpenCV cannot open a desktop window here. Run the Streamlit "
                    "app and use Live webcam, or install opencv-python (not the "
                    "headless build) on a computer with a display. "
                    f"Details: {exc}"
                ) from exc

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            # Closing the window with the mouse should also release the camera.
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        # Ctrl+C is a safe exit: the finally block still releases the camera.
        pass
    finally:
        capture.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def save_annotated_image(image_path: str | Path, output_path: str | Path, **settings) -> DetectionResult:
    """Detect faces in a file and write the annotated image."""
    result = analyze_image(load_image(image_path), **settings)
    destination = Path(output_path)
    if not cv2.imwrite(str(destination), result.annotated_bgr):
        raise FaceDetectionError(f"Could not save the result to {destination}.")
    return result


def main(argv: list[str] | None = None) -> int:
    """Command-line entry: annotate one image, or open the webcam window."""
    parser = argparse.ArgumentParser(
        description="Detect faces with OpenCV's Haar Cascade classifier."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Image file to process. If omitted, the webcam window opens.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="detected-faces.png",
        help="Where to save the annotated image (default: detected-faces.png).",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--scale-factor", type=float, default=1.1)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--min-size", type=int, default=30)
    args = parser.parse_args(argv)

    settings = {
        "scale_factor": args.scale_factor,
        "min_neighbors": args.min_neighbors,
        "min_size": args.min_size,
    }
    try:
        if args.image:
            result = save_annotated_image(args.image, args.output, **settings)
            print(result.message)
            print(f"Saved {args.output}")
            return 0
        run_webcam(camera_index=args.camera, **settings)
        return 0
    except FaceDetectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
