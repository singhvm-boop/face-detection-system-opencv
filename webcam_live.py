"""Open a desktop window and detect faces from the webcam.

Press Q, Esc, or close the window to stop and release the camera.
For the upload-and-download interface, run ``streamlit run app.py`` instead.
"""

from __future__ import annotations

import sys

from face_detector import FaceDetectionError, run_webcam


def main() -> int:
    try:
        run_webcam()
    except FaceDetectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
