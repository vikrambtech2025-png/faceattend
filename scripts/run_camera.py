"""Headless camera loop with OpenCV window."""

from __future__ import annotations

import argparse

import cv2

from faceattend.alerts import AlertManager
from faceattend.camera import CameraSource
from faceattend.config import get_settings
from faceattend.logging import configure_logging
from faceattend.pipeline import Pipeline
from faceattend.storage import make_session_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--source", default=None, help="video file or rtsp url")
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    if args.camera is not None:
        settings = settings  # keep
    sf = make_session_factory(str(settings.db_path))
    pipeline = Pipeline(sf, camera_id="default")
    am = AlertManager(sf)
    pipeline.set_alert_manager(am)

    src = (
        args.source
        if args.source
        else (args.camera if args.camera is not None else settings.camera_index)
    )
    cam = CameraSource(src)
    cam.start()

    print("Press 'q' to quit, 'r' to reload gallery.")
    try:
        while True:
            pkt = cam.read()
            if not pkt.ok or pkt.frame is None:
                continue
            result = pipeline.process(pkt.frame)
            frame = pkt.frame.copy()
            for pf in result.faces:
                b = pf.bbox
                color = (0, 255, 0) if not pf.is_unknown else (0, 0, 255)
                cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), color, 2)
                label = "Unknown" if pf.is_unknown else f"{pf.name} {1-pf.distance:.2f}"
                if pf.liveness is not None:
                    label += f" live:{pf.liveness.is_live}"
                cv2.putText(
                    frame, label, (b.x1, max(0, b.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
            cv2.putText(
                frame,
                f"FPS {result.fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            cv2.imshow("FaceAttend", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                pipeline.reload_gallery()
                print("gallery reloaded, size=", pipeline.gallery.size)
    finally:
        cam.release()
        pipeline.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
