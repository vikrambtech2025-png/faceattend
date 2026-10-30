"""CLI enrollment from an image or folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from faceattend.config import get_settings
from faceattend.gallery import GalleryManager
from faceattend.logging import configure_logging
from faceattend.storage import make_session_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--images", required=True, help="image file or directory")
    parser.add_argument("--employee-id", default=None)
    parser.add_argument("--consent", action="store_true", required=True)
    parser.add_argument("--pose-tag", default=None)
    args = parser.parse_args()

    configure_logging()
    sf = make_session_factory(str(get_settings().db_path))
    gm = GalleryManager(sf)

    p = Path(args.images)
    files = [p] if p.is_file() else sorted(p.glob("*.*"))
    ids = set()
    for f in files:
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        try:
            pid = gm.enroll_from_image(
                name=args.name,
                image_path=f,
                employee_id=args.employee_id,
                consent=args.consent,
                pose_tag=args.pose_tag,
            )
            ids.add(pid)
            print(f"enrolled {f.name} -> {pid}")
        except Exception as e:
            print(f"skip {f.name}: {e}")
    print("done, person_ids:", ids)


if __name__ == "__main__":
    main()
