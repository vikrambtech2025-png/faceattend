"""CLI entrypoint."""

from __future__ import annotations

import argparse

from .logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="faceattend")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("serve", help="run api server")
    args = parser.parse_args()
    configure_logging()
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("faceattend.api.main:app", host="0.0.0.0", port=8000)
    else:
        print("use: faceattend serve")


if __name__ == "__main__":
    main()
