# ADR 0001: Use insightface (buffalo_l) for detection + recognition

## Status
Accepted

## Context
Need accurate, installable face detection + recognition on Windows/CPU with anti-spoof extensibility.

## Decision
Use insightface `buffalo_l` (RetinaFace + ArcFace) via ONNX runtime.

## Consequences
- Single package provides detection + embedding + alignment.
- Non-commercial license: document alternative for commercial use.
- Model ~300MB download; cache in Docker layer.