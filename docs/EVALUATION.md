# Evaluation

## Dataset
- Own set: 20 identities, 5 images each, varied pose/lighting.
- LFW subset: 100 pairs.

## Metrics (to be measured)
| Metric | Value |
|---|---|
| Top-1 accuracy | TBD |
| TAR@FAR=1e-2 | TBD |
| TAR@FAR=1e-3 | TBD |
| Spoof HTER | TBD |
| Spoof EER | TBD |
| Detection+embed p50 (ms) | TBD |
| Detection+embed p95 (ms) | TBD |
| FPS (CPU, 640px) | TBD |

## Ablations
- threshold sweep (FAR/FRR curve)
- liveness on/off
- det_size 320 vs 640