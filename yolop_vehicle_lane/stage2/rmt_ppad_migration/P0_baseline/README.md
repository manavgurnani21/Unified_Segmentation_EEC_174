# P0 — Baseline reproduction

**Status: in_progress** (Task #1)

## What this phase produces

`yolop_vehicle_lane/stage2/rmt_ppad_migration/results/baseline_metrics.json`
containing the metrics that RMT-PPAD's published `best.pt` achieves on BDD100K val,
plus the full stdout log at `baseline_metrics.log` next to it.

## What runs

The execution lives in `notebooks/stage2_notebook_79_P0_baseline.ipynb`. Cells map to:

| Cell | Purpose | Time |
|---|---|---|
| 1 | Mount Drive + install RMT-PPAD's pip deps | 30 s |
| 2 | Vendor RMT-PPAD source (writable copy of `external_repos/RMT-PPAD-main/ultralytics/` into `vendor/RMT-PPAD/`) | 1-3 min |
| 3 | Download SharePoint files INTO DRIVE: `rmt_ppad_best.pt` + `BDD_detection_labels.zip` + `BDD_seg_masks.zip` | 2-10 min |
| 4 | Extract zips INTO `/content/BDD_seg_mask/` (LOCAL, not Drive — Drive extraction of thousands of small files is brutally slow). Symlink BDD images from your existing source. | 1-3 min |
| 5 | Run `MTDETR.val(...)` against the LOCAL `BDD_ROOT` via `run_baseline_val.py` | 10-30 min |
| 6 | Parse + acceptance-test the metrics | <10 s |

**Drive vs local I/O rule (project-wide):** Drive write API throttles per-file, not per-byte. A single 1 GB blob extracts to Drive in ~30 s; the same data as 70 000 small JPGs takes 2-3 hours and frequently fails. Always extract to `/content/` and only persist the OUTPUT (.pt checkpoints, .json metrics, .log files) back to Drive.

## Scripts in this folder

- `download_rmt_ppad_pretrained.py` — SharePoint downloader. Tries `urllib`, `curl`, then `wget`. Has size-threshold checks to detect HTML-preview returns. Prints manual-download URL on failure.
- `vendor_rmt_ppad.py` — shallow copy of RMT-PPAD source (skips `build/`, `*.egg-info`, `docs/`, etc.). Idempotent.
- `run_baseline_val.py` — patches `BDD_full.yaml` with the local BDD root, runs `MTDETR.val(...)` as a subprocess, parses stdout into a metrics dict, writes JSON + full log.

## Acceptance criterion

From the source plan:
```
mAP50 ≈ 0.849   drivable mIoU ≈ 0.926   lane IoU ≈ 0.568   lane ACC ≈ 0.847
```
Cell 6's acceptance test accepts ±0.02 absolute on each. If any check fails or returns "not parsed", investigate the log before proceeding to P1.

## Known issues to be ready for

1. **SharePoint anonymous-link policy** sometimes rejects automated downloads. The downloader prints the browser-fallback URL when this happens; download the file manually, drop it in `/content/drive/MyDrive/EcoCAR/downloads/rmt_ppad_weights/`, then re-run cell 3.
2. **BDD images path mismatch.** RMT-PPAD's `BDD_full.yaml` expects `images/train2017` / `images/val2017` (despite being BDD100K, not COCO). If your existing BDD images are under different subdirs, symlink them or rename.
3. **No GPU on the Colab session** — `MTDETR.val(...)` requires CUDA; will fail with a clear error if Colab gave a CPU-only runtime.
4. **mmcv version conflict.** RMT-PPAD's `pyproject.toml` doesn't pin mmcv; their training requires `mmcv-full==1.3.x` (CULane vintage). If a later mmcv is already installed, validation may still work; if not, `pip install mmcv==2.2.0` then retry.

## Next phase

When P0 passes (PASS in cell 6), update Task #1 to `completed` and Task #2 (P1) becomes unblocked.
