"""Generate NB79 — Phase P0 baseline reproduction notebook."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
NB_DIR = REPO / 'yolop_vehicle_lane' / 'stage2' / 'rmt_ppad_migration' / 'notebooks'
NB_DIR.mkdir(parents=True, exist_ok=True)

nb = {
    'cells': [
        # Cell 0: title
        {'cell_type': 'markdown', 'metadata': {}, 'source': (
            "# RMT-PPAD migration NB79 - Phase P0 (baseline reproduction)\n"
            "\n"
            "**Purpose.** Reproduce the published RMT-PPAD numbers on BDD100K so we have a\n"
            "fixed reference target. The full migration plan (P0-P8) is in\n"
            "`yolop_vehicle_lane/stage2/rmt_ppad_migration/README.md`.\n"
            "\n"
            "**Acceptance criterion (from the source plan):** the printed numbers should be\n"
            "approximately:\n"
            "- Detection: Recall ~= 0.954, mAP50 ~= 0.849\n"
            "- Drivable area mIoU ~= 0.926 (this will be removed in P4+)\n"
            "- Lane line IoU ~= 0.568, ACC ~= 0.847\n"
            "\n"
            "**Output:** `yolop_vehicle_lane/stage2/rmt_ppad_migration/results/baseline_metrics.json`\n"
            "\n"
            "**Time:** ~30-90 min wall clock (mostly downloads + 5000-image val pass)."
        )},
        # Cell 1: setup
        {'cell_type': 'markdown', 'metadata': {}, 'source': (
            "### Cell 1: Mount Drive, install RMT-PPAD deps, set REPO_ROOT\n"
            "RMT-PPAD's environment.yml requires Python 3.8 + PyTorch 2.4. Colab usually\n"
            "ships PyTorch 2.x already, so we only need to pip-install their extras."
        )},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': (
            "import os, sys, subprocess\n"
            "from google.colab import drive\n"
            "os.environ['PYTHONUNBUFFERED'] = '1'\n"
            "drive.mount('/content/drive')\n"
            "\n"
            "REPO_ROOT = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'\n"
            "if not os.path.isdir(REPO_ROOT):\n"
            "    raise FileNotFoundError(f'Missing {REPO_ROOT} — verify Drive sync.')\n"
            "os.chdir(REPO_ROOT)\n"
            "if REPO_ROOT not in sys.path:\n"
            "    sys.path.insert(0, REPO_ROOT)\n"
            "\n"
            "# Ultralytics + its deps. RMT-PPAD ships its own ultralytics fork, so we install\n"
            "# the runtime requirements but NOT the upstream ultralytics package (would conflict).\n"
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
            "    'pyyaml', 'tqdm', 'matplotlib', 'opencv-python-headless', 'scipy',\n"
            "    'pandas', 'seaborn', 'requests', 'thop', 'psutil', 'py-cpuinfo'])\n"
            "print('repo:', REPO_ROOT)\n"
            "\n"
            "from stage2.scripts.notebook_utils import run_streaming\n"
            "LOG_DIR = '/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs'\n"
            "os.makedirs(LOG_DIR, exist_ok=True)"
        )},
        # Cell 2: vendor source
        {'cell_type': 'markdown', 'metadata': {}, 'source': (
            "### Cell 2: Vendor RMT-PPAD source (shallow copy)\n"
            "Copies `external_repos/RMT-PPAD-main/ultralytics/` to a writable working folder\n"
            "under `stage2/rmt_ppad_migration/vendor/RMT-PPAD/`. P2-P8 will modify files here;\n"
            "the original `external_repos/` copy stays untouched."
        )},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': (
            "import sys, os\n"
            "\n"
            "RMT_PPAD_SRC = '/content/drive/MyDrive/EcoCAR/external_repos/RMT-PPAD-main'\n"
            "RMT_PPAD_DST = ('/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/'\n"
            "                'stage2/rmt_ppad_migration/vendor/RMT-PPAD')\n"
            "\n"
            "if not os.path.isdir(RMT_PPAD_SRC):\n"
            "    raise FileNotFoundError(f'{RMT_PPAD_SRC} not present. Did you clone the repo into Drive?')\n"
            "\n"
            "cmd = [sys.executable, '-u',\n"
            "       'stage2/rmt_ppad_migration/P0_baseline/vendor_rmt_ppad.py',\n"
            "       '--src', RMT_PPAD_SRC, '--dst', RMT_PPAD_DST]\n"
            "log = os.path.join(LOG_DIR, 'NB79_vendor.log')\n"
            "run_streaming(cmd, log_path=log)"
        )},
        # Cell 3: download
        {'cell_type': 'markdown', 'metadata': {}, 'source': (
            "### Cell 3: Download RMT-PPAD pretrained weights + BDD detection labels + masks\n"
            "Three SharePoint links from RMT-PPAD's README. The downloader appends\n"
            "`?download=1` to force binary download instead of HTML preview.\n"
            "\n"
            "If any of the three fails automatically, the script prints a manual-fallback URL\n"
            "you can open in a browser; save the file to `DOWNLOADS_DIR` and re-run this cell."
        )},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': (
            "import sys, os\n"
            "\n"
            "DOWNLOADS_DIR = '/content/drive/MyDrive/EcoCAR/downloads/rmt_ppad_weights'\n"
            "os.makedirs(DOWNLOADS_DIR, exist_ok=True)\n"
            "\n"
            "cmd = [sys.executable, '-u',\n"
            "       'stage2/rmt_ppad_migration/P0_baseline/download_rmt_ppad_pretrained.py',\n"
            "       '--dest', DOWNLOADS_DIR]\n"
            "log = os.path.join(LOG_DIR, 'NB79_download.log')\n"
            "run_streaming(cmd, log_path=log)"
        )},
        # Cell 4: BDD data layout check
        {'cell_type': 'markdown', 'metadata': {}, 'source': (
            "### Cell 4: Verify BDD images + extract RMT-PPAD's labels/masks\n"
            "RMT-PPAD's `BDD_full.yaml` expects this exact layout:\n"
            "```\n"
            "BDD_seg_mask/\n"
            "  images/{train2017,val2017}/*.jpg\n"
            "  labels/{train2017,val2017}/*.txt   (YOLO format: class cx cy w h)\n"
            "  mask/lane/{train2017,val2017}/*.png\n"
            "  mask/drivable/{train2017,val2017}/*.png\n"
            "```\n"
            "If `BDD_ROOT` doesn't have this exact layout we lay it out from the downloaded\n"
            "zip files. Note: this expects BDD100K's raw images are already on Drive (from\n"
            "your earlier NB00 / NB04 work). If not, download them from bdd-data.berkeley.edu.\n"
        )},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': (
            "import os, zipfile, subprocess\n"
            "from pathlib import Path\n"
            "\n"
            "BDD_ROOT = '/content/drive/MyDrive/EcoCAR/datasets/BDD_seg_mask'\n"
            "Path(BDD_ROOT).mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "# Inspect current layout.\n"
            "print(f'[inspect] {BDD_ROOT}')\n"
            "for sub in ('images/train2017', 'images/val2017',\n"
            "            'labels/train2017', 'labels/val2017',\n"
            "            'mask/lane/train2017', 'mask/lane/val2017',\n"
            "            'mask/drivable/train2017', 'mask/drivable/val2017'):\n"
            "    p = Path(BDD_ROOT) / sub\n"
            "    if p.exists():\n"
            "        n = sum(1 for _ in p.iterdir())\n"
            "        print(f'  OK  {sub:34s} ({n} files)')\n"
            "    else:\n"
            "        print(f'  --  {sub:34s} (missing)')\n"
            "\n"
            "# Extract RMT-PPAD's labels + masks if their zips were downloaded.\n"
            "DOWNLOADS_DIR = '/content/drive/MyDrive/EcoCAR/downloads/rmt_ppad_weights'\n"
            "for zip_name in ('BDD_detection_labels.zip', 'BDD_seg_masks.zip'):\n"
            "    zp = Path(DOWNLOADS_DIR) / zip_name\n"
            "    if not zp.exists():\n"
            "        print(f'[skip] {zp.name} not present (Cell 3 must succeed first).')\n"
            "        continue\n"
            "    print(f'[extract] {zp} -> {BDD_ROOT}')\n"
            "    with zipfile.ZipFile(zp) as zf:\n"
            "        zf.extractall(BDD_ROOT)\n"
            "\n"
            "print('\\n[re-inspect]')\n"
            "for sub in ('images/train2017', 'images/val2017',\n"
            "            'labels/train2017', 'labels/val2017',\n"
            "            'mask/lane/train2017', 'mask/lane/val2017',\n"
            "            'mask/drivable/train2017', 'mask/drivable/val2017'):\n"
            "    p = Path(BDD_ROOT) / sub\n"
            "    if p.exists():\n"
            "        n = sum(1 for _ in p.iterdir())\n"
            "        print(f'  OK  {sub:34s} ({n} files)')\n"
            "    else:\n"
            "        print(f'  MISSING  {sub}')"
        )},
        # Cell 5: run validation
        {'cell_type': 'markdown', 'metadata': {}, 'source': (
            "### Cell 5: Run RMT-PPAD's MTDETR.val() on the BDD val split\n"
            "Patches `BDD_full.yaml` to point at our actual BDD root, then invokes\n"
            "MTDETR.val(...) via the runner script. Wall-clock ~10-30 minutes on a single\n"
            "GPU with the full 5000-image val set."
        )},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': (
            "import os, sys\n"
            "\n"
            "BDD_ROOT = '/content/drive/MyDrive/EcoCAR/datasets/BDD_seg_mask'\n"
            "RMT_PPAD_DST = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage2/rmt_ppad_migration/vendor/RMT-PPAD'\n"
            "CHECKPOINT = '/content/drive/MyDrive/EcoCAR/downloads/rmt_ppad_weights/rmt_ppad_best.pt'\n"
            "OUTPUT_JSON = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage2/rmt_ppad_migration/results/baseline_metrics.json'\n"
            "\n"
            "cmd = [sys.executable, '-u',\n"
            "       'stage2/rmt_ppad_migration/P0_baseline/run_baseline_val.py',\n"
            "       '--rmt-ppad-root', RMT_PPAD_DST,\n"
            "       '--checkpoint', CHECKPOINT,\n"
            "       '--bdd-root', BDD_ROOT,\n"
            "       '--output-json', OUTPUT_JSON,\n"
            "       '--batch', '1',\n"
            "       '--imgsz', '640',\n"
            "       '--mask-thr', '0.45,0.9']\n"
            "log = os.path.join(LOG_DIR, 'NB79_baseline_val.log')\n"
            "run_streaming(cmd, log_path=log)"
        )},
        # Cell 6: parse and report
        {'cell_type': 'markdown', 'metadata': {}, 'source': (
            "### Cell 6: Inspect parsed metrics\n"
            "If the runner parsed numbers from stdout they'll be in `baseline_metrics.json`.\n"
            "If parsing failed, the runner still wrote the full log; eyeball it manually."
        )},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': (
            "import json\n"
            "from pathlib import Path\n"
            "\n"
            "OUTPUT_JSON = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage2/rmt_ppad_migration/results/baseline_metrics.json'\n"
            "if not Path(OUTPUT_JSON).exists():\n"
            "    raise FileNotFoundError('baseline_metrics.json not produced; Cell 5 must complete first.')\n"
            "\n"
            "rec = json.loads(Path(OUTPUT_JSON).read_text(encoding='utf-8'))\n"
            "print(json.dumps(rec, indent=2))\n"
            "\n"
            "print('\\n=== Reference numbers from RMT-PPAD paper ===')\n"
            "print('  Detection Recall = 0.954, mAP50 = 0.849')\n"
            "print('  Drivable mIoU    = 0.926')\n"
            "print('  Lane IoU         = 0.568')\n"
            "print('  Lane ACC         = 0.847')\n"
            "\n"
            "# P0 acceptance: numbers within +/- 2 absolute points of reference.\n"
            "metrics = rec.get('metrics', {}) or {}\n"
            "checks = {\n"
            "    'detection_map50': (0.849, 0.02),\n"
            "    'drivable_miou':   (0.926, 0.02),\n"
            "    'lane_iou':        (0.568, 0.02),\n"
            "    'lane_acc':        (0.847, 0.02),\n"
            "}\n"
            "print('\\n[acceptance test]')\n"
            "fails = 0\n"
            "for key, (ref, tol) in checks.items():\n"
            "    val = metrics.get(key)\n"
            "    if val is None:\n"
            "        print(f'  ??  {key}: not parsed (review log)')\n"
            "        fails += 1\n"
            "    elif abs(val - ref) <= tol:\n"
            "        print(f'  OK  {key}: {val:.3f}  (reference {ref:.3f})')\n"
            "    else:\n"
            "        print(f'  X   {key}: {val:.3f}  (reference {ref:.3f}, tol {tol})')\n"
            "        fails += 1\n"
            "print('\\n[P0 result]', 'PASS' if fails == 0 else f'FAIL ({fails} checks not OK)')"
        )},
    ],
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

# Convert source strings to list-of-lines for canonical .ipynb format.
for c in nb['cells']:
    s = c['source']
    if isinstance(s, str):
        parts = s.split('\n')
        c['source'] = [p + '\n' for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])

out = NB_DIR / 'stage2_notebook_79_P0_baseline.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('wrote', out)
