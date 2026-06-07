"""Update NB74 cell 3 to use the new auto-repair prep + add explanatory text.
Clears NB74 cells 3 & 4 of stale outputs so user can re-run cleanly.
"""
import json
from pathlib import Path

P = Path(__file__).resolve().parents[3] / 'yolop_vehicle_lane' / 'stage2' / 'notebooks' / 'stage2_notebook_74_culane_lane_only_pretrain.ipynb'
nb = json.loads(P.read_text(encoding='utf-8'))

# Cell 3: add --force option and integrity-aware behavior explanation.
nb['cells'][3]['source'] = (
    "# Step 1: extract CULane archives.\n"
    "#\n"
    "# v3 (after the first run crashed with FileNotFoundError on missing .jpgs):\n"
    "# the prep script now AUTO-REPAIRS partial extractions. For each driver\n"
    "# archive, it counts .jpg files actually on disk and compares against\n"
    "# CULane's published per-driver counts. If a marker dir has < 90 percent\n"
    "# of expected jpgs, it re-extracts on top (filling gaps). The post-run\n"
    "# integrity report prints the ratio so you can see which drivers were\n"
    "# repaired.\n"
    "#\n"
    "# If you want to FORCE a clean re-extract of everything (rare; useful\n"
    "# when prior extractions corrupted file metadata), set FORCE=True below.\n"
    "from pathlib import Path\n"
    "import os, sys, subprocess\n"
    "\n"
    "CULANE_SRC = '/content/drive/MyDrive/EcoCAR/downloads/CULane'\n"
    "CULANE_DST = '/content/CULane'\n"
    "FORCE = False  # set True only if 90%-threshold logic doesn't fix the gaps\n"
    "\n"
    "print(f'[diagnostic] listing {CULANE_SRC} ...')\n"
    "try:\n"
    "    for f in sorted(Path(CULANE_SRC).iterdir()):\n"
    "        try:\n"
    "            print(f'  {f.name}  ({f.stat().st_size/1e9:.2f} GB)')\n"
    "        except OSError as e:\n"
    "            print(f'  {f.name}  (stat err: {e})')\n"
    "except Exception as e:\n"
    "    print(f'[diagnostic] cannot list {CULANE_SRC}: {e}')\n"
    "    parent = '/content/drive/MyDrive/EcoCAR/downloads'\n"
    "    try:\n"
    "        print(f'[diagnostic] parent contents of {parent}:')\n"
    "        for f in sorted(Path(parent).iterdir()):\n"
    "            print(f'  {f.name}')\n"
    "    except Exception as e2:\n"
    "        print(f'[diagnostic] cannot list parent: {e2}')\n"
    "\n"
    "cmd = [sys.executable, '-u', 'stage2/scripts/prepare_culane_dataset.py',\n"
    "       '--src', CULANE_SRC,\n"
    "       '--dest', CULANE_DST]\n"
    "if FORCE:\n"
    "    cmd.append('--force')\n"
    "LOG_FILE = os.path.join(LOG_DIR, 'culane_prepare.log')\n"
    "print('Extracting CULane archives...  (force=' + str(FORCE) + ')')\n"
    "run_streaming(cmd, log_path=LOG_FILE)"
)
nb['cells'][3]['outputs'] = []
nb['cells'][3]['execution_count'] = None

# Cell 4: clear outputs from the failed run; no other code change needed
# (the train script's dataset already filters missing images upfront after
# my fix in train_joint_model_experiment.py).
nb['cells'][4]['outputs'] = []
nb['cells'][4]['execution_count'] = None

# Cell 5: update guidance with the integrity report explanation.
nb['cells'][5]['source'] = (
    "## What to watch in NB74 (v3 auto-repair flow)\n"
    "\n"
    "After cell 3 finishes you should see an `[integrity] per-driver jpg counts` block "
    "near the end of its output. Every driver should show `[OK]`. If any show `[PARTIAL]`, "
    "re-run cell 3 (the script will re-extract those drivers only). If `[MISSING]`, "
    "verify the corresponding tar.gz is in your Drive folder.\n"
    "\n"
    "Cell 4 trains the lane head on CULane. With v3 of the training script (dataset "
    "filter), it tolerates a small number of missing .jpg files: the dataset prints "
    "`dropped N / M samples because their .jpg files do not exist on disk` at init time "
    "and continues with the surviving N. If N drops below say 80k (out of 88880 total), "
    "the prep step had bigger issues -- re-run cell 3 with FORCE=True.\n"
    "\n"
    "Pass criteria at epoch 8 (CULane lane-only):\n"
    "- `val/matched_line_iou >= 0.55` (CULane is cleaner than BDD; should converge fast).\n"
    "- `val/lane_best_f1 >= 0.50` -- CULane has simple highway lanes; this is achievable.\n"
    "- `val/lane/decoded_f1 >= 0.30`.\n"
    "- A `best.pt` and `backbone_pretrained.pt` get saved into the output tar so NB77 "
    "can use them as the teacher / backbone init."
)

P.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('NB74 updated; cells 3+4 cleared.')
