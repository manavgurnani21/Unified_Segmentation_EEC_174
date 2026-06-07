"""Generate NB78 — Option C smoke test for CLRKDNet CLRHead integration."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NB_PATH = REPO / 'yolop_vehicle_lane' / 'stage2' / 'notebooks' / 'stage2_notebook_78_optionC_smoke_test_clrnet_head.ipynb'

cells = []

# --- cell 0: markdown title ---
cells.append({
    'cell_type': 'markdown', 'metadata': {}, 'id': 'nb78-title',
    'source': (
        '# NB78 — Option C smoke test: drive CLRKDNet `CLRHead` from our pipeline\n'
        '\n'
        '**Goal**: confirm that CLRKDNet\'s actual `CLRHead` (imported from '
        '`external_repos/CLRKDNet-master/`) can be instantiated and forwarded '
        'using our backbone\'s 3-level feature pyramid. If this passes, '
        'Option A (execute the full Path-3 plan in RMT-PPAD\'s repo) or '
        'Option B (adapt the plan into our existing codebase) becomes viable.\n'
        '\n'
        '**Scope** (intentionally narrow):\n'
        '- Build a joint model with `lane_head.type=\'clrnet_official\'`.\n'
        '- Forward a random `(2, 3, 384, 640)` tensor.\n'
        '- Assert lane-head output shapes: `cls_logits` `(2, 192)`, '
        '`coord_pred` `(2, 192, 72, 2)`, `lane_param` `(2, 192, 4)`.\n'
        '- No training, no real data, no loss path. Eval-mode forward only.\n'
        '\n'
        '**Dependency risk**: CLRHead imports `mmcv.cnn.ConvModule` and '
        '`clrkd.ops.nms`. We monkey-patch the NMS C++ extension to a Python '
        'stub (we never invoke it in the smoke test). `mmcv` must be '
        'pip-installed; cell 2 handles that.\n'
        '\n'
        '**Decision tree based on outcome**:\n'
        '- Exit 0 → smoke passed → tell me which path to commit to '
        '(Option A vs B) and I\'ll write the next concrete plan.\n'
        '- Exit 1 → import failed → most likely missing `mmcv`. Cell 2 should '
        'have installed it; if it didn\'t, the install logs tell us why.\n'
        '- Exit 2 → shape assertion failed → my wrapper has a translation bug '
        'between CLRHead\'s 78-D format and our pipeline\'s dict.\n'
        '- Exit 3 → forward crashed → likely a tensor-shape mismatch between '
        'our backbone\'s output channels and the adapters I wired up.\n'
    ),
})

# --- cell 1: markdown run mode ---
cells.append({
    'cell_type': 'markdown', 'metadata': {}, 'id': 'nb78-runmode',
    'source': (
        '### Run mode\n'
        '1. Cell 2 mounts Drive, sets `REPO_ROOT`, and `pip install`s mmcv.\n'
        '2. Cell 3 runs the smoke test script and prints its full output.\n'
        '3. Cell 4 summarizes the exit code in plain English.\n'
        '\n'
        'Expected wall-clock: ~3-5 minutes (mostly the mmcv install).'
    ),
})

# --- cell 2: install + mount ---
cells.append({
    'cell_type': 'code', 'metadata': {}, 'id': 'nb78-mount-install',
    'execution_count': None, 'outputs': [],
    'source': (
        "import os, sys, subprocess, textwrap\n"
        "from google.colab import drive\n"
        "os.environ['PYTHONUNBUFFERED'] = '1'\n"
        "drive.mount('/content/drive')\n"
        "\n"
        "REPO_ROOT = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'\n"
        "if not os.path.isdir(REPO_ROOT):\n"
        "    raise FileNotFoundError(f'Missing project root: {REPO_ROOT}')\n"
        "os.chdir(REPO_ROOT)\n"
        "if REPO_ROOT not in sys.path:\n"
        "    sys.path.insert(0, REPO_ROOT)\n"
        "\n"
        "# Install mmcv. CLRHead uses mmcv.cnn.ConvModule. We try mmcv-lite\n"
        "# first (smaller, no CUDA ops needed for ConvModule) and fall back to\n"
        "# full mmcv if needed.\n"
        "try:\n"
        "    import mmcv  # noqa: F401\n"
        "    print('[ok] mmcv already installed')\n"
        "except ImportError:\n"
        "    print('Installing mmcv...')\n"
        "    rc = subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'mmcv-lite'])\n"
        "    if rc != 0:\n"
        "        print('mmcv-lite failed; trying full mmcv')\n"
        "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'mmcv'])\n"
        "    import mmcv  # noqa: F401\n"
        "    print('[ok] mmcv installed:', mmcv.__version__)\n"
        "\n"
        "# Also install scipy + opencv-python-headless if not present (already\n"
        "# on Colab usually, but doesn't hurt to confirm).\n"
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
        "                       'pyyaml', 'scipy', 'opencv-python-headless'])\n"
        "\n"
        "from stage2.scripts.notebook_utils import run_streaming\n"
        "LOG_DIR = '/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs'\n"
        "os.makedirs(LOG_DIR, exist_ok=True)\n"
        "print('repo:', REPO_ROOT)"
    ),
})

# --- cell 3: run smoke test ---
cells.append({
    'cell_type': 'code', 'metadata': {}, 'id': 'nb78-run-smoke',
    'execution_count': None, 'outputs': [],
    'source': (
        "from pathlib import Path\n"
        "import os, sys\n"
        "\n"
        "SCRIPT = 'stage2/scripts/smoke_test_clrnet_head.py'\n"
        "LOG_FILE = os.path.join(LOG_DIR, 'optionC_smoke_clrnet_head.log')\n"
        "\n"
        "# Use check=False so we capture the exit code rather than raising.\n"
        "import subprocess\n"
        "print('Running:', SCRIPT)\n"
        "print('Log:', LOG_FILE)\n"
        "with open(LOG_FILE, 'w', encoding='utf-8') as logf:\n"
        "    proc = subprocess.run(\n"
        "        [sys.executable, '-u', SCRIPT],\n"
        "        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True\n"
        "    )\n"
        "    logf.write(proc.stdout)\n"
        "print(proc.stdout)\n"
        "print(f'\\n[exit_code] {proc.returncode}')\n"
        "SMOKE_EXIT = proc.returncode"
    ),
})

# --- cell 4: summary ---
cells.append({
    'cell_type': 'code', 'metadata': {}, 'id': 'nb78-summary',
    'execution_count': None, 'outputs': [],
    'source': (
        "if SMOKE_EXIT == 0:\n"
        "    print('=' * 60)\n"
        "    print('PASS: CLRKDNet CLRHead can be driven from our backbone.')\n"
        "    print('Decision needed: tell me which to commit to:')\n"
        "    print('  Option A -- full Path-3 in external_repos/RMT-PPAD-main/')\n"
        "    print('  Option B -- adapt Path-3 into yolop_vehicle_lane/stage2/')\n"
        "    print('=' * 60)\n"
        "elif SMOKE_EXIT == 1:\n"
        "    print('=' * 60)\n"
        "    print('FAIL: import error. Likely fix:')\n"
        "    print('  - Confirm cell 2 finished `pip install mmcv` without error.')\n"
        "    print('  - Check the log above for the actual ImportError trace.')\n"
        "    print('=' * 60)\n"
        "elif SMOKE_EXIT == 2:\n"
        "    print('=' * 60)\n"
        "    print('FAIL: shape assertion. The wrapper translation between')\n"
        "    print('CLRHead 78-D format and our pipeline dict is wrong.')\n"
        "    print('Check `vendor_clrnet_head.VendorCLRNetHead.forward()`.')\n"
        "    print('=' * 60)\n"
        "elif SMOKE_EXIT == 3:\n"
        "    print('=' * 60)\n"
        "    print('FAIL: forward crashed. Likely fix:')\n"
        "    print('  - Backbone output channels do not match adapter `in_channels`.')\n"
        "    print('  - CLRHead expects a different number of feature levels.')\n"
        "    print('  - Check the log above for the full traceback.')\n"
        "    print('=' * 60)\n"
        "else:\n"
        "    print(f'Unexpected exit code: {SMOKE_EXIT}')"
    ),
})

# Wrap into notebook JSON
nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('wrote', NB_PATH)
