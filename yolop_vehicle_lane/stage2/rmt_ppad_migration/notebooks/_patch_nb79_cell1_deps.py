"""Patch NB79 cell 1 (the install cell) to also install mmcv. RMT-PPAD's
MTDETR imports mmcv.cnn.ConvModule indirectly via CLRKDNet-style modules,
and Colab's stock environment doesn't ship mmcv.
"""
import json
from pathlib import Path

p = Path(__file__).resolve().parent / 'stage2_notebook_79_P0_baseline.ipynb'
nb = json.loads(p.read_text(encoding='utf-8'))

# Cell index 2 is the install cell (markdown at index 1 introduces it).
new_install = (
    "import os, sys, subprocess\n"
    "from google.colab import drive\n"
    "os.environ['PYTHONUNBUFFERED'] = '1'\n"
    "drive.mount('/content/drive')\n"
    "\n"
    "REPO_ROOT = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'\n"
    "if not os.path.isdir(REPO_ROOT):\n"
    "    raise FileNotFoundError(f'Missing {REPO_ROOT} -- verify Drive sync.')\n"
    "os.chdir(REPO_ROOT)\n"
    "if REPO_ROOT not in sys.path:\n"
    "    sys.path.insert(0, REPO_ROOT)\n"
    "\n"
    "# Two pip installs because mmcv often pulls in build deps that conflict\n"
    "# with the lighter list if combined.\n"
    "#\n"
    "# 1) Ultralytics-style runtime deps (RMT-PPAD reuses these).\n"
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "    'pyyaml', 'tqdm', 'matplotlib', 'opencv-python-headless', 'scipy',\n"
    "    'pandas', 'seaborn', 'requests', 'thop', 'psutil', 'py-cpuinfo'])\n"
    "\n"
    "# 2) mmcv -- RMT-PPAD's MTDETR (and CLRKDNet's CLRHead) imports\n"
    "#    mmcv.cnn.ConvModule. Without this, the first `from ultralytics import\n"
    "#    MTDETR` crashes at module load with ModuleNotFoundError, which is what\n"
    "#    happened in NB79 v1 (subprocess exited rc=1 in 24 s during cell 5).\n"
    "try:\n"
    "    import mmcv  # noqa: F401\n"
    "    print(f'[ok] mmcv already installed: {mmcv.__version__}')\n"
    "except ImportError:\n"
    "    print('[install] mmcv (~1-3 min wheel build)...')\n"
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'mmcv'])\n"
    "    import mmcv  # noqa: F401\n"
    "    print(f'[ok] mmcv installed: {mmcv.__version__}')\n"
    "\n"
    "print('repo:', REPO_ROOT)\n"
    "\n"
    "from stage2.scripts.notebook_utils import run_streaming\n"
    "LOG_DIR = '/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs'\n"
    "os.makedirs(LOG_DIR, exist_ok=True)"
)

# Source as list-of-lines (canonical .ipynb).
parts = new_install.split('\n')
nb['cells'][2]['source'] = [p_ + '\n' for p_ in parts[:-1]] + ([parts[-1]] if parts[-1] else [])

# Clear stale outputs from the install cell + the runner cell (10).
nb['cells'][2]['outputs'] = []
nb['cells'][2]['execution_count'] = None
nb['cells'][10]['outputs'] = []
nb['cells'][10]['execution_count'] = None

p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('NB79 cell 1 now installs mmcv; cells 1 + 5 outputs cleared.')
