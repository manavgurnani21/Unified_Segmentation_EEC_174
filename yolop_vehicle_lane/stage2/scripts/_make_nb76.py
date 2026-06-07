"""Generate NB76 - download CLRKDNet checkpoints from GitHub."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NOTEBOOKS = REPO / 'yolop_vehicle_lane' / 'stage2' / 'notebooks'
TEMPLATE = NOTEBOOKS / 'stage2_notebook_40_exp2kk_anchor_asl_amp_joint.ipynb'

nb = json.loads(TEMPLATE.read_text(encoding='utf-8'))
for c in nb['cells']:
    if c.get('cell_type') == 'code':
        c['outputs'] = []
        c['execution_count'] = None

# Keep cells 0/1/2 layout (markdown/markdown/install). Repurpose cells 3-5.
nb['cells'][0]['source'] = (
    "# Stage 2 Notebook 76 - Download CLRKDNet GitHub checkpoints\n"
    "\n"
    "**Purpose.** NB72 self-distilled from NB62 (a student-quality model) and got nothing -- "
    "garbage teacher in, garbage student out. This notebook downloads CLRKDNet's published "
    "GitHub checkpoint, which scores 80.87 F1 on CULane -- a genuinely strong teacher.\n"
    "\n"
    "What we pull:\n"
    "- `dla34_clrnet_culane_8087.pth` (~80 MB): the DLA-34 CLRNet teacher used in the CLRKDNet paper.\n"
    "- The four training-log .txt files for reference and debugging.\n"
    "\n"
    "These go into `/content/drive/MyDrive/EcoCAR/downloads/clrkdnet_weights/` so they survive Colab "
    "session resets. NB77 (joint BDD training with CLRKDNet KD teacher) then consumes "
    "`dla34_clrnet_culane_8087.pth`.\n"
    "\n"
    "Note on architecture mismatch: CLRKDNet's backbone is DLA-34, our model uses RMT-GCA. We "
    "do NOT load CLRKDNet's backbone weights into our backbone (incompatible). Instead, we run "
    "CLRKDNet as a teacher in inference mode and distill its lane curve outputs into our model "
    "via FusionLaneLoss.w_distill. The head architectures differ (CLRNet 192 priors vs our 192-anchor "
    "CLRKDLaneHead -- happily very similar) but the OUTPUT format is the same (lane curve "
    "coordinates + cls scores), so distillation is well-defined."
)

nb['cells'][1]['source'] = (
    "### Run mode\n"
    "1. Run cell 2 first to mount Drive and install dependencies.\n"
    "2. Run cell 3 to download. ~80 MB over network -- 2-5 minutes typical.\n"
    "3. Cell 4 verifies the checkpoint loads in PyTorch and prints its key structure.\n"
    "4. NB77 will reference `/content/drive/MyDrive/EcoCAR/downloads/clrkdnet_weights/dla34_clrnet_culane_8087.pth`."
)

# Cell 2 (already has the install block from template) -- keep as-is but rename to NB76 context.

nb['cells'][3]['source'] = (
    "# Step 1: download CLRKDNet's GitHub release checkpoints.\n"
    "from pathlib import Path\n"
    "import os, sys\n"
    "\n"
    "DEST = '/content/drive/MyDrive/EcoCAR/downloads/clrkdnet_weights'\n"
    "Path(DEST).mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "cmd = [sys.executable, '-u', 'stage2/scripts/download_clrkdnet_weights.py',\n"
    "       '--dest', DEST]\n"
    "LOG_FILE = os.path.join(LOG_DIR, 'clrkdnet_download.log')\n"
    "print('Downloading CLRKDNet GitHub checkpoints to', DEST)\n"
    "run_streaming(cmd, log_path=LOG_FILE)"
)

nb['cells'][4]['source'] = (
    "# Step 2: verify the checkpoint structure so NB77 can load it cleanly.\n"
    "import torch\n"
    "from pathlib import Path\n"
    "\n"
    "DEST = '/content/drive/MyDrive/EcoCAR/downloads/clrkdnet_weights'\n"
    "weight_path = Path(DEST) / 'dla34_clrnet_culane_8087.pth'\n"
    "if not weight_path.exists():\n"
    "    raise FileNotFoundError(f'Cell 3 did not produce {weight_path}; re-run it.')\n"
    "\n"
    "sd = torch.load(weight_path, map_location='cpu', weights_only=False)\n"
    "if isinstance(sd, dict) and 'state_dict' in sd:\n"
    "    sd = sd['state_dict']\n"
    "elif isinstance(sd, dict) and 'model' in sd:\n"
    "    sd = sd['model']\n"
    "if hasattr(sd, 'state_dict'):\n"
    "    sd = sd.state_dict()\n"
    "print(f'top-level type: {type(sd).__name__}')\n"
    "print(f'num parameters: {len(sd) if hasattr(sd, \"__len__\") else \"<unknown>\"}')\n"
    "print(f'file size: {weight_path.stat().st_size/1e6:.1f} MB')\n"
    "\n"
    "# Print sample parameter names by prefix family -- helps NB77 build the adapter.\n"
    "if isinstance(sd, dict):\n"
    "    families = {}\n"
    "    for k in sd.keys():\n"
    "        prefix = k.split('.')[0]\n"
    "        families.setdefault(prefix, []).append(k)\n"
    "    print('\\nparameter families (prefix -> count, first key):')\n"
    "    for prefix, keys in sorted(families.items(), key=lambda x: -len(x[1])):\n"
    "        print(f'  {prefix:25s}  {len(keys):4d} params  e.g. {keys[0]}')"
)

nb['cells'][5]['source'] = (
    "## Notes on next steps\n"
    "\n"
    "After this notebook runs successfully:\n"
    "- `/content/drive/MyDrive/EcoCAR/downloads/clrkdnet_weights/dla34_clrnet_culane_8087.pth` exists\n"
    "- The verification print confirms it's a CLRNet checkpoint (backbone=DLA-34, heads=CLRNet's "
    "192-prior style).\n"
    "\n"
    "NB77 will:\n"
    "1. Load the teacher in inference mode (no gradients) on a separate GPU stream.\n"
    "2. For each training image, run the teacher to get teacher_cls_logits and teacher_coord_pred.\n"
    "3. Pass them to FusionLaneLoss via the existing `teacher=` argument. `w_distill > 0` triggers "
    "the MSE distillation loss between student and teacher outputs.\n"
    "4. Train the student (our RMT-GCA + CLRKDLaneHead) jointly with BDD det.\n"
    "\n"
    "Caveat: the teacher was trained on CULane (rural Chinese highway scenes), we're training the "
    "student on BDD100K (urban US driving). Distribution shift may limit the teacher's signal, "
    "but it should still pull our lane curves toward better geometry."
)

out = NOTEBOOKS / 'stage2_notebook_76_download_clrkdnet_weights.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('wrote', out)
