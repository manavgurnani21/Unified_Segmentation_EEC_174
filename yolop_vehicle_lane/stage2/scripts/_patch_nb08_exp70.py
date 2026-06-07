"""Append exp70 (NB77 CULane-KD joint) to NB08."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NB_PATH = REPO / 'yolop_vehicle_lane' / 'stage2' / 'notebooks' / 'stage2_notebook_08_joint_eval_visualization_and_profile.ipynb'

STEM = 'exp70_rmt_gca_anchor_cls_sep_vfl_culane_kd_full_data_joint'
RUN_TAG = 'full12'


def _eval_entry():
    cands = ''.join(
        f"            '/content/drive/MyDrive/EcoCAR/training_runs/{STEM}{('_' + s) if s else ''}.tar',\n"
        for s in ['full12', 'debug', '']
    )
    return (
        f"    {{\n        'config': 'stage2/configs/{STEM}.yaml',\n"
        f"        'candidates': [\n{cands}        ],\n    }},\n"
    )


def _to_text(s):
    return ''.join(s) if isinstance(s, list) else s


def _to_lines(text):
    if not text:
        return []
    parts = text.split('\n')
    out = [p + '\n' for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def patch_eval(text):
    if STEM in text:
        return text
    # Anchor on the last config entry we have.
    candidates = [
        "    {\n        'config': 'stage2/configs/exp67_rmt_gca_anchor_cls_sep_vfl_kd_from_nb62_full_data_joint.yaml',\n",
        "    {\n        'config': 'stage2/configs/exp66_rmt_gca_anchor_cls_sep_vfl_deep_roi_full_data_joint.yaml',\n",
        "    {\n        'config': 'stage2/configs/exp65_rmt_gca_anchor_cls_sep_vfl_hires_mask_full_data_joint.yaml',\n",
    ]
    anchor = next((c for c in candidates if c in text), None)
    if anchor is None:
        raise RuntimeError('No anchor in eval cell.')
    idx = text.find(anchor)
    end = text.find("    },\n", idx) + len("    },\n")
    return text[:end] + _eval_entry() + text[end:]


def patch_plot(text):
    metric = f"    '/content/drive/MyDrive/EcoCAR/training_runs/{STEM}_{RUN_TAG}_metrics.json',\n"
    if metric in text:
        return text
    candidates = [
        "    '/content/drive/MyDrive/EcoCAR/training_runs/exp67_rmt_gca_anchor_cls_sep_vfl_kd_from_nb62_full_data_joint_full12_metrics.json',\n",
        "    '/content/drive/MyDrive/EcoCAR/training_runs/exp66_rmt_gca_anchor_cls_sep_vfl_deep_roi_full_data_joint_full12_metrics.json',\n",
        "    '/content/drive/MyDrive/EcoCAR/training_runs/exp65_rmt_gca_anchor_cls_sep_vfl_hires_mask_full_data_joint_full12_metrics.json',\n",
    ]
    anchor = next((c for c in candidates if c in text), None)
    if anchor is None:
        raise RuntimeError('No plot anchor.')
    return text.replace(anchor, anchor + metric, 1)


def patch_video(text):
    video = (
        f"    ('/content/drive/MyDrive/EcoCAR/training_runs/{STEM}_{RUN_TAG}.tar',"
        f" 'stage2/configs/{STEM}.yaml', 'video_profile_exp70'),\n"
    )
    if video in text:
        return text
    anchor = "PROFILE_CANDIDATES = [\n"
    if anchor not in text:
        return text
    return text.replace(anchor, anchor + video, 1)


def main():
    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
    c3 = _to_text(nb['cells'][3]['source'])
    c5 = _to_text(nb['cells'][5]['source'])
    c7 = _to_text(nb['cells'][7]['source'])
    new3 = patch_eval(c3)
    new5 = patch_plot(c5)
    new7 = patch_video(c7)
    nb['cells'][3]['source'] = _to_lines(new3)
    nb['cells'][5]['source'] = _to_lines(new5)
    nb['cells'][7]['source'] = _to_lines(new7)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print('NB08 patched:', NB_PATH)
    print(' eval changed:', new3 != c3)
    print(' plot changed:', new5 != c5)
    print(' video changed:', new7 != c7)


if __name__ == '__main__':
    main()
