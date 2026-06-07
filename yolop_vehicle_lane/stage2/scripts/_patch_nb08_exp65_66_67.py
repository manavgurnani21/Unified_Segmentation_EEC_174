"""Append exp65/66/67 entries to NB08."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NB_PATH = REPO / 'yolop_vehicle_lane' / 'stage2' / 'notebooks' / 'stage2_notebook_08_joint_eval_visualization_and_profile.ipynb'

EVAL_ENTRIES = [
    ('exp65_rmt_gca_anchor_cls_sep_vfl_hires_mask_full_data_joint', ['full12', 'debug', '']),
    ('exp66_rmt_gca_anchor_cls_sep_vfl_deep_roi_full_data_joint', ['full12', 'debug', '']),
    ('exp67_rmt_gca_anchor_cls_sep_vfl_kd_from_nb62_full_data_joint', ['full12', 'debug', '']),
]
PLOT_TAG = {k: 'full12' for k in [e[0] for e in EVAL_ENTRIES]}


def _eval_entry(stem, suffixes):
    cands = ''.join(
        f"            '/content/drive/MyDrive/EcoCAR/training_runs/{stem}{('_' + s) if s else ''}.tar',\n"
        for s in suffixes
    )
    return (
        f"    {{\n        'config': 'stage2/configs/{stem}.yaml',\n        'candidates': [\n{cands}        ],\n    }},\n"
    )


def _plot_entry(stem, run_tag):
    return f"    '/content/drive/MyDrive/EcoCAR/training_runs/{stem}_{run_tag}_metrics.json',\n"


def _video_entry(stem, run_tag):
    out_tag = stem.split('_', 1)[0]
    return (
        f"    ('/content/drive/MyDrive/EcoCAR/training_runs/{stem}_{run_tag}.tar',"
        f" 'stage2/configs/{stem}.yaml', 'video_profile_{out_tag}'),\n"
    )


def _to_text(s):
    return ''.join(s) if isinstance(s, list) else s


def _to_lines(text):
    if not text:
        return []
    parts = text.split('\n')
    out = [p + '\n' for p in parts[:-1]]
    if parts[-1] != '':
        out.append(parts[-1])
    return out


def patch_eval(text):
    candidates = [
        "    {\n        'config': 'stage2/configs/exp64_rmt_gca_anchor_cls_sep_vfl_long_warmup_full_data_joint.yaml',\n",
        "    {\n        'config': 'stage2/configs/exp63_rmt_gca_anchor_cls_sep_topk4_vfl_full_data_joint.yaml',\n",
        "    {\n        'config': 'stage2/configs/exp62_rmt_gca_anchor_cls_sep_vfl_iou_match_long14_joint.yaml',\n",
        "    {\n        'config': 'stage2/configs/exp57_rmt_gca_anchor_cls_sep_vfl_full_data_long12_joint.yaml',\n",
    ]
    anchor = next((c for c in candidates if c in text), None)
    if anchor is None:
        raise RuntimeError('No anchor in eval cell.')
    idx = text.find(anchor)
    end = text.find("    },\n", idx) + len("    },\n")
    blocks = ''
    for stem, suf in EVAL_ENTRIES:
        if stem in text:
            continue
        blocks += _eval_entry(stem, suf)
    if not blocks:
        return text
    return text[:end] + blocks + text[end:]


def patch_plot(text):
    candidates = [
        "    '/content/drive/MyDrive/EcoCAR/training_runs/exp64_rmt_gca_anchor_cls_sep_vfl_long_warmup_full_data_joint_full14_metrics.json',\n",
        "    '/content/drive/MyDrive/EcoCAR/training_runs/exp63_rmt_gca_anchor_cls_sep_topk4_vfl_full_data_joint_full12_metrics.json',\n",
        "    '/content/drive/MyDrive/EcoCAR/training_runs/exp62_rmt_gca_anchor_cls_sep_vfl_iou_match_long14_joint_full14_metrics.json',\n",
        "    '/content/drive/MyDrive/EcoCAR/training_runs/exp57_rmt_gca_anchor_cls_sep_vfl_full_data_long12_joint_full12_metrics.json',\n",
    ]
    anchor = next((c for c in candidates if c in text), None)
    if anchor is None:
        raise RuntimeError('No plot anchor.')
    additions = ''
    for stem, _suf in EVAL_ENTRIES:
        run_tag = PLOT_TAG[stem]
        if f'{stem}_{run_tag}_metrics.json' in text:
            continue
        additions += _plot_entry(stem, run_tag)
    if not additions:
        return text
    return text.replace(anchor, anchor + additions, 1)


def patch_video(text):
    anchor = "PROFILE_CANDIDATES = [\n"
    if anchor not in text:
        return text
    additions = ''
    for stem, _suf in EVAL_ENTRIES:
        run_tag = PLOT_TAG[stem]
        if f'{stem}_{run_tag}.tar' in text:
            continue
        additions += _video_entry(stem, run_tag)
    if not additions:
        return text
    return text.replace(anchor, anchor + additions, 1)


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
