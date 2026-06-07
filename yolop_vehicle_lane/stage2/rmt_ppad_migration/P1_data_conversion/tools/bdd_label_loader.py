"""Phase P1 (polyline-native path): wrapper around the project's existing,
battle-tested BDD lane label loader at
    yolop_vehicle_lane/stage2/scripts/04_prepare_bdd_curve_labels.py

That script's `load_lane_records` + `locate_lane_source` + `extract_lanes`
already handle every BDD label-format quirk this project has encountered:

  - Path layout: consolidated per-split JSON at
        bdd100k/labels/lane/polygons/lane_{train,val}.json
    OR per-image JSONs at
        bdd100k/labels/100k/{train,val}/<stem>.json
    OR various fallback layouts (the existing locate_lane_source tries 4+
    candidate paths before giving up).
  - Category encoding: handles "lane/single white" (already-namespaced),
    "lane" + attributes.laneType (older v1 style), raw subtype as category
    (newer v2 style), and multi-key attribute lookups (laneTypes,
    laneType, lane_type, type, types, subtype, subtypes).
  - Geometry: handles dict-shaped polygons, list-of-list polygons,
    bezier control points (cubic + quadratic sampling), closed/open polylines.
  - Filtering: drops drivable-area objects and crosswalk by default
    (LANE_TRAIN_CATS in that file).

We import that module dynamically because its filename starts with `04_`
(not a valid Python identifier for `import` statements). The Colab session
chdir's to REPO_ROOT, so we can locate it via Path traversal from this
file's own location.

Public API exposed here:
    KNOWN_LANE_SUBTYPES, LANE_TRAIN_CATS  - constants
    locate_lane_source(raw_root, split)    - find lane JSON file or dir
    load_lane_records(source)              - parse all records
    iter_records_auto(source)              - yield (image_stem, polylines)
                                             where polylines = list[ndarray(N,2)]
                                             in original-image pixel coords
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _import_prep_module():
    """Dynamically import stage2/scripts/04_prepare_bdd_curve_labels.py.

    Path layout (Colab working dir = REPO_ROOT = yolop_vehicle_lane):
        REPO_ROOT/stage2/rmt_ppad_migration/P1_data_conversion/tools/THIS_FILE.py
        REPO_ROOT/stage2/scripts/04_prepare_bdd_curve_labels.py
    """
    here = Path(__file__).resolve().parent  # tools/
    stage2_root = here.parent.parent.parent  # stage2/
    candidates = [
        stage2_root / 'scripts' / '04_prepare_bdd_curve_labels.py',
    ]
    for cand in candidates:
        if cand.exists():
            spec = importlib.util.spec_from_file_location('bdd_prepare_v1', cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        f'Could not locate 04_prepare_bdd_curve_labels.py near {here}. '
        f'Tried: {[str(c) for c in candidates]}'
    )


_prep = _import_prep_module()

# Re-export the constants/functions we use downstream.
KNOWN_LANE_SUBTYPES = _prep.KNOWN_LANE_SUBTYPES
LANE_TRAIN_CATS = _prep.LANE_TRAIN_CATS
locate_lane_source = _prep.locate_lane_source
load_lane_records = _prep.load_lane_records
extract_lanes = _prep.extract_lanes


def iter_records_auto(source: Path) -> Iterable[Tuple[str, List[np.ndarray]]]:
    """Yield (image_stem, [vertices_array, ...]) for EVERY per-image record.

    source can be:
      - a path to a consolidated per-split JSON (an array of records)
      - a directory of per-image JSON files (each a dict, or 1-element array)

    Unlike load_lane_records (which silently DROPS records with zero kept
    lanes), we yield every record we can identify by stem - polylines may
    be an empty list. This preserves 1:1 image<->target pairing in the
    downstream dataloader: images with no painted lane markings (night
    scenes, parking lots, etc.) still get a no-lane .pt placeholder.

    image_stem is `Path(record_image_name).stem` so it matches the BDD
    image file's basename without extension.

    vertices_array is shape (N, 2) in ORIGINAL image pixel coords
    (720x1280 for BDD). Crosswalks, drivable-area polygons, and other
    non-lane objects are filtered out by `extract_lanes`.
    """
    import json as _json
    source = Path(source)
    if source.is_file() and source.suffix.lower() == '.json':
        with open(source, 'r', encoding='utf-8') as f:
            data = _json.load(f)
        items = data if isinstance(data, list) else [data]
        for rec in items:
            if not isinstance(rec, dict):
                continue
            name = _prep.record_image_name(rec, str(source))
            if not name:
                continue
            stem = Path(name).stem
            lanes = _prep.extract_lanes(rec)
            yield stem, lanes
    elif source.is_dir():
        for f in sorted(source.glob('*.json')):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = _json.load(fh)
            except Exception as e:
                # Malformed JSON - still emit an empty record so the
                # downstream .pt-per-image invariant holds.
                print(f'  [warn] {f.name}: {type(e).__name__}: {e}', flush=True)
                yield f.stem, []
                continue
            items = data if isinstance(data, list) else [data]
            rec = items[0] if items and isinstance(items[0], dict) else None
            if rec is None:
                yield f.stem, []
                continue
            name = _prep.record_image_name(rec, str(f))
            stem = Path(name).stem if name else f.stem
            lanes = _prep.extract_lanes(rec)
            yield stem, lanes
    else:
        raise ValueError(f'Input is neither a .json file nor a directory: {source}')


def smoke_test() -> int:
    """Build a synthetic record in memory, write it to a temp dir, and
    confirm iter_records_auto reads it back as expected."""
    import json
    import tempfile
    print('[smoke] bdd_label_loader starting', flush=True)
    print(f'[smoke] imported 04_prepare_bdd_curve_labels from {Path(_prep.__file__)}',
          flush=True)
    print(f'[smoke] LANE_TRAIN_CATS = {sorted(LANE_TRAIN_CATS)}', flush=True)

    rec = {
        'name': 'fake_image.jpg',
        'labels': [
            {  # KEEP: real lane line
                'category': 'lane',
                'attributes': {'laneType': 'single white',
                               'laneDirection': 'parallel', 'laneStyle': 'solid'},
                'poly2d': [{'vertices': [[100, 700], [200, 400], [300, 200]],
                            'types': 'LLL', 'closed': False}],
            },
            {  # DROP: crosswalk
                'category': 'lane',
                'attributes': {'laneType': 'crosswalk'},
                'poly2d': [{'vertices': [[10, 700], [1200, 700]], 'types': 'LL'}],
            },
            {  # DROP: drivable area (different category, has poly2d but isn't a lane)
                'category': 'drivable area',
                'attributes': {'areaType': 'direct'},
                'poly2d': [{'vertices': [[0, 700], [1280, 700], [640, 400]],
                            'types': 'LLL', 'closed': True}],
            },
        ],
    }
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as fh:
        json.dump([rec], fh)
        tmp = Path(fh.name)

    records = list(iter_records_auto(tmp))
    print(f'[smoke] iter_records_auto yielded {len(records)} record(s)  (expected 1)')
    if len(records) != 1:
        print('[smoke] FAIL: wrong record count')
        return 1
    stem, polys = records[0]
    print(f'[smoke] stem={stem!r}  n_polylines={len(polys)}  (expected: fake_image, 1)')
    if stem != 'fake_image' or len(polys) != 1:
        print('[smoke] FAIL: stem or polyline count wrong')
        return 1
    verts = polys[0]
    print(f'[smoke] polyline 0 shape={verts.shape}')
    if verts.shape[1] != 2 or verts.shape[0] < 2:
        print('[smoke] FAIL: polyline shape wrong')
        return 1
    print('[smoke] PASS')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(smoke_test())
