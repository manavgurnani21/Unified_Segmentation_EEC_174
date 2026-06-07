# P1 migration log

Phase: convert BDD lane masks to CLRHead 78-D polyline targets.

## 2026-05-25  P1 tools authored

Created four scripts under `tools/`:
- `mask_to_polylines.py` - skeletonize + connected components + 3rd-order polyfit
- `polyline_to_clrnet_format.py` - polynomial -> 78-D target vector
  (matches CLRKDNet `transform_annotation` exactly, including the `theta_far`
  averaging and the outside/inside hstack convention)
- `verify_conversion.py` - 3-panel visual check + dilated-IoU sanity test
- `convert_dataset.py` - full-dataset batch converter

NB80 (`notebooks/stage2_notebook_80_P1_lane_conversion.ipynb`) drives the
above on Colab. Output `.pt` files write to `/content/lane_targets/` LOCALLY
(per the Drive-vs-local I/O rule); a single tarball
`lane_targets_clr_v1.tar.gz` is then copied back to Drive for persistence.

## Format notes (deviations from the appendix path-3 spec)

- The appendix's `extract_lane_polylines(mask, min_length, poly_order)` returns
  a list of polynomial coefficient arrays. Our implementation returns a list of
  `(coeffs, y_min, y_max)` tuples - the y-range is essential for the next
  conversion step (computing `start_y`, `length`) and there is no other way
  to recover it from coeffs alone. Documented in the docstring.
- Below-bottom extrapolation uses the polynomial's tangent slope at y_max
  (one-point analytic derivative) rather than CLRKDNet's two-closest-points
  linear regression. Same first-order behavior, simpler and faster for our
  polynomial input.
- Above-top extrapolation: we DROP these grid points (do not extend the lane
  toward the horizon). CLRKDNet does the same (the `interp` spline is
  defined only inside `[y.min(), y.max()]`).

## 2026-05-25  Bug: scipy.ndimage.label 4-connectivity dropped every diagonal lane

NB80 cell 2 smoke test failed with `Found 0 polylines (expected 1)` on a clean
synthetic diagonal stripe. Root cause: `scipy.ndimage.label(skel)` defaults to
`generate_binary_structure(2, 1)` = 4-connected cross. The 1-px-wide skeleton
of any diagonal line is a staircase where consecutive pixels are diagonal
neighbors, NOT 4-connected. 4-connectivity splits the staircase into 2-pixel
fragments, all below `min_length=20`, so the polylines list was empty.

Fix: pass `structure=np.ones((3, 3))` to force 8-connectivity. This is the
right choice anyway for any "lane-like" geometry where the skeleton can turn
diagonally between rows.

Would also have broken real BDD masks at scale - every diagonal lane on
720x1280 would have come out empty, and the `--min_length 20` threshold
would have silently masked the bug as "noisy mask".

## 2026-05-25  Bug: bottom-extrap added fake lane tails (NB80 cell 5 panels)

User's NB80 verification panels showed lanes extending way past the source
mask, often dragged into the bottom corner at extreme angles. Root cause:
`polyline_to_clr_format` copied CLRKDNet's behavior of linearly extrapolating
each lane down to the image bottom using the polynomial's tangent at y_max.

CLRKDNet's assumption was correct for CULane (every annotated lane is a
road marking that physically continues into the camera-near foreground).
That assumption is FALSE for BDD's lane mask channel, which includes short
distant lanes, stop lines, crosswalks, and partial-frame markings that
genuinely end mid-image.

For a near-horizontal lane (small y_max - y_min, large dx/dy from polyfit),
the linear extrap multiplied a steep slope by a huge `extrap_ys - y_max`,
sending x off-screen at a sharp angle.

Fix: `_sample_lane_from_poly` now returns `extrap_xs = []` always. Only
sample x at offsets_ys that fall inside [y_min, y_max].

## 2026-05-25  Filter: drop near-horizontal and high-residual components

Same NB80 panel review showed two additional failure modes:
- Crosswalks / stop lines (very high x_span / y_span ratio) entered polyfit,
  produced near-vertical polynomials (steep slope), and after the extrap fix
  still rasterized poorly because there are only 0-1 grid offsets inside
  their tiny y-span.
- V-shapes formed by two lanes touching at the apex: skeletonize+label
  collapses them into one component; polyfit fits a compromise curve through
  both lines that matches neither. Visible as a smooth bow in image 2.

Fix: `extract_lane_polylines` now rejects components when ANY of:
- `y_span < min_y_span` (default 15 px) - too short to be a real x=f(y) lane
- `x_span / y_span > max_aspect` (default 3.0) - near-horizontal
- polyfit RMS residual > `max_residual_px` (default 12 px) - bad fit

V-shape detection by residual is a heuristic, not a perfect splitter. The
"right" fix (skeleton branch-point splitting) is deferred to a later
iteration if the filtered version causes too much recall loss on real
BDD lanes.

## 2026-05-25  Bug: slot misalignment after dropping the bottom-extrap

After removing the bottom-extrap (previous fix), NB80 panels showed lanes
with correct SHAPE but WRONG POSITION - green lines were uniformly dragged
toward the bottom of the canvas and toward the right, regardless of where
the source white lane actually was.

Root cause: CLRKDNet's `transform_annotation` left-justifies x values into
slots starting at index 0. That accidentally aligns slot 0 with the
image-bottom row of `prior_ys` because CLRKDNet's CULane targets ALWAYS
reach the image bottom (via mandatory extrapolation). When I disabled the
extrap, a mid-image lane's x values still ended up in slots [0:length],
but those slots map to `prior_ys[0:length]` = image-bottom rows. The
rasterizer dutifully drew the lane at the bottom of the image.

Fix: `polyline_to_clr_format` now writes x values at `out[6+start:6+start+length]`
where `start` is the slot index of the lane's bottom-most offset_y. Other
slots stay at -1e5 sentinel. `out[2] = start / n_strips` honestly reflects
where the lane begins.

This is a semantic change from CLRKDNet's left-justified convention. The
rasterizer and P5's loss code (when written) must both expect slot-aligned
packing. Document this in the format spec for P5+.

## 2026-05-25  PATH PIVOT: mask-derived -> BDD native polyline JSONs

After three rounds of mask-pipeline bug fixes (4/8 connectivity, no-extrap
slot misalignment, V-shape merging, near-horizontal filtering), the user
decided the mask path was fundamentally inappropriate and asked to switch
to BDD's native polyline JSON labels.

Rationale: BDD's `bdd100k/labels/lane/polygons/*.json` files contain one
`poly2d` entry per lane object with exact vertex sequences. There is no
upstream mask -> skeleton -> components -> polyfit loss to recover.

New scripts (added; old ones kept for reference but no longer wired into
NB80):
- `bdd_label_loader.py` - JSON parser, handles v1 and v2 layouts, category
  filter (drops crosswalk by default)
- `vertices_to_clr_vector.py` - per-polyline -> 78-D, slot-aligned packing
  via InterpolatedUnivariateSpline (k=min(3, n-1))
- `convert_bdd_labels.py` - batch driver
- `verify_polyline_conversion.py` - polyline-aware verifier (no mask round-trip)

NB80 now downloads `bdd100k_lane_labels_trainval.zip` from Drive (one-time
manual download required), extracts to `/content/bdd100k_lane/`, runs
verification + batch conversion + tarball back to Drive as
`lane_targets_clr_v1_polyline.tar.gz`.

Old artifacts retained:
- `mask_to_polylines.py`, `polyline_to_clrnet_format.py`,
  `convert_dataset.py`, `verify_conversion.py` remain in tools/ as a
  reference implementation of the mask path. Not deleted because the
  bug analysis is documented above and may be useful when comparing
  approaches.

## 2026-05-25  Source data: BDD v1 monolithic JSONs (not v2 per-image)

Project uses the ORIGINAL BDD100K v1 dataset, NOT the v2 release. The
v1 lane labels live inside `bdd100k_labels.zip` as two monolithic JSONs:
    `bdd100k/labels/bdd100k_labels_images_train.json`  (~370 MB, ~70k records)
    `bdd100k/labels/bdd100k_labels_images_val.json`    (~50 MB, ~10k records)
Each JSON is a list of per-image records. Each record has `labels`
containing both `category="lane"` objects (the lane lines) AND
`category="drivable area"` objects (skip these).

For v1 lane objects, the subtype is in `attributes.laneType`
(e.g. "single white", "crosswalk"), NOT in `category`. The category
field is always literally "lane" for lane lines.

Loader changes:
- `_normalize_category` now skips `drivable area`, dives into
  `attributes.laneType` when category == "lane", and falls through to
  category-as-subtype for v2.
- New `iter_v1_records(big_json_path)` parses the monolithic file and
  yields (image_stem, polylines) per record.
- New `iter_records_auto(path)` auto-detects file vs directory and
  dispatches between v1 and v2 iterators.

Batch converter and verifier now take a single `--input` (file or dir);
NB80 passes the v1 .json paths discovered after extracting
`bdd100k_labels.zip` from Drive.

## 2026-05-25  Correction: BDD lane labels are at lane_{train,val}.json, NOT bdd100k_labels_images_*.json

I had assumed the user's bdd100k_labels.zip contained the monolithic
all-tasks labels (`bdd100k_labels_images_{train,val}.json`). User pointed
out the project already has a working BDD lane loader at
`stage2/scripts/04_prepare_bdd_curve_labels.py` and asked me to actually
inspect it.

That existing loader's `locate_lane_source` looks for, in priority order:
  1. `bdd100k/labels/lane/polygons/lane_{train,val}.json` (consolidated)
  2. `bdd100k/labels/100k/{train,val}/*.json` (per-image dir)
  3. Several rglob fallbacks

It also has 691 lines of battle-tested parsing logic covering:
  - bezier-curve sampling (LLCCL vertex-type strings -> dense points)
  - multiple polygon container shapes
  - multiple category-string shapes
  - drivable-area filtering
  - already-correct crosswalk drop via LANE_TRAIN_CATS

Fix: my `bdd_label_loader.py` no longer reinvents this. It imports the
existing helpers dynamically (importlib because of the `04_` filename
prefix) and exposes:
  - `locate_lane_source` (re-export)
  - `load_lane_records`  (re-export)
  - `iter_records_auto`  (thin wrapper yielding (stem, polylines) tuples)

`convert_bdd_labels.py` and `verify_polyline_conversion.py` updated to use
the new signature (list of vertex arrays, no category tuples - category
filtering is done upstream by the proven loader).

NB80 cell 3 now uses `locate_lane_source` to discover the lane JSONs after
extraction, instead of hardcoding a path.

## 2026-05-25  Selection + snap fixes for "green shorter than yellow" panels

User reported in NB80 cell 5 panels: many green lines shorter than yellow,
far lanes missing entirely. Two root causes:

1. **max_lanes=4 truncation made invisible.** The existing extract_lanes
   sort `(y_span DESC, arc DESC)` favors ego lanes (which span more
   vertical pixels). Far lanes (which span less) lose the top-4 contest
   and get dropped. Yellow panel drew ALL polylines; green only the 4 that
   survived. Looked like missing lanes; really truncation.

2. **Slot-quantization edge loss.** The slot range `inside_grid = (offsets_ys
   >= y_t_min) & (offsets_ys <= y_t_max)` is strictly inside the polyline's
   y span. Up to ~9 px (one strip) is lost at each end where offsets_ys
   doesn't perfectly align with the polyline endpoints.

Fixes:
- `select_lanes_spatial` (new, in vertices_to_clr_vector.py): picks the
  longest lane first, then maximizes mean-x distance from already-picked
  lanes. Result: ego-left + ego-right + far-left + far-right instead of
  4 ego-clustered lanes.
- `build_full_target_tensor_v2` gained a `selection='spatial'|'first'`
  flag; default is spatial.
- `vertices_to_clr_format` now extends the slot range by half a strip on
  each side AND clips the spline-evaluation y to the polyline's true
  domain - we snap to the nearest grid slot without dangerous cubic
  extrapolation. Recovers most of the ~9 px lost at the endpoints.
- `verify_polyline_conversion.verify_record` now draws PICKED polylines
  in yellow and DROPPED polylines in dim gray, so the user sees
  truncation directly instead of inferring it. Per-sample log line now
  reports loaded / picked / dropped / encoded counts.

This does NOT change max_lanes=4 itself (that's a CLRHead architectural
constant from the CULane config). If we later find BDD scenes routinely
have >4 worth keeping, P5 can bump it.

## 2026-05-25  max_lanes bumped 4 -> 8 to match BDD's lane density

CULane (CLRHead's original training set) tops out at 4 lanes per scene, so
CLRKDNet hardcoded `max_lanes = 4`. BDD scenes routinely have 5-8 visible
markings (ego left/right + 2-3 adjacent lanes), so `max_lanes = 4` was
silently truncating informative far-lane targets and producing
"green-shorter-than-yellow" panels.

This is a TARGET-TENSOR-SHAPE change only:
- CLRHead's model output is still `(B, num_priors=192, 78)` - the 192
  learned anchor priors and the dynamic SimOTA-style matching loss
  (`assign(predictions, target)` at clr_head.py:377) are unchanged.
- The training loss does `target = target[target[:,1] == 1]` to drop
  padding rows, so any unused capacity costs nothing.
- Inference NMS just keeps top_k = max_lanes - P7 must set `nms_topk = 8`.

What changed in P1 scripts:
- `vertices_to_clr_vector.py`: defaults bumped to max_lanes=8 in both
  `build_full_target_tensor_v2` and `select_lanes_spatial`
- `convert_bdd_labels.py`: CLI `--max_lanes` default now 8 (with help text
  explaining the BDD deviation)
- `verify_polyline_conversion.py`: new CLI `--max_lanes` flag (default 8)
  - must match convert's value
- NB80 cell 4 declares `MAX_LANES = 8` and both cells 4 and 6 pass it
  through, so the value lives in one place per run.
- Output .pt shape is now `(8, 78)`, ~2.5 KB each.

Downstream phases that need to align with this:
- P5 loss: no code change (filter handles any size); just keep CULane
  loss weights as starting point
- P6 dataloader: stack into (B, 8, 78)
- P7 validator: set `nms_topk = 8` in test_parameters

If P8 ablation shows max_lanes=8 hurts vs 4, easy to revert by re-running
NB80 with `MAX_LANES = 4`.

## 2026-05-25  Bug: iter_records_auto dropped no-lane records via load_lane_records

First NB80 polyline-path run finished with 65813 train .pt vs 70000 source
JSONs - 4187 records (~6%) silently disappeared. Same on val: 9425 vs 10000.

Root cause: my `iter_records_auto` delegated to `load_lane_records` from
the existing 04_prepare_bdd_curve_labels.py, which does
    if name and lanes:
        records[name] = lanes
- records with ZERO valid lanes after filtering get dropped entirely.

That's appropriate for the original consumer (a mask renderer that has
nothing to render without lanes) but wrong for ours: the downstream P6
dataloader needs a .pt file PER IMAGE so the image<->target pairing is
1:1. Missing .pts break the dataloader.

Fix: rewrote `iter_records_auto` to NOT call load_lane_records. Now it
iterates records directly (with explicit handling of v1 array-shaped
files vs per-image dirs) and yields (stem, lanes_or_empty_list) for
EVERY record. extract_lanes is still called per record so all the
category filtering is preserved.

After the fix, expected counts:
  train: 70000 .pt, ~5000 empty (~7%)
  val:   10000 .pt, ~720 empty (~7%)

NB80 cell 9 empty-fraction tolerance bumped from <30% to <15% to reflect
the more honest baseline (records with all-filtered lanes are now visible
in the count instead of silently dropped).

## Pending follow-ups

- `min_length=20` is a guess from the appendix. May need tuning if BDD has
  many short lane segments that should be kept (or many noise blobs that
  should be dropped). Cell 4 of NB80 will surface this empirically.
- The full-dataset run will reveal if BDD's "lane" mask channel actually
  encodes per-instance lanes or just per-class pixels (where many lanes
  fuse into one connected component). If the latter, we need to add a
  line-instance-splitting step before polyfit.
