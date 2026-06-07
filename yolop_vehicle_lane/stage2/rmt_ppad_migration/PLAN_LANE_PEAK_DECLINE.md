# Lane IoU peaks at ep~11 then declines - diagnosis + experiment plan

Date 2026-05-30. From NB97 (clr_lane_polyline_full_v2, the STALE config
run: diff_clamp=None, no fliplr).

## Diagnosis (high confidence)

Per-epoch evidence:
```
       ep1    ep11(peak)  ep25
IoU    0.024  0.0915      0.0850   <- lane peaks ep11, declines
subAcc 0.540  0.690       0.683    <- same peak epoch
mAP50  0.18   0.80        0.82     <- detection KEEPS rising
Det_gate 0.51 0.62        0.65     <- detection decouples (rising)
Seg_gate 0.51 0.54        0.55     <- LANE STAYS COUPLED to shared trunk
```

This is multi-task **negative transfer**, not a single bug and not mainly
the spikes:
- GCA fuses `out = shared + gate*(task - shared)`.
- Detection's gate climbs to 0.65 (it adapts onto its own features).
- The lane gate barely moves (0.54) => the lane branch stays ~46% on the
  SHARED backbone.
- The shared backbone keeps being optimized for detection (its gradient
  dominates the trunk), so after the lane peak (~ep11) it overwrites the
  lane-relevant features -> lane IoU/subAcc decline while detection rises.

The spikes (353, up to 27,700, from clrkd+noclamp) add gradient noise that
worsens the post-peak wobble but are NOT the primary cause.

Caveat: pixel-IoU(lane) is a weak proxy (rasterized top-N raw curves vs
thin GT lines; saturates ~0.09). The TRUE lane quality (CLRKDNet curve-F1)
is unknown - part of the "decline" may be metric noise near saturation.

## What is already in place (mitigations, not cures)
- Combined fitness + EarlyStopping => best.pt is kept at the ep~11 peak, so
  the DELIVERABLE is the peak model. The decline doesn't lose the best ckpt.
- clamp=100 (restored) => removes the spike noise.
- fliplr=0.5 + weight_decay=0.05 => regularization vs overfitting.
NONE of these were in the run that declined (stale Drive).

## Experiment plan (prioritized, cheap -> targeted)

### Exp 1 - re-run with the synced fixes (CHEAP, do first)
Re-run NB97 as-is (now clamp=100 + fliplr=0.5 + wd=0.05). Tests whether
removing spike noise + adding augmentation alone flattens the decline.
PASS if IoU plateaus instead of declining, or peaks higher.

### Exp 2 - two-phase trunk freeze (THE targeted fix; implemented)
`--freeze-trunk-after 12`: freeze the shared backbone+neck (model[0:28])
at epoch 12 (just after the lane peak). Then ONLY the task decoders + GCA
adapters + lane head train, on a locked trunk. Detection won't degrade
(its decoder still trains); lanes can keep climbing because the trunk no
longer drifts to detection. Already wired into train_lane_only.py
(default -1 = off). Expect: IoU keeps rising past ep11 instead of declining.

### Exp 3 - curve-F1 metric (know the TRUTH)
Build the CLRKDNet-style curve-F1 (decode -> NMS/top-k -> spline ->
rasterize width-30 -> Hungarian match -> F1@0.5) as an alternate column.
If curve-F1 keeps RISING while pixel-IoU plateaus/declines, the model is
still improving and the "decline" is a pixel-IoU artifact -> switch the
headline metric. Decisive for whether Exp 2 is even needed.

### Exp 4 - strengthen GCA seg decoupling (if Exp 2 insufficient)
The root is Seg_gate stuck at 0.54. Options: raise the gate floor for the
seg path, enlarge task_adapter_seg, or add a small lane-gate-up
regularizer. Lets lanes decouple from the trunk WITHOUT a hard freeze
(keeps both tasks improving jointly).

## What NOT to do
- Kendall uncertainty weighting (prior exp01-71 found it migrates weight
  away from detection - hurts).
- Reading pixel-IoU as ground truth before Exp 3.
- Training 250 epochs (lanes peak ~11; use 120 + early stop + best.pt).

## Recommended sequence
1. Exp 1 (re-run synced) - 1 session.
2. In parallel build Exp 3 (curve-F1) so we know the true trajectory.
3. If Exp 1 still declines: Exp 2 (`--freeze-trunk-after 12`).
4. If still stuck: Exp 4 (GCA).
