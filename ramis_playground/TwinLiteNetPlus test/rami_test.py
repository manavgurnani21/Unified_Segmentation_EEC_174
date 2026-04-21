import cv2
import torch
import numpy as np
from pathlib import Path
from model.model import TwinLiteNetPlus
from model import config
from types import SimpleNamespace
import utils

WEIGHTS = "weights/medium.pth"
VIDEO_PATH = "inputs/dashcam_florida_rain.mp4"

# ---------------------------
# Load Model
# ---------------------------
def load_twinlitenet(device):
    model = TwinLiteNetPlus(args=SimpleNamespace(config="medium")).to(device)
    checkpoint = torch.load(WEIGHTS, map_location=device)

    # handle DataParallel prefix if present
    state_dict = checkpoint
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state)
    model.eval()

    # FP16 optimization
    if device.type == "cuda":
        model.half()
    return model


# ---------------------------
# Run Inference
# ---------------------------
def run(output="twinlitenet_out.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_twinlitenet(device)

    cap = cv2.VideoCapture(VIDEO_PATH)

    # fixed resolution
    target_w, target_h = 640, 480

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, fps, (target_w, target_h))

    import time
    frame_count = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        print("hi")
        if not ret:
            break

        # resize + fix shape
        frame = cv2.resize(frame, (target_w, target_h))
        img = frame[:, :, ::-1].copy()  # BGR -> RGB
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        inp = torch.from_numpy(img).unsqueeze(0).to(device)

        # FP16 if GPU
        if device.type == "cuda":
            inp = inp.half()

        start = time.time()

        with torch.no_grad():
            seg_out = model(inp)  # segmentation output

        elapsed = time.time() - start
        total_time += elapsed
        frame_count += 1
        fps_inst = 1.0 / elapsed

        # process segmentation output
        # TwinLiteNet outputs height×width×classes masks;
        # pick lane class channel
        seg_mask = seg_out.squeeze().cpu().float().numpy()
        lane_mask = seg_mask[1] if seg_mask.shape[0] > 1 else seg_mask[0]
        lane_mask = (lane_mask > 0.5).astype(np.uint8) * 255

        # overlay color
        color_mask = np.zeros_like(frame)
        color_mask[:, :, 1] = lane_mask  # green

        overlay = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)

        cv2.putText(
            overlay,
            f"{fps_inst:.1f} FPS",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("TwinLiteNet", overlay)
        out.write(overlay)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("bye")

    print("\n=== SUMMARY ===")
    print(f"Frames: {frame_count}")
    print(f"Avg FPS: {frame_count/total_time:.2f}")

if __name__ == "__main__":
    run()