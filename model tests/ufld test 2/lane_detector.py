import cv2
import time
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np

VIDEO_PATH = "inputs/dashcam_florida.mp4"

class LaneDetector:
    def __init__(self, weight_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(weight_path, self.device)

    def run(self, source, save=True, show=True, output="output.mp4"):
        cap = cv2.VideoCapture(source)

        fps = cap.get(cv2.CAP_PROP_FPS)
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width, height = 640, 480

        if save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output, fourcc, fps, (width, height))

        frame_count = 0
        total_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()

            # inference
            frame = cv2.resize(frame, (640, 480))
            inp = preprocess(frame).to(self.device)
            with torch.no_grad():
                _ = self.model(inp)

            # draw lanes
            frame = draw_lanes(frame)

            # timing
            elapsed = time.time() - start
            total_time += elapsed
            frame_count += 1

            fps_inst = 1.0 / elapsed

            # overlay FPS
            cv2.putText(frame, f"{fps_inst:.1f} FPS", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if show:
                cv2.imshow("Lane Detection", frame)

            if save:
                out.write(frame)

            print(f"Frame {frame_count}: {elapsed*1000:.2f} ms ({fps_inst:.1f} FPS)")

            if show and cv2.waitKey(1) == 27:
                break

        cap.release()
        if save:
            out.release()
        cv2.destroyAllWindows()

        print("\n=== SUMMARY ===")
        print(f"Avg FPS: {frame_count / total_time:.2f}")

# -----------------------------
# Simple UFLD-style model
# -----------------------------
class UFLD(nn.Module):
    def __init__(self, num_lanes=4, griding_num=100):
        super().__init__()
        self.num_lanes = num_lanes
        self.griding_num = griding_num

        backbone = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_lanes * griding_num)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x.view(-1, self.num_lanes, self.griding_num)


# -----------------------------
# Load model
# -----------------------------
def load_model(weight_path, device):
    model = UFLD().to(device)

    try:
        state = torch.load(weight_path, map_location=device)

        # handle different checkpoint formats
        if "state_dict" in state:
            state = state["state_dict"]

        new_state = {}

        for k, v in state.items():
            # remove "module." prefix
            if k.startswith("module."):
                k = k[7:]

            # map backbone weights
            if k.startswith("model."):
                k = k.replace("model.", "encoder.")

            # ignore classifier mismatch
            if k.startswith("cls"):
                continue

            new_state[k] = v

        missing, unexpected = model.load_state_dict(new_state, strict=False)

        print("Loaded weights with:")
        print("  Missing keys:", len(missing))
        print("  Unexpected keys:", len(unexpected))

    except Exception as e:
        print("⚠️ Weight loading failed:", e)
        print("Continuing with random weights")

    model.eval()
    return model


def preprocess(frame):
    img = cv2.resize(frame, (800, 288))
    img = img[:, :, ::-1]  # BGR → RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img).unsqueeze(0)


def draw_lanes(frame, max_lanes=5):
    """
    Detects and draws multiple lanes on a frame.
    Returns the frame with lanes overlaid.
    """
    frame_h, frame_w = frame.shape[:2]
    
    # --- Preprocess ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # --- ROI mask ---
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, frame_h),
        (frame_w, frame_h),
        (int(frame_w*0.6), int(frame_h*0.6)),
        (int(frame_w*0.4), int(frame_h*0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    edges = cv2.bitwise_and(edges, mask)
    
    # --- Hough lines ---
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=50)
    if lines is None:
        return frame
    
    # --- Cluster lines ---
    lanes = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if abs(slope) < 0.1:  # ignore horizontal
            continue
        x_bottom = x1 if y1 > y2 else x2

        assigned = False
        for lane in lanes:
            if abs(lane['slope'] - slope) < 0.2 and abs(lane['x_bottom'] - x_bottom) < 50:
                lane['points'].extend([(x1, y1), (x2, y2)])
                lane['slope'] = (lane['slope'] + slope) / 2
                lane['x_bottom'] = (lane['x_bottom'] + x_bottom) / 2
                assigned = True
                break
        if not assigned:
            lanes.append({'points': [(x1, y1), (x2, y2)], 'slope': slope, 'x_bottom': x_bottom})
    
    # Sort left -> right
    lanes.sort(key=lambda l: l['x_bottom'])
    lanes = lanes[:max_lanes]

    # --- Fit curves and draw ---
    overlay = frame.copy()
    for lane in lanes:
        pts = lane['points']
        if len(pts) < 4:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])

        # remove duplicate y-values to avoid RankWarning
        ys_unique, idx = np.unique(ys, return_index=True)
        xs_unique = xs[idx]
        if len(xs_unique) < 4:
            continue

        # quadratic fit
        poly = np.polyfit(ys_unique, xs_unique, 2)
        y_vals = np.linspace(int(frame_h*0.6), frame_h, 50)
        x_vals = np.polyval(poly, y_vals)

        for i in range(len(y_vals)-1):
            pt1 = (int(x_vals[i]), int(y_vals[i]))
            pt2 = (int(x_vals[i+1]), int(y_vals[i+1]))
            cv2.line(overlay, pt1, pt2, (0, 255, 0), 3)

    # overlay lanes
    return cv2.addWeighted(frame, 0.7, overlay, 1, 0)


if __name__ == "__main__":
    lane_model = LaneDetector("weights/culane_res18.pth")
    lane_model.run(
        source=VIDEO_PATH,
        save=True,
        show=True
    )