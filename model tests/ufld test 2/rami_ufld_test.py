import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np

VIDEO_PATH = "inputs/dashcam_florida.mp4"

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


# -----------------------------
# Preprocess
# -----------------------------
def preprocess(frame):
    img = cv2.resize(frame, (800, 288))
    img = img[:, :, ::-1]  # BGR → RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img).unsqueeze(0)


# -----------------------------
# Dummy lane decoding (visual)
# -----------------------------
def draw_lanes(frame):
    h, w, _ = frame.shape

    # Fake lanes (for now — replace with real decode later)
    for i in range(4):
        for y in range(0, h, 20):
            x = int(w * (0.2 + i * 0.15))
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    return frame


# -----------------------------
# Main loop
# -----------------------------
def run(video_path, weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(weight_path, device)

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp = preprocess(frame).to(device)

        with torch.no_grad():
            output = model(inp)

        # TODO: replace with real lane decoding
        frame = draw_lanes(frame)

        cv2.imshow("Lane Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def draw_lanes(frame):
    h, w, _ = frame.shape

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Region of interest (bottom half of image)
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, h),
        (w, h),
        (int(w * 0.6), int(h * 0.6)),
        (int(w * 0.4), int(h * 0.6)),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough line detection
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    # Draw lines
    line_img = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Overlay lines on original frame
    result = cv2.addWeighted(frame, 0.8, line_img, 1, 0)

    return result

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    run(
        video_path=VIDEO_PATH,
        weight_path="weights/culane_res18.pth"
    )