from ultralytics import YOLO
import cv2
from tqdm import tqdm

video_path = "inputs/dashcam_country.mp4"
output_path = "outputs/video1/output.mp4"

# Load model
model = YOLO("yolo26n.pt").to("cuda")

# Open video
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

batch_size = 16
frames = []

for _ in tqdm(range(total_frames)):

    ret, frame = cap.read()
    if not ret:
        break

    frames.append(frame)

    # Run batch inference
    if len(frames) == batch_size:
        results = model(frames)

        for r in results:
            annotated = r.plot()
            out.write(annotated)

        frames = []

# process leftover frames
if frames:
    results = model(frames)
    for r in results:
        annotated = r.plot()
        out.write(annotated)

cap.release()
out.release()