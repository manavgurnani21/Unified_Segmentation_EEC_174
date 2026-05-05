from ultralytics import YOLO
import cv2

VIDEO_PATH = "inputs/dashcam_florida_rain.mp4"

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo26m-seg.pt").to("cuda") # models: n, s, m, l, "-seg" for segmentation, otherwise detection
# model.model.half() # use fp16 instead of fp32, negligible accuracy loss but twice the speed
# model.export(format="engine") # for deployment not experimentation, cannot add training to the model

# Train the model using the 'coco8.yaml' dataset for 3 epochs
# results = model.train(data="coco8.yaml", epochs=3)
# Evaluate the model's performance on the validation set
# results = model.val()

# Perform object detection on an image using the model
results = model(VIDEO_PATH, device="cuda", imgsz=640, save=True, stream=True, project="outputs", name="video")

# required to pass every frame to save the video when stream=True. Can remove this when using an actual video stream
for num, frame in enumerate(results):
	pass

# Export the model to ONNX format
# success = model.export(format="onnx")