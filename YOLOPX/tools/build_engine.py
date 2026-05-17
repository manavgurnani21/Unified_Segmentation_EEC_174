import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import cv2
import numpy as np
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def preprocess_image(image_path, img_size=640):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img

class Int8EntropyCalibrator2(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_dir, batch_size=8, calibration_samples=500, cache_file="calibration.cache"):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.shape = (batch_size, 3, 640, 640)
        self.device_input = cuda.mem_alloc(trt.volume(self.shape) * trt.float32.itemsize)

        # Get image files
        self.image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.image_files = self.image_files[:calibration_samples]
        self.batches = self.load_batches()
        self.batch_idx = 0

    def load_batches(self):
        batches = []
        for i in range(0, len(self.image_files), self.batch_size):
            batch_files = self.image_files[i:i + self.batch_size]
            if len(batch_files) < self.batch_size:
                break # drop last incomplete batch
            batch_imgs = [preprocess_image(f) for f in batch_files]
            batch_data = np.stack(batch_imgs, axis=0)
            batches.append(batch_data)
        return batches

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.batch_idx < len(self.batches):
            batch_data = self.batches[self.batch_idx]
            cuda.memcpy_htod(self.device_input, batch_data.ravel())
            self.batch_idx += 1
            print(f"Calibrating batch {self.batch_idx}/{len(self.batches)}...")
            return [int(self.device_input)]
        return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_engine(onnx_file_path, engine_file_path, calibrator):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Set memory pool limit (e.g., 4GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    
    # Enable INT8 and FP16 fallback
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator

    print(f"Parsing ONNX file: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print(f"Building TensorRT engine (this may take several minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("ERROR: Engine build failed.")
        return None

    print(f"Saving engine to {engine_file_path}...")
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
    
    print("Engine build successful!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='weights/epoch-195.onnx', help='ONNX file path')
    parser.add_argument('--engine', type=str, default='weights/yolopx_int8.engine', help='Output Engine path')
    parser.add_argument('--data-dir', type=str, default='../bdd100k/bdd100k/images/100k/val', help='Calibration images dir')
    parser.add_argument('--calib-samples', type=int, default=500, help='Number of images for calibration')
    args = parser.parse_args()

    calibrator = Int8EntropyCalibrator2(data_dir=args.data_dir, batch_size=8, calibration_samples=args.calib_samples)
    build_engine(args.onnx, args.engine, calibrator)
