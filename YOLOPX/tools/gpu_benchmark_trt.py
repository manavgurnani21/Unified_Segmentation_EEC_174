import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import cv2
import argparse
from pathlib import Path

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def preprocess_frame(frame, img_size):
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    return img.astype(np.float32) / 255.0

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine, context):
    inputs = []
    outputs = []
    stream = cuda.Stream()
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        size = trt.volume(engine.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        context.set_tensor_address(name, int(device_mem))
        
        # Append to the appropriate list
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            
    return inputs, outputs, stream

def do_inference_v3(context, inputs, outputs, stream):
    # Transfer input data to the GPU
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    # Run inference
    context.execute_async_v3(stream_handle=stream.handle)
    # Transfer predictions back from the GPU
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    # Synchronize the stream
    stream.synchronize()
    return [out.host for out in outputs]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='YOLOPX/weights/yolopx_qat_int8.engine')
    parser.add_argument('--source', type=str, default='YOLOPX/demo/2.gif')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-frames', type=int, default=None)
    opt = parser.parse_args()

    print(f"=== Loading TensorRT Engine: {opt.engine} ===")
    with open(opt.engine, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # We must explicitly set the input shape for dynamic/explicit batch engines if needed
    input_name = engine.get_tensor_name(0)
    context.set_input_shape(input_name, (opt.batch_size, 3, opt.img_size, opt.img_size))

    inputs, outputs, stream = allocate_buffers(engine, context)

    print(f"=== Preprocessing frames ===")
    cap = cv2.VideoCapture(opt.source)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(preprocess_frame(frame, opt.img_size))
        count += 1
        if opt.max_frames and count >= opt.max_frames: break
    cap.release()

    if len(frames) == 0:
        print("No frames loaded.")
        return
    print(f"  Done — {len(frames)} frames loaded")
    
    all_frames = np.stack(frames)
    
    # Warmup
    print("=== Warming up ===")
    warmup_batch = all_frames[:opt.batch_size]
    if warmup_batch.shape[0] < opt.batch_size:
        pad_size = opt.batch_size - warmup_batch.shape[0]
        warmup_batch = np.pad(warmup_batch, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='constant')
    
    np.copyto(inputs[0].host, warmup_batch.ravel())
    do_inference_v3(context, inputs, outputs, stream)
    
    print(f"=== Running batched inference (Batch Size: {opt.batch_size}) ===")
    total_frames = len(frames)
    start = time.time()
    
    processed_frames = 0
    for i in range(0, total_frames, opt.batch_size):
        batch = all_frames[i:i+opt.batch_size]
        if batch.shape[0] != opt.batch_size:
            print(f"  Skipping last batch of size {batch.shape[0]}")
            continue
            
        np.copyto(inputs[0].host, batch.ravel())
        do_inference_v3(context, inputs, outputs, stream)
        processed_frames += opt.batch_size
        
    elapsed = time.time() - start

    print(f"\n=== Results (Native TensorRT INT8) ===")
    print(f"  Frames      : {processed_frames} (out of {total_frames})")
    print(f"  Batch size  : {opt.batch_size}")
    print(f"  Time        : {elapsed:.2f}s")
    print(f"  Pure GPU FPS: {processed_frames / elapsed:.1f}")
    print(f"  ms/frame    : {1000 / (processed_frames / elapsed):.1f}")

if __name__ == "__main__":
    main()