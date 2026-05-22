import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path, engine_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Set memory pool limit (e.g., 4GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    
    # Enable FP16 fallback
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8) # REQUIRED for QDQ graphs to use INT8 math!
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    # NO CALIBRATOR needed because the ONNX already contains QDQ nodes (QAT)

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
    parser.add_argument('--onnx', type=str, default='YOLOPX/weights/yolopx_qat_epoch_0.onnx', help='ONNX file path')
    parser.add_argument('--engine', type=str, default='YOLOPX/weights/yolopx_qat_int8.engine', help='Output Engine path')
    args = parser.parse_args()

    build_engine(args.onnx, args.engine)
