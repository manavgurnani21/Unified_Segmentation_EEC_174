# python tools/build_trt.py --onnx weights/yolopx.onnx --engine weights/yolopx_fp16.trt
# Builds a TensorRT FP16 engine from the exported ONNX file.
# Takes several minutes on first run; result is cached to disk.

import argparse


def build_engine(onnx_path, engine_path, fp16=True, max_batch=16, workspace_gb=4):
    import tensorrt as trt

    logger  = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed — check export_onnx.py output")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled")
    elif fp16:
        print("WARNING: GPU does not support fast FP16 — building FP32 engine")

    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    profile.set_shape(
        inp.name,
        min=(1,            3, 640, 640),
        opt=(max_batch//2, 3, 640, 640),
        max=(max_batch,    3, 640, 640),
    )
    config.add_optimization_profile(profile)

    print(f"Building engine (max_batch={max_batch}, workspace={workspace_gb}GB) ...")
    print("This typically takes 3–10 minutes on first run.")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed")

    with open(engine_path, 'wb') as f:
        f.write(memoryview(serialized))
    print(f"Engine saved → {engine_path}  ({serialized.nbytes/1e6:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx',      default='weights/yolopx.onnx')
    parser.add_argument('--engine',    default='weights/yolopx_fp16.trt')
    parser.add_argument('--max-batch', type=int,   default=16)
    parser.add_argument('--workspace', type=int,   default=4, help='workspace memory in GB')
    parser.add_argument('--fp32',      action='store_true', help='build FP32 engine instead of FP16')
    opt = parser.parse_args()

    build_engine(opt.onnx, opt.engine,
                 fp16=not opt.fp32,
                 max_batch=opt.max_batch,
                 workspace_gb=opt.workspace)
