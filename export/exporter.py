import torch
from pathlib import Path


def export_onnx(model, output_path="model.onnx", imgsz=640, batch=1, opset=17, simplify=True):
    """Export model to ONNX format."""
    import onnx
    model.cpu().eval()  # export on CPU so model and dummy input share a device
    dummy = torch.zeros(batch, 3, imgsz, imgsz)
    output_path = Path(output_path)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={
            "images": {0: "batch"},
            "output0": {0: "batch"},
        },
    )

    model_onnx = onnx.load(str(output_path))
    onnx.checker.check_model(model_onnx)

    if simplify:
        try:
            import onnxsim
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "ONNX simplify failed"
            onnx.save(model_onnx, str(output_path))
        except ImportError:
            print("onnxsim not installed; skipping simplification")

    print(f"ONNX model saved to {output_path}")
    return output_path


def export_torchscript(model, output_path="model.torchscript", imgsz=640, batch=1):
    """Export model to TorchScript."""
    model.eval()
    dummy = torch.zeros(batch, 3, imgsz, imgsz)
    ts = torch.jit.trace(model, dummy, strict=False)
    output_path = Path(output_path)
    ts.save(str(output_path))
    print(f"TorchScript saved to {output_path}")
    return output_path


def export_tensorrt(onnx_path, output_path="model.engine", fp16=True, workspace_gb=4):
    """Export ONNX → TensorRT engine (requires tensorrt installed)."""
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError("tensorrt not installed")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for e in range(parser.num_errors):
                print(parser.get_error(e))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine_bytes = builder.build_serialized_network(network, config)
    with open(output_path, "wb") as f:
        f.write(engine_bytes)
    print(f"TensorRT engine saved to {output_path}")
    return output_path
