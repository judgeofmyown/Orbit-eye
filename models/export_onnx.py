"""
Exports the trained models to ONNX, applies dynamic INT8 quantization, and benchmarks
inference latency across variants — this is the actual "can this run on a Jetson
Nano" check the README's edge-friendly architecture claims were aiming at, instead of
just trusting the depthwise-separable-conv param count.

Compares:
  - PyTorch FP32  (CPU and GPU, if available)      — what run_edge_pipeline.py uses today
  - ONNX Runtime FP32 (CPU)                          — export correctness baseline
  - ONNX Runtime INT8 dynamic-quantized (CPU)        — the actual onboard-deployable target

Exports whichever of {cloud+event separate, fusion} has trained weights available.

    python models/export_onnx.py --n_runs 200

Writes .onnx / .int8.onnx files next to the .pth checkpoints in models/weights/, and
prints a latency comparison table.
"""
import argparse
import os
import time

import numpy as np
import torch

from cloud_classifier import TinyCloudNet, CLOUD_CLASSES
from event_detector import TinyEventNet, EVENT_CLASSES
from fusion_net import OrbitFusionNet

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
DUMMY_INPUT = torch.randn(1, 3, 128, 128)


def export_and_quantize(model: torch.nn.Module, name: str):
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model.eval()
    fp32_path = os.path.join(WEIGHTS_DIR, f"{name}.onnx")
    int8_path = os.path.join(WEIGHTS_DIR, f"{name}.int8.onnx")

    with torch.no_grad():
        dummy_out = model(DUMMY_INPUT)
    multi_output = isinstance(dummy_out, tuple)
    output_names = ["cloud_logits", "event_logits"] if multi_output else ["output"]
    dynamic_axes = {"input": {0: "batch"}}
    for n in output_names:
        dynamic_axes[n] = {0: "batch"}

    torch.onnx.export(
        model, DUMMY_INPUT, fp32_path,
        input_names=["input"], output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        dynamo=False,  # the new dynamo exporter's graph shape trips up onnxruntime's
                        # static quantizer (shape-inference mismatch); the legacy
                        # TorchScript-based exporter produces a graph it handles fine
    )
    onnx.checker.check_model(fp32_path)

    try:
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    except Exception as e:
        print(f"  {name}: INT8 quantization failed ({type(e).__name__}: {e}) — "
              f"reporting fp32 numbers only for this model.")
        return fp32_path, None

    fp32_size = os.path.getsize(fp32_path) / 1024
    int8_size = os.path.getsize(int8_path) / 1024
    print(f"  {name}: {fp32_size:.1f}KB fp32 -> {int8_size:.1f}KB int8 "
          f"({fp32_size / int8_size:.2f}x smaller)")
    return fp32_path, int8_path


def bench_pytorch(model, device, n_runs):
    model = model.to(device).eval()
    x = DUMMY_INPUT.to(device)
    with torch.no_grad():
        for _ in range(10):  # warmup
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed / n_runs * 1000.0  # ms/frame


def bench_onnx(path, n_runs):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = DUMMY_INPUT.numpy()
    for _ in range(10):
        sess.run(None, {"input": x})
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {"input": x})
    elapsed = time.perf_counter() - t0
    return elapsed / n_runs * 1000.0


def run_for(name, model, n_runs):
    print(f"\n=== {name} ===")
    fp32_path, int8_path = export_and_quantize(model, name)

    results = {}
    results["pytorch_cpu"] = bench_pytorch(model, "cpu", n_runs)
    if torch.cuda.is_available():
        results["pytorch_cuda"] = bench_pytorch(model, "cuda", n_runs)
    results["onnx_fp32_cpu"] = bench_onnx(fp32_path, n_runs)
    if int8_path is not None:
        results["onnx_int8_cpu"] = bench_onnx(int8_path, n_runs)

    for k, v in results.items():
        print(f"  {k:16s}: {v:6.3f} ms/frame  ({1000/v:7.1f} fps)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=200)
    args = parser.parse_args()

    fusion_path = os.path.join(WEIGHTS_DIR, "fusion_net.pth")
    cloud_path = os.path.join(WEIGHTS_DIR, "cloud_classifier.pth")
    event_path = os.path.join(WEIGHTS_DIR, "event_detector.pth")

    all_results = {}

    if os.path.exists(fusion_path):
        model = OrbitFusionNet(len(CLOUD_CLASSES), len(EVENT_CLASSES))
        model.load_state_dict(torch.load(fusion_path, map_location="cpu")["model_state_dict"])
        all_results["fusion_net (cloud+event, 1 pass)"] = run_for("fusion_net", model, args.n_runs)

    if os.path.exists(cloud_path) and os.path.exists(event_path):
        cloud_model = TinyCloudNet(len(CLOUD_CLASSES))
        cloud_model.load_state_dict(torch.load(cloud_path, map_location="cpu")["model_state_dict"])
        event_model = TinyEventNet(len(EVENT_CLASSES))
        event_model.load_state_dict(torch.load(event_path, map_location="cpu")["model_state_dict"])

        cloud_results = run_for("cloud_classifier", cloud_model, args.n_runs)
        event_results = run_for("event_detector", event_model, args.n_runs)
        combined = {k: cloud_results[k] + event_results.get(k, 0) for k in cloud_results}
        all_results["separate models (cloud + event, 2 passes)"] = combined
        print("\n=== combined separate-model latency (cloud + event back to back) ===")
        for k, v in combined.items():
            print(f"  {k:16s}: {v:6.3f} ms/frame  ({1000/v:7.1f} fps)")

    if "fusion_net (cloud+event, 1 pass)" in all_results and \
            "separate models (cloud + event, 2 passes)" in all_results:
        fusion_results = all_results["fusion_net (cloud+event, 1 pass)"]
        separate_results = all_results["separate models (cloud + event, 2 passes)"]
        compare_key = "onnx_int8_cpu" if "onnx_int8_cpu" in fusion_results else "onnx_fp32_cpu"
        print(f"\n=== fusion vs. separate, {compare_key} ===")
        fused = fusion_results[compare_key]
        separate = separate_results[compare_key]
        print(f"  fusion:   {fused:.3f} ms/frame")
        print(f"  separate: {separate:.3f} ms/frame")
        print(f"  speedup:  {separate / fused:.2f}x")

    if not all_results:
        print("No trained weights found in models/weights/ — train the models first.")


if __name__ == "__main__":
    main()
