from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import torch
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}\nRun: pip install -r requirements.txt")
    sys.exit(1)

import config
import utils

logger = utils.setup_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a trained checkpoint for deployment.")
    p.add_argument("--checkpoint", type=str, default=str(config.BEST_MODEL_PATH))
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--formats", nargs="+", default=["state_dict", "torchscript", "onnx"],
                   choices=["state_dict", "torchscript", "onnx"])
    p.add_argument("--out-dir", type=str, default=str(config.EXPORT_DIR))
    return p.parse_args()


def human_size(path: Path) -> str:
    size_bytes = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def _verify_onnx(onnx_path: Path, torch_model, dummy_input) -> None:
    import numpy as np
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = torch_model(dummy_input).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy_input.cpu().numpy()})[0]

    if np.allclose(torch_out, onnx_out, atol=1e-3):
        logger.info("[onnx] verification PASSED — ONNX Runtime output matches PyTorch within tolerance.")
    else:
        max_diff = np.abs(torch_out - onnx_out).max()
        logger.warning(f"[onnx] verification WARNING — max abs diff {max_diff:.6f} exceeds tolerance.")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    try:
        model, model_name, class_names = utils.load_model_for_inference(Path(args.checkpoint), device)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Loaded '{model_name}' from {args.checkpoint}")
    dummy_input = torch.randn(1, config.IN_CHANNELS, args.img_size, args.img_size, device=device)

    if "state_dict" in args.formats:
        path = out_dir / f"{model_name}_weights.pth"
        torch.save({"model_state_dict": model.state_dict(), "model_name": model_name,
                    "img_size": args.img_size, "class_names": class_names}, path)
        logger.info(f"[state_dict] saved -> {path} ({human_size(path)})")

    if "torchscript" in args.formats:
        try:
            traced = torch.jit.trace(model, dummy_input)
            path = out_dir / f"{model_name}_torchscript.pt"
            traced.save(str(path))
            logger.info(f"[torchscript] saved -> {path} ({human_size(path)})")
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")

    if "onnx" in args.formats:
        try:
            path = out_dir / f"{model_name}.onnx"
            torch.onnx.export(
                model,
                dummy_input,
                str(path),
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
                opset_version=17,
            )
            logger.info(f"[onnx] saved -> {path} ({human_size(path)})")
            _verify_onnx(path, model, dummy_input)
        except ImportError:
            logger.warning("onnx / onnxruntime not installed — skipping ONNX export and verification.")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")

    logger.info(f"All exports written to {out_dir}")


if __name__ == "__main__":
    main()
