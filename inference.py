#!/usr/bin/env python3
"""Command-line inference for the standalone OmniVoice implementation."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import soundfile as sf
import torch


_IMPLEMENTATION_NAME = "_omnivoice_standalone"


def _load_implementation():
    """Load the sibling omnivoice.py without resolving the omnivoice package."""
    module = sys.modules.get(_IMPLEMENTATION_NAME)
    if module is not None:
        return module

    path = Path(__file__).resolve().with_name("omnivoice.py")
    if not path.is_file():
        raise RuntimeError(f"OmniVoice implementation not found: {path}")

    spec = importlib.util.spec_from_file_location(_IMPLEMENTATION_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load OmniVoice implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_IMPLEMENTATION_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_IMPLEMENTATION_NAME, None)
        raise
    return module


_impl = _load_implementation()
OmniVoice = _impl.OmniVoice


def _load_flashinfer():
    """Load the sibling FlashInfer extension without using the legacy package."""
    module_name = "_omnivoice_standalone_flashinfer"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    path = Path(__file__).resolve().with_name("omnivoice_flashinfer.py")
    if not path.is_file():
        raise RuntimeError(f"FlashInfer extension not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load FlashInfer extension from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def str2bool(value):
    """Parse a command-line boolean."""
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("yes", "true", "t", "y", "1", "on"):
        return True
    if lowered in ("no", "false", "f", "n", "0", "off"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_best_device():
    """Auto-detect the best available inference device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone OmniVoice inference (native-equivalent)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "models" / "OmniVoice"),
        help="Local checkpoint path or Hugging Face repository id.",
    )
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--output", required=True, help="Output WAV path.")
    parser.add_argument("--ref_audio", default=None, help="Reference audio path.")
    parser.add_argument("--ref_text", default=None, help="Reference transcript.")
    parser.add_argument("--instruct", default=None, help="Voice-design instruction.")
    parser.add_argument("--language", default=None, help="Language name or code.")
    parser.add_argument("--num_step", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--denoise", type=str2bool, default=True)
    parser.add_argument("--postprocess_output", type=str2bool, default=True)
    parser.add_argument("--layer_penalty_factor", type=float, default=5.0)
    parser.add_argument("--position_temperature", type=float, default=5.0)
    parser.add_argument("--class_temperature", type=float, default=0.0)
    parser.add_argument(
        "--normalize_text",
        "--text_normal",
        dest="normalize_text",
        type=str2bool,
        default=True,
        help="Normalize numbers, dates, and currency before synthesis.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--flashinfer",
        default="auto",
        choices=["auto", "on", "off"],
        help="Use FlashInfer packed attention when available.",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture FlashInfer forwards with CUDA graphs.",
    )
    return parser


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        force=True,
    )
    args = build_parser().parse_args(argv)
    device = args.device or get_best_device()
    dtype = getattr(torch, args.dtype)
    logging.info("Loading model from %s on %s (%s) ...", args.model, device, args.dtype)
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=dtype,
        load_asr=False,
    )

    if args.flashinfer != "off" and model.device.type != "cuda":
        if args.flashinfer == "on":
            raise RuntimeError("--flashinfer on requires a CUDA device")
        logging.warning("FlashInfer requires CUDA; using standard attention")
    elif args.flashinfer != "off":
        try:
            flashinfer_ext = _load_flashinfer()
        except ImportError as exc:
            if args.flashinfer == "on":
                raise RuntimeError(
                    "--flashinfer on requires flashinfer-python and "
                    "flashinfer-jit-cache to be installed"
                ) from exc
            logging.warning("FlashInfer unavailable; using standard attention: %s", exc)
        else:
            flashinfer_ext.apply_flashinfer(
                model,
                enable_cuda_graph=args.cuda_graph,
            )
            logging.info(
                "FlashInfer enabled (cuda_graph=%s)", args.cuda_graph
            )

    logging.info("Generating audio for: %s", args.text[:80])
    audios = model.generate(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        instruct=args.instruct,
        duration=args.duration,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        speed=args.speed,
        t_shift=args.t_shift,
        denoise=args.denoise,
        postprocess_output=args.postprocess_output,
        layer_penalty_factor=args.layer_penalty_factor,
        position_temperature=args.position_temperature,
        class_temperature=args.class_temperature,
        normalize_text=args.normalize_text,
    )
    sf.write(args.output, audios[0], model.sampling_rate)
    logging.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
