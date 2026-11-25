#!/usr/bin/env python3
"""OmniVoice command-line tools for creating prompts and generated speech."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import sys
from pathlib import Path

import soundfile as sf
import torch


_PKG_DIR = Path(__file__).resolve().parent
_IMPLEMENTATION_NAME = "_omnivoice_standalone"
_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}
_MAX_REF_BYTES = 30 * 1024 * 1024


def _load_implementation():
    """Load the sibling omnivoice.py without relying on package resolution."""
    module = sys.modules.get(_IMPLEMENTATION_NAME)
    if module is not None:
        return module

    path = _PKG_DIR / "omnivoice.py"
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
VoiceClonePrompt = _impl.VoiceClonePrompt


def _load_flashinfer():
    """Load the sibling FlashInfer extension without using the legacy package."""
    module_name = "_omnivoice_standalone_flashinfer"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    path = _PKG_DIR / "omnivoice_flashinfer.py"
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


def _resolve_dtype(args, device: str) -> torch.dtype:
    if args.dtype is not None:
        return getattr(torch, args.dtype)
    if str(device).startswith(("cuda", "xpu")):
        return torch.float16
    return torch.float32


def _add_model_args(parser: argparse.ArgumentParser, include_backend=False) -> None:
    parser.add_argument(
        "--model",
        default=str(_PKG_DIR / "models" / "OmniVoice"),
        help="Local checkpoint path or Hugging Face repository ID.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float16", "bfloat16", "float32"],
    )
    if include_backend:
        parser.add_argument(
            "--flashinfer",
            default="auto",
            choices=["auto", "on", "off"],
            help="Use FlashInfer when available.",
        )
        parser.add_argument(
            "--cuda-graph",
            action="store_true",
            help="Capture FlashInfer forwards with CUDA graphs.",
        )


def discover_references(root: Path) -> list[tuple[str, Path, str]]:
    if not root.is_dir():
        raise RuntimeError(f"assets directory does not exist: {root}")

    audio_by_name: dict[str, Path] = {}
    text_by_name: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix in _AUDIO_EXTENSIONS:
            if path.stem in audio_by_name:
                raise RuntimeError(f"duplicate reference audio: {path.stem}")
            audio_by_name[path.stem] = path
        elif suffix == ".txt":
            if path.stem in text_by_name:
                raise RuntimeError(f"duplicate reference text: {path.stem}")
            text_by_name[path.stem] = path

    audio_only = sorted(set(audio_by_name) - set(text_by_name))
    text_only = sorted(set(text_by_name) - set(audio_by_name))
    if audio_only:
        raise RuntimeError(f"missing reference text for: {', '.join(audio_only)}")
    if text_only:
        raise RuntimeError(f"missing reference audio for: {', '.join(text_only)}")
    if not audio_by_name:
        raise RuntimeError(f"no reference pairs found in {root}")

    references = []
    for name in sorted(audio_by_name):
        if re.fullmatch(r"[\w-]+", name) is None:
            raise RuntimeError(f"invalid voice name: {name}")
        text_path = text_by_name[name]
        try:
            ref_text = text_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"cannot read reference text {text_path}: {exc}") from exc
        if not ref_text:
            raise RuntimeError(f"reference text is empty: {text_path}")
        references.append((name, audio_by_name[name], ref_text))
    return references


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create OmniVoice prompts or generate cloned speech.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser(
        "create-prompt",
        help="Create voices/<name>.pt from assets/<name> audio and text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_model_args(create)
    create.add_argument(
        "--assets-dir",
        default=str(_PKG_DIR / "assets"),
        help="Directory containing <name> audio files and <name>.txt.",
    )
    create.add_argument(
        "--voices-dir",
        default=str(_PKG_DIR / "voices"),
        help="Output directory for <name>.pt prompt files.",
    )

    clone = subparsers.add_parser(
        "clone",
        help="Generate speech with a saved prompt or reference audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_model_args(clone, include_backend=True)
    clone.add_argument("--text", required=True)
    clone.add_argument("--output", required=True)
    clone.add_argument("--prompt", default=None, help="Saved voices/<name>.pt file.")
    clone.add_argument("--ref_audio", default=None)
    clone.add_argument("--ref_text", default=None)
    clone.add_argument("--instruct", default=None)
    clone.add_argument("--language", default=None)
    clone.add_argument("--num_step", type=int, default=32)
    clone.add_argument("--guidance_scale", type=float, default=2.0)
    clone.add_argument("--speed", type=float, default=1.0)
    clone.add_argument("--duration", type=float, default=None)
    clone.add_argument("--t_shift", type=float, default=0.1)
    clone.add_argument("--denoise", type=str2bool, default=True)
    clone.add_argument("--postprocess_output", type=str2bool, default=True)
    clone.add_argument("--layer_penalty_factor", type=float, default=5.0)
    clone.add_argument("--position_temperature", type=float, default=5.0)
    clone.add_argument("--class_temperature", type=float, default=0.0)
    clone.add_argument(
        "--normalize_text",
        "--text_normal",
        dest="normalize_text",
        type=str2bool,
        default=True,
    )
    return parser


def _load_model(args):
    device = args.device or get_best_device()
    dtype = _resolve_dtype(args, device)
    logging.info("Loading model from %s on %s (%s)", args.model, device, dtype)
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=dtype,
        load_asr=False,
    )
    return model.eval()


def _enable_flashinfer(model, args) -> None:
    if args.flashinfer == "off":
        return
    if model.device.type != "cuda":
        if args.flashinfer == "on":
            raise RuntimeError("--flashinfer on requires a CUDA device")
        logging.warning("FlashInfer requires CUDA; using standard attention")
        return

    try:
        flashinfer = _load_flashinfer()
    except ImportError as exc:
        if args.flashinfer == "on":
            raise RuntimeError(
                "--flashinfer on requires flashinfer-python "
                "and flashinfer-jit-cache"
            ) from exc
        logging.warning("FlashInfer unavailable; using standard attention: %s", exc)
        return

    flashinfer.apply_flashinfer(
        model,
        enable_cuda_graph=args.cuda_graph,
    )
    logging.info("FlashInfer enabled (cuda_graph=%s)", args.cuda_graph)


def create_prompts(args) -> None:
    assets_dir = Path(args.assets_dir).expanduser().resolve()
    voices_dir = Path(args.voices_dir).expanduser().resolve()
    references = discover_references(assets_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(args)

    for name, audio_path, ref_text in references:
        if audio_path.stat().st_size > _MAX_REF_BYTES:
            raise RuntimeError(f"reference audio exceeds 30 MB: {audio_path}")
        wav = _impl.load_audio(str(audio_path), model.sampling_rate)
        duration = wav.shape[-1] / model.sampling_rate
        if duration < 0.5:
            raise RuntimeError(f"reference audio is shorter than 0.5 seconds: {audio_path}")

        with torch.inference_mode():
            prompt = model.create_voice_clone_prompt(
                ref_audio=(wav, model.sampling_rate),
                ref_text=ref_text,
                preprocess_prompt=True,
            )
        prompt.ref_audio_tokens = prompt.ref_audio_tokens.cpu()
        output_path = voices_dir / f"{name}.pt"
        prompt.save(str(output_path))
        logging.info(
            "Created %s (%.2fs, tokens=%s)",
            output_path,
            duration,
            tuple(prompt.ref_audio_tokens.shape),
        )


def clone(args) -> None:
    if args.prompt and (args.ref_audio or args.ref_text):
        raise RuntimeError("use either --prompt or --ref_audio/--ref_text")
    if not args.prompt and not (args.ref_audio and args.ref_text):
        raise RuntimeError("clone requires --prompt or --ref_audio with --ref_text")
    if bool(args.ref_audio) != bool(args.ref_text):
        raise RuntimeError("--ref_audio and --ref_text must be provided together")

    model = _load_model(args)
    _enable_flashinfer(model, args)

    prompt = None
    if args.prompt:
        prompt_path = Path(args.prompt).expanduser()
        if prompt_path.suffix.lower() != ".pt":
            raise ValueError("voice prompt must use the .pt extension")
        prompt = VoiceClonePrompt.load(str(prompt_path), map_location="cpu")

    audios = model.generate(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        voice_clone_prompt=prompt,
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


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        force=True,
    )
    if argv is None:
        argv = sys.argv[1:]
    command_names = {"create-prompt", "clone", "-h", "--help"}
    if argv and argv[0] not in command_names:
        argv = ["clone", *argv]

    args = build_parser().parse_args(argv)
    if args.command == "create-prompt":
        create_prompts(args)
    else:
        clone(args)


if __name__ == "__main__":
    main()
