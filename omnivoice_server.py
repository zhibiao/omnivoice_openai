#!/usr/bin/env python3
"""OpenAI-compatible OmniVoice TTS server with startup-loaded voice presets."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import io
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

LOG = logging.getLogger("omnivoice-api")

_PKG_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = str(_PKG_DIR / "models" / "OmniVoice")
DEFAULT_VOICES_DIR = str(_PKG_DIR / "voices")
DEFAULT_CACHE_DIR = str(_PKG_DIR / ".inductor-cache")
SAMPLE_RATE = 24000
SUPPORTED_FORMATS = ("wav", "pcm")
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}
MAX_REF_BYTES = 30 * 1024 * 1024


def _load_implementation():
    """Load the sibling omnivoice.py without resolving the legacy package."""
    module_name = "_omnivoice_standalone"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    path = _PKG_DIR / "omnivoice.py"
    if not path.is_file():
        raise RuntimeError(f"OmniVoice implementation not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load OmniVoice implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


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


_IMPLEMENTATION = _load_implementation()
_OMNIVOICE = _IMPLEMENTATION.OmniVoice
_GENERATION_CONFIG = _IMPLEMENTATION.OmniVoiceGenerationConfig


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: Optional[dict] = None):
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}
        super().__init__(message)


def error_body(message: str, code: str, type_: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": type_, "param": None, "code": code}}


def decode_audio(raw: bytes) -> np.ndarray:
    if len(raw) > MAX_REF_BYTES:
        raise ApiError(413, "payload_too_large", f"reference audio exceeds {MAX_REF_BYTES // 1024 // 1024} MB")
    try:
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ApiError(
            400,
            "invalid_audio",
            f"cannot decode reference audio ({exc}); supported: wav/flac/mp3/ogg",
        ) from exc
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import torchaudio

        wav = (
            torchaudio.functional.resample(torch.from_numpy(wav), int(sr), SAMPLE_RATE)
            .clamp(-1.0, 1.0)
            .numpy()
            .astype(np.float32)
        )
    if wav.size < SAMPLE_RATE // 2:
        raise ApiError(400, "audio_too_short", "reference audio must be at least 0.5 s")
    return wav


def encode_wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def encode_pcm(audio: np.ndarray) -> bytes:
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes()


@dataclass(frozen=True)
class ReferenceSpec:
    name: str
    audio_path: Path
    ref_text: str


def discover_references(root: str, default_voice: str) -> tuple[list[ReferenceSpec], str]:
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise RuntimeError(f"reference directory does not exist: {root_path}")

    audio_by_name: dict[str, Path] = {}
    text_by_name: dict[str, Path] = {}
    for path in sorted(root_path.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix in AUDIO_EXTENSIONS:
            if path.stem in audio_by_name:
                raise RuntimeError(
                    f"multiple reference audio files use the name '{path.stem}': "
                    f"{audio_by_name[path.stem].name}, {path.name}"
                )
            audio_by_name[path.stem] = path
        elif suffix == ".txt":
            if path.stem in text_by_name:
                raise RuntimeError(
                    f"multiple reference text files use the name '{path.stem}': "
                    f"{text_by_name[path.stem].name}, {path.name}"
                )
            text_by_name[path.stem] = path

    audio_only = sorted(set(audio_by_name) - set(text_by_name))
    text_only = sorted(set(text_by_name) - set(audio_by_name))
    if audio_only:
        raise RuntimeError(f"missing reference text for: {', '.join(audio_only)}")
    if text_only:
        raise RuntimeError(f"missing reference audio for: {', '.join(text_only)}")
    if not audio_by_name:
        raise RuntimeError(
            f"no reference pairs found in {root_path}; expected <name>.wav + <name>.txt"
        )

    specs = []
    for name in sorted(audio_by_name):
        if re.fullmatch(r"[\w-]+", name) is None:
            raise RuntimeError(
                f"invalid reference name '{name}'; use letters, numbers, Chinese characters, "
                "underscores and hyphens only"
            )
        text_path = text_by_name[name]
        try:
            ref_text = text_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"cannot read reference text {text_path}: {exc}") from exc
        if not ref_text:
            raise RuntimeError(f"reference text is empty: {text_path}")
        specs.append(ReferenceSpec(name, audio_by_name[name], ref_text))

    requested = (default_voice or "").strip()
    if requested:
        if requested not in audio_by_name:
            raise RuntimeError(
                f"default voice '{requested}' was not found; available: {', '.join(sorted(audio_by_name))}"
            )
        resolved_default = requested
    elif len(specs) == 1:
        resolved_default = specs[0].name
    else:
        raise RuntimeError(
            "multiple reference presets were found; pass --default-voice with one of: "
            f"{', '.join(spec.name for spec in specs)}"
        )
    return specs, resolved_default


class PresetVoiceRegistry:
    def __init__(
        self,
        specs: list[ReferenceSpec],
        engine: "Engine",
        default_name: str,
    ):
        self.default_name = default_name
        self._prompts: dict[str, Any] = {}
        self._durations: dict[str, float] = {}

        started = time.perf_counter()
        for spec in specs:
            try:
                wav = decode_audio(spec.audio_path.read_bytes())
            except OSError as exc:
                raise RuntimeError(f"cannot read reference audio {spec.audio_path}: {exc}") from exc
            prompt = engine.build_prompt(wav, spec.ref_text)
            prompt.ref_audio_tokens = prompt.ref_audio_tokens.cpu()
            self._prompts[spec.name] = prompt
            duration = len(wav) / SAMPLE_RATE
            self._durations[spec.name] = duration
            LOG.info(
                "loaded preset %s (%.1fs audio, %d chars ref_text)",
                spec.name,
                duration,
                len(spec.ref_text),
            )
        LOG.info(
            "preprocessed %d preset(s) from %s in %.2fs (default=%s)",
            len(self._prompts),
            specs[0].audio_path.parent,
            time.perf_counter() - started,
            self.default_name,
        )

    def get(self, name: Optional[str]):
        resolved = (name or "").strip() or self.default_name
        prompt = self._prompts.get(resolved)
        if prompt is None:
            raise ApiError(
                404,
                "preset_not_found",
                f"preset '{resolved}' does not exist",
            )
        return resolved, prompt

    def list(self) -> list[dict]:
        names = sorted(self._prompts, key=lambda name: (name != self.default_name, name))
        return [
            {
                "id": name,
                "object": "voice",
                "name": name,
                "duration": round(self._durations[name], 3),
                "default": name == self.default_name,
            }
            for name in names
        ]

    def __len__(self) -> int:
        return len(self._prompts)


class Engine:
    def __init__(self, args: argparse.Namespace):
        if not torch.cuda.is_available():
            sys.exit("[fatal] CUDA is required, but torch.cuda.is_available() is False")
        if not str(args.device).startswith("cuda"):
            sys.exit(f"[fatal] only CUDA devices are supported, got --device {args.device}")

        self.args = args
        self.device = args.device
        LOG.info("loading model from %s on %s (float16) ...", args.model, args.device)
        t0 = time.perf_counter()
        self.model = _OMNIVOICE.from_pretrained(
            args.model, device_map=args.device, dtype=torch.float16, load_asr=False
        )
        self.model.eval()
        LOG.info("model loaded in %.1fs", time.perf_counter() - t0)

        if getattr(self.model, "_asr_pipe", None) is not None:
            sys.exit("[fatal] ASR pipeline unexpectedly loaded; this server must not use Whisper")

        self.flashinfer = self._enable_flashinfer()
        self.compile_mode = "off" if self.flashinfer else args.compile
        if self.flashinfer:
            if args.compile != "off":
                LOG.info("FlashInfer enabled; skipping torch.compile(model.llm)")
        elif self.compile_mode != "off":
            LOG.info("torch.compile(model.llm, dynamic=%s) ...", args.compile == "dynamic")
            self.model.llm = torch.compile(self.model.llm, dynamic=(args.compile == "dynamic"))

        self.backend = (
            "flashinfer"
            if self.flashinfer
            else (f"torch.compile:{self.compile_mode}" if self.compile_mode != "off" else "eager")
        )
        self.sr = int(self.model.sampling_rate)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omnivoice-gpu")
        self._warmup()

    def _enable_flashinfer(self) -> bool:
        mode = self.args.flashinfer
        if mode == "off":
            return False

        try:
            flashinfer_ext = _load_flashinfer()
        except ImportError as exc:
            if mode == "on":
                raise RuntimeError(
                    "--flashinfer on requires flashinfer-python and "
                    "flashinfer-jit-cache to be installed"
                ) from exc
            LOG.warning("FlashInfer unavailable; falling back: %s", exc)
            return False

        LOG.info("enabling FlashInfer packed decoding and fused kernels ...")
        t0 = time.perf_counter()
        flashinfer_ext.apply_flashinfer(self.model)
        LOG.info("FlashInfer enabled in %.1fs", time.perf_counter() - t0)
        return True

    def build_prompt(self, wav: np.ndarray, ref_text: str):
        with torch.inference_mode():
            return self.model.create_voice_clone_prompt(
                ref_audio=(wav[np.newaxis, :], SAMPLE_RATE),
                ref_text=ref_text,
                preprocess_prompt=True,
            )

    def synth(
        self,
        text: str,
        prompt,
        num_step: int,
        guidance_scale: float,
        speed: float,
        language: Optional[str],
    ) -> np.ndarray:
        cfg = _GENERATION_CONFIG(num_step=num_step, guidance_scale=guidance_scale)
        with torch.inference_mode():
            audios = self.model.generate(
                text=text,
                voice_clone_prompt=prompt,
                language=language,
                speed=speed,
                generation_config=cfg,
                normalize_text=True,
            )
        return audios[0]

    def _warmup(self) -> None:
        steps = min(self.args.num_step, 8)
        samples = [
            "你好，这是服务预热。",
            "语音合成技术近年来进步非常明显，尤其是在零样本音色克隆方面。",
        ]
        for text in samples:
            t0 = time.perf_counter()
            try:
                self.synth(text, None, steps, 2.0, 1.0, None)
                LOG.info("warmup ok (%.1fs): %s", time.perf_counter() - t0, text[:20])
            except Exception as exc:
                if self.flashinfer:
                    raise RuntimeError(
                        "FlashInfer warmup failed; refusing to start a broken backend"
                    ) from exc
                LOG.warning("warmup failed (%s): %s", type(exc).__name__, exc)

    def health(self) -> dict:
        free, total = torch.cuda.mem_get_info()
        return {
            "status": "ok",
            "device": self.device,
            "dtype": "float16",
            "backend": self.backend,
            "flashinfer": self.flashinfer,
            "compile": self.compile_mode,
            "num_step_default": self.args.num_step,
            "model": str(self.args.model),
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "total_gb": round(total / 1e9, 2),
                "free_gb": round(free / 1e9, 2),
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            },
        }


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = "omnivoice"
    input: str = ""
    voice: Optional[str] = None
    response_format: Optional[str] = None
    speed: Optional[float] = None
    num_step: Optional[int] = None
    guidance_scale: Optional[float] = None
    language: Optional[str] = None


ARGS: argparse.Namespace
ENGINE: Engine
PRESETS: PresetVoiceRegistry
GPU_SEM: asyncio.Semaphore
PENDING = [0]
STARTED_AT = time.time()

app = FastAPI(title="语音合成 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Audio-Duration", "X-RTF", "X-Num-Step"],
)


@app.exception_handler(ApiError)
async def _api_error_handler(request: Request, exc: ApiError):
    type_ = "server_error" if exc.status >= 500 else "invalid_request_error"
    return JSONResponse(status_code=exc.status, content=error_body(exc.message, exc.code, type_), headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content=error_body(f"invalid request body: {exc.errors()}", "invalid_request"))


def check_auth(authorization: Optional[str]) -> None:
    if not ARGS.api_key:
        return
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token != ARGS.api_key:
        raise ApiError(401, "invalid_api_key", "invalid or missing API key")


async def run_gpu(fn):
    if PENDING[0] >= ARGS.max_queue:
        raise ApiError(
            429,
            "queue_full",
            f"too many pending requests (max {ARGS.max_queue}); retry later",
            headers={"Retry-After": "5"},
        )
    PENDING[0] += 1
    try:
        await GPU_SEM.acquire()
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(ENGINE.executor, fn)
        try:
            result = await asyncio.wait_for(asyncio.shield(fut), timeout=ARGS.timeout)
        except asyncio.TimeoutError:
            fut.add_done_callback(lambda _f: loop.call_soon_threadsafe(GPU_SEM.release))
            raise ApiError(504, "timeout", f"generation exceeded {ARGS.timeout}s")
        except BaseException:
            GPU_SEM.release()
            raise
        GPU_SEM.release()
        return result
    finally:
        PENDING[0] -= 1


@app.get("/health")
async def health():
    data = ENGINE.health()
    data["queue"] = {"pending": PENDING[0], "max": ARGS.max_queue}
    data["uptime_s"] = round(time.time() - STARTED_AT, 1)
    return data


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(default=None)):
    check_auth(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": "omnivoice",
                "object": "model",
                "created": int(STARTED_AT),
                "owned_by": "local",
                "num_step": ARGS.num_step,
            }
        ],
    }


@app.get("/v1/audio/voices")
async def list_voices(authorization: Optional[str] = Header(default=None)):
    check_auth(authorization)
    return {"object": "list", "data": PRESETS.list()}


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest, authorization: Optional[str] = Header(default=None)):
    check_auth(authorization)

    model_name = (req.model or "").strip() or "omnivoice"

    text = (req.input or "").strip()
    if not text:
        raise ApiError(400, "input_required", "'input' must not be empty")
    if len(text) > ARGS.max_chars:
        raise ApiError(413, "input_too_long", f"'input' exceeds {ARGS.max_chars} characters")

    fmt = (req.response_format or ARGS.format).lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ApiError(
            400,
            "unsupported_format",
            f"response_format '{fmt}' is not supported; use one of: {', '.join(SUPPORTED_FORMATS)}",
        )

    num_step = req.num_step if req.num_step is not None else ARGS.num_step
    if not (4 <= num_step <= 64):
        raise ApiError(400, "invalid_num_step", "num_step must be between 4 and 64")

    guidance = req.guidance_scale if req.guidance_scale is not None else 2.0
    speed = req.speed if req.speed is not None else 1.0
    if not (0.25 <= speed <= 4.0):
        raise ApiError(400, "invalid_speed", "speed must be between 0.25 and 4.0")

    voice_name, prompt = PRESETS.get(req.voice)
    started = time.perf_counter()

    audio = await run_gpu(
        lambda: ENGINE.synth(text, prompt, num_step, guidance, speed, req.language)
    )
    took = time.perf_counter() - started
    duration = len(audio) / ENGINE.sr
    rtf = took / duration if duration > 0 else 0.0

    LOG.info(
        "[speech] model=%s voice=%s chars=%d num_step=%d audio=%.2fs took=%.2fs rtf=%.4f format=%s",
        model_name, voice_name, len(text), num_step, duration, took, rtf, fmt,
    )

    body = encode_wav(audio) if fmt == "wav" else encode_pcm(audio)
    media = "audio/wav" if fmt == "wav" else "audio/L16"
    headers = {
        "X-Audio-Duration": f"{duration:.3f}",
        "X-RTF": f"{rtf:.4f}",
        "X-Num-Step": str(num_step),
        "Cache-Control": "no-store",
    }
    return Response(content=body, media_type=media, headers=headers)


WEB_DIR = Path(__file__).resolve().parent / "web"


@app.get("/", include_in_schema=False)
async def index_redirect():
    return RedirectResponse(url="/web/")


if WEB_DIR.is_dir():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OmniVoice OpenAI-compatible TTS server (GPU, fp16, FlashInfer, voice cloning)"
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint path or HF repo id")
    p.add_argument("--device", default="cuda:0", help="cuda / cuda:N (CPU is not supported)")
    p.add_argument("--compile", default="dynamic", choices=["off", "dynamic", "static"],
                   help="torch.compile mode for the LLM backbone (default: dynamic)")
    p.add_argument(
        "--flashinfer",
        default="auto",
        choices=["auto", "on", "off"],
        help="FlashInfer backend: auto enables it when installed, on requires it, "
        "off disables it (default: auto). FlashInfer takes precedence over --compile.",
    )
    p.add_argument("--num-step", type=int, default=32, dest="num_step", help="default diffusion steps (16 or 32)")
    p.add_argument("--format", default="wav", choices=list(SUPPORTED_FORMATS), help="default response format")
    p.add_argument("--max-queue", type=int, default=8, dest="max_queue", help="max pending requests before 429")
    p.add_argument("--timeout", type=float, default=180.0, help="per-request timeout in seconds")
    p.add_argument("--max-chars", type=int, default=2000, dest="max_chars")
    p.add_argument(
        "--voices-dir",
        default=DEFAULT_VOICES_DIR,
        dest="voices_dir",
        help="directory containing <name>.wav/flac/mp3/ogg + <name>.txt reference pairs",
    )
    p.add_argument(
        "--default-voice",
        default="nana",
        dest="default_voice",
        help="default preset name used when /v1/audio/speech omits voice",
    )
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, dest="cache_dir")
    p.add_argument("--api-key", default="", dest="api_key")
    p.add_argument("--log-level", default="info", dest="log_level")
    return p.parse_args()


def main() -> None:
    global ARGS, ENGINE, PRESETS, GPU_SEM

    args = parse_args()
    ARGS = args
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", args.cache_dir)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    STARTED = time.time()
    try:
        reference_specs, default_name = discover_references(
            args.voices_dir,
            args.default_voice,
        )
    except RuntimeError as exc:
        raise SystemExit(f"[fatal] {exc}") from exc

    ENGINE = Engine(args)
    try:
        PRESETS = PresetVoiceRegistry(reference_specs, ENGINE, default_name)
    except (ApiError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"[fatal] failed to preprocess reference presets: {exc}") from exc
    GPU_SEM = asyncio.Semaphore(1)

    LOG.info(
        "ready in %.1fs | device=%s dtype=float16 backend=%s num_step=%d format=%s presets=%d default=%s",
        time.time() - STARTED,
        args.device,
        ENGINE.backend,
        args.num_step,
        args.format,
        len(PRESETS),
        PRESETS.default_name,
    )
    LOG.info("listening on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, access_log=False)


if __name__ == "__main__":
    main()
