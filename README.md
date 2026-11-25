# OmniVoice OpenAI API

Standalone OmniVoice text-to-speech service with an OpenAI-compatible API.

## Requirements

- Linux
- Python 3.10+
- NVIDIA GPU with CUDA
- Model files in `models/OmniVoice/`
- Source audio and UTF-8 transcripts in `assets/`

## Install

```bash
uv sync
```

Optional FlashInfer support:

```bash
uv sync --extra flashinfer
```

## Create a voice

Reference audio is normally kept in `assets/`. The transcript may be provided
as a `.txt` file or as literal text:

```text
assets/
├── nana.wav
└── nana.txt
```

Supported audio formats: `wav`, `flac`, `mp3`, and `ogg`.

Create or rebuild one voice prompt:

```bash
.venv/bin/python omnivoice_cli.py create-prompt \
  --model models/OmniVoice \
  --voice-id nana \
  --ref-audio assets/nana.wav \
  --ref-text assets/nana.txt \
  --voices-dir voices \
  --device cuda:0 \
  --dtype float16
```

`--ref-text` accepts either an existing `.txt` file or literal text:

```bash
.venv/bin/python omnivoice_cli.py create-prompt \
  --voice-id nana \
  --ref-audio assets/nana.wav \
  --ref-text "Reference transcript." \
  --voices-dir voices
```

For `assets/nana.wav` and `assets/nana.txt`, this creates:

```text
voices/
└── nana.pt
```

Use `--device cpu --dtype float32` when preparing prompts without a GPU.
The prompt name is derived from the reference audio filename. To add another
voice, run `create-prompt` again with its audio file and transcript.

The server loads `voices/*.pt` only and does not read files from `assets/`.

## Start the server

```bash
.venv/bin/python omnivoice_server.py \
  --flashinfer off \
  --compile dynamic \
  --port 8003
```

The default backend is fp16 with `torch.compile(model.llm, dynamic=True)`.
Use `--flashinfer on` only on a supported GPU with FlashInfer installed.

Open the web interface:

```text
http://127.0.0.1:8003/web/
```

## API

List voices:

```bash
curl http://127.0.0.1:8003/v1/audio/voices
```

Generate speech:

```bash
curl -X POST http://127.0.0.1:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "omnivoice",
    "input": "Hello, this is OmniVoice.",
    "voice": "nana",
    "response_format": "wav",
    "num_step": 32
  }' \
  --output output.wav
```

Endpoints:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/v1/audio/voices` | List available voices |
| `POST` | `/v1/audio/speech` | Generate speech |
| `GET` | `/v1/models` | List models |
| `GET` | `/health` | Check server status |

Request fields:

| Field | Default | Description |
| --- | --- | --- |
| `input` | required | Text to synthesize |
| `voice` | `nana` | Voice preset name |
| `response_format` | `wav` | `wav` or `pcm` |
| `speed` | `1.0` | Speed from `0.25` to `4.0` |
| `num_step` | `32` | Generation steps from `4` to `64` |
| `guidance_scale` | `2.0` | CFG guidance scale |
| `language` | auto | Language name or code |

## OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8003/v1",
    api_key="not-needed",
)

response = client.audio.speech.create(
    model="omnivoice",
    voice="nana",
    input="Hello from the OpenAI SDK.",
    response_format="wav",
)
response.write_to_file("output.wav")
```

## CLI

Create with a saved prompt:

```bash
.venv/bin/python omnivoice_cli.py clone \
  --text "Hello, this is OmniVoice." \
  --output output.wav \
  --voice-id nana \
  --language English \
  --num_step 32
```

Use `--ref_audio assets/nana.wav --ref_text "Reference transcript."` instead
of `--voice-id` to clone directly from reference files. Use `--prompt` to
load a specific prompt path.

The CLI uses `--compile dynamic` by default. Use `--compile off` for eager
execution or `--flashinfer on` to require FlashInfer.

## Docker

```bash
docker build -t omnivoice-openai .

docker run --rm --gpus all --ipc=host \
  -p 8003:8003 \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/voices:/app/voices:ro" \
  omnivoice-openai
```

Build with FlashInfer:

```bash
docker build --build-arg INSTALL_FLASHINFER=true -t omnivoice-openai .
```

## Server options

| Option | Default | Description |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Listen address |
| `--port` | `8003` | Listen port |
| `--model` | `models/OmniVoice` | Model path or Hugging Face repo ID |
| `--device` | `cuda:0` | CUDA device |
| `--num-step` | `32` | Default generation steps |
| `--flashinfer` | `off` | `auto`, `on`, or `off` |
| `--voices-dir` | `voices` | Voice prompt directory |
| `--default-voice` | `nana` | Default voice |
| `--api-key` | empty | Optional bearer token |
