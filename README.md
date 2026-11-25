# OmniVoice OpenAI API

Standalone OmniVoice text-to-speech service with an OpenAI-compatible API.

## Requirements

- Linux
- Python 3.10+
- NVIDIA GPU with CUDA
- Model files in `models/OmniVoice/`
- Reference audio and UTF-8 transcript pairs in `voices/`

## Install

```bash
uv sync
```

Optional FlashInfer support:

```bash
uv sync --extra flashinfer
```

## Prepare voices

Each voice requires a matching audio and text file:

```text
voices/
├── nana.wav
└── nana.txt
```

Supported audio formats: `wav`, `flac`, `mp3`, and `ogg`.

## Start the server

```bash
.venv/bin/python omnivoice_server.py --port 8003
```

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

```bash
.venv/bin/python inference.py \
  --text "Hello, this is OmniVoice." \
  --output output.wav \
  --ref_audio voices/nana.wav \
  --ref_text "Reference transcript." \
  --language English \
  --num_step 32
```

Use `--flashinfer off` to disable FlashInfer or `--flashinfer on` to require it.

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
| `--flashinfer` | `auto` | `auto`, `on`, or `off` |
| `--voices-dir` | `voices` | Voice preset directory |
| `--default-voice` | `nana` | Default voice |
| `--api-key` | empty | Optional bearer token |
