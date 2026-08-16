# SOP Video Analysis Demo

A single-user demo that audits a 15–60 second video against the seven visually auditable steps in the WHO alcohol-based handrub procedure. The V2 pipeline samples up to 32 timestamped frames, classifies one dominant action in each overlapping 4-frame window, and aggregates the evidence by step and time.

## What V2 includes

- English-only upload and evidence UI
- One active asynchronous analysis at a time
- `passed`, `failed`, and `uncertain` step results
- Procedure duration validation against the WHO 20–30 second range
- Local browser playback with clickable timestamp evidence
- No source-video persistence; server temporary files are deleted after analysis

## Requirements

- Python 3.10+
- Ollama for the local Windows demo, or another OpenAI-compatible multimodal endpoint

The FastAPI app does not load model weights or require Torch. Ollama loads the local model.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen3-vl:2b-instruct
.\start_demo.ps1
```

`start_demo.ps1` closes the Ollama tray app and stale model runners, then restarts
Ollama with the `cpu_avx2` runner. GPU inference is disabled because the Vulkan
backend caused system-level black screens on the current AMD GPU. The application
keeps one model and one inference request active at a time.

Open <http://127.0.0.1:8000>.

The checked-in `.env.example` targets Ollama at `http://127.0.0.1:11434/v1`.
The local `.env` uses `qwen3-vl:2b-instruct` by default with 32-frame/448px
sampling and overlapping 3-frame inference windows to reduce adjacent-gesture
mixing. The 4B model remains an optional, slower fallback.

To run the 4B fallback explicitly:

```powershell
ollama pull qwen3-vl:4b-instruct
.\start_demo.ps1 -Model qwen3-vl:4b-instruct
```

## Model endpoint contract

The app calls:

```text
POST {MODEL_BASE_URL}/chat/completions
```

The endpoint must accept OpenAI-style multimodal messages containing base64 JPEG `image_url` items. Sampled frames—not the complete source video—leave the app and are sent to this endpoint.

## API

Create one analysis:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/analyses `
  -F "video=@C:\path\to\handrub.mp4" `
  -F "sop_id=who_handrub"
```

Poll the returned ID:

```powershell
curl.exe http://127.0.0.1:8000/api/analyses/ANALYSIS_ID
```

Only one job may be `queued` or `analyzing`. A concurrent create request returns HTTP 409. Completed results remain in memory for 30 minutes; expired or server-lost IDs return HTTP 410.

## Tests

```powershell
pip install -r requirements-dev.txt
pytest -q
```

Unit tests do not call the external model service.
