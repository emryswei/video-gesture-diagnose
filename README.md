# SOP Video Analysis Demo

A single-user demo that audits a 15–60 second video against the Hong Kong CHP alcohol-based handrub procedure. The default SOP includes product application plus seven hand-rubbing actions, ending with wrist rubbing. Fast Demo mode samples up to 24 timestamped frames and classifies twelve non-overlapping 2-frame windows before targeted gesture checks.

## What V2 includes

- English-only upload and evidence UI
- One active asynchronous analysis at a time
- `passed`, `failed`, and `uncertain` step results
- Procedure duration validation against the Hong Kong CHP minimum of 20 seconds
- Local browser playback with clickable timestamp evidence
- 30-minute in-memory result cache for identical video, SOP, and model combinations
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
.\scripts\setup_mediapipe.ps1
.\start_demo.ps1
```

The setup script downloads Google's Hand Landmarker model to the ignored local
`models` directory. MediaPipe supplies finger-shape evidence to Qwen3-VL and
guards the easily confused palm, interlaced-fingers, dorsum, and backs-of-fingers
steps. A separate 4 FPS low-resolution pass detects visual transitions so the
timeline uses action boundaries instead of the edges of broad VLM windows. If
the landmarker is unavailable, analysis continues with a warning and VLM-only
output.

`start_demo.ps1` closes the Ollama tray app and stale model runners, then restarts
Ollama with the `cpu_avx2` runner. GPU inference is disabled because the Vulkan
backend caused system-level black screens on the current AMD GPU. The application
keeps one model and one inference request active at a time.

After startup, `ollama ps` should show `100% CPU`. Do not launch the Ollama tray
app during analysis because it can replace the controlled CPU server with its own
environment settings.

Fast Demo mode reduces repeated image processing with 24 smaller frames and no
window overlap. CPU inference remains
slower than GPU inference, but avoids the unstable experimental Vulkan path on the
installed AMD driver. Repeating an identical analysis within 30 minutes returns the
cached result without running the model again.

Open <http://127.0.0.1:8000>.

The checked-in `.env.example` targets Ollama at `http://127.0.0.1:11434/v1`.
The local `.env` uses `qwen3-vl:2b-instruct` by default with 24-frame/336px
sampling and non-overlapping 2-frame inference windows. MediaPipe and visual
transition refinement remain enabled for gesture and timestamp accuracy. The 4B
model remains an optional, slower fallback.

Fast Demo mode disables extra VLM candidate rechecks by default, keeping the first
analysis to 12 model calls. Set `MODEL_TARGETED_RECHECKS=true` for a slower,
higher-redundancy pass. MediaPipe evidence collection remains enabled in both modes.

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
  -F "sop_id=hk_chp_handrub"
```

Poll the returned ID:

```powershell
curl.exe http://127.0.0.1:8000/api/analyses/ANALYSIS_ID
```

Only one job may be `queued` or `analyzing`. A concurrent create request returns HTTP 409. Completed results and matching cache entries remain in memory for 30 minutes; expired or server-lost IDs return HTTP 410. Cached jobs return `cache_hit: true`.

## Tests

```powershell
pip install -r requirements-dev.txt
pytest -q
```

Unit tests do not call the external model service.
