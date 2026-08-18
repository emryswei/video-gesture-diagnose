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
- An AWS account with Amazon Bedrock access
- AWS credentials allowed to call `bedrock:InvokeModel`
- Qwen3-VL 235B available in the selected AWS Region

The FastAPI app does not load model weights or require Torch. Amazon Bedrock runs
Qwen3-VL as a managed service. The application sends sampled JPEG frames, not the
complete source video, to Bedrock.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
.\scripts\setup_mediapipe.ps1
Copy-Item .env.example .env
aws configure
.\start_aws_demo.ps1
```

The setup script downloads Google's Hand Landmarker model to the ignored local
`models` directory. MediaPipe supplies finger-shape evidence to Qwen3-VL and
guards the easily confused palm, interlaced-fingers, dorsum, and backs-of-fingers
steps. A separate 4 FPS low-resolution pass detects visual transitions so the
timeline uses action boundaries instead of the edges of broad VLM windows. If
the landmarker is unavailable, analysis continues with a warning and VLM-only
output.

`start_aws_demo.ps1` defaults to the Tokyo Region (`ap-northeast-1`), the
`qwen.qwen3-vl-235b-a22b` Bedrock model, and local port 8500. Override a named AWS
profile when needed:

```powershell
.\start_aws_demo.ps1 -Profile demo -Region ap-northeast-1
```

Open <http://127.0.0.1:8500>.

The AWS SDK uses the standard credential chain: environment variables, the profile
selected by `AWS_PROFILE`, shared AWS credential files, or an IAM role. Do not put
AWS secret keys in `.env` or commit them to Git.

Fast Demo mode uses 24 resized frames and twelve non-overlapping inference windows.
Repeating the same video, SOP, and model within 30 minutes returns the in-memory
cached result without another Bedrock request. MediaPipe and visual transition
refinement remain local and do not require AWS.

Fast Demo mode disables extra VLM candidate rechecks by default, keeping the first
analysis to 12 model calls. Set `MODEL_TARGETED_RECHECKS=true` for a slower,
higher-redundancy pass. MediaPipe evidence collection remains enabled in both modes.

## Amazon Bedrock inference

The app calls the Amazon Bedrock Runtime `Converse` API through Boto3. Each request
contains the SOP prompt and up to two timestamped JPEG frames. Configure:

```dotenv
MODEL_PROVIDER=aws_bedrock
AWS_REGION=ap-northeast-1
AWS_BEDROCK_MODEL_ID=qwen.qwen3-vl-235b-a22b
```

The IAM identity needs at least:

```json
{
  "Effect": "Allow",
  "Action": "bedrock:InvokeModel",
  "Resource": "*"
}
```

To use the original local OpenAI-compatible adapter instead, set
`MODEL_PROVIDER=openai_compatible` and configure `MODEL_BASE_URL`, `MODEL_API_KEY`,
and `MODEL_NAME`.

## API

Create one analysis:

```powershell
curl.exe -X POST http://127.0.0.1:8500/api/analyses `
  -F "video=@C:\path\to\handrub.mp4" `
  -F "sop_id=hk_chp_handrub"
```

Poll the returned ID:

```powershell
curl.exe http://127.0.0.1:8500/api/analyses/ANALYSIS_ID
```

Only one job may be `queued` or `analyzing`. A concurrent create request returns HTTP 409. Completed results and matching cache entries remain in memory for 30 minutes; expired or server-lost IDs return HTTP 410. Cached jobs return `cache_hit: true`.

## Tests

```powershell
pip install -r requirements-dev.txt
pytest -q
```

Unit tests do not call the external model service.
