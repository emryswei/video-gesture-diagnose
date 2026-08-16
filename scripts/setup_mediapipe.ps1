param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$modelDirectory = Join-Path $projectRoot "models"
$modelPath = Join-Path $modelDirectory "hand_landmarker.task"
$modelUrl = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

New-Item -ItemType Directory -Path $modelDirectory -Force | Out-Null
if ((Test-Path -LiteralPath $modelPath) -and -not $Force) {
    Write-Host "MediaPipe model already exists: $modelPath"
    exit 0
}

Write-Host "Downloading the official MediaPipe Hand Landmarker model..."
Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath
Write-Host "Saved: $modelPath"
