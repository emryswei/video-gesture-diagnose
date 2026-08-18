param(
    [string]$Region = "ap-northeast-1",
    [string]$Model = "qwen.qwen3-vl-235b-a22b",
    [string]$Profile = "",
    [int]$Port = 8500
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python virtual environment not found. Run: python -m venv .venv"
}

$env:MODEL_PROVIDER = "aws_bedrock"
$env:AWS_REGION = $Region
$env:AWS_BEDROCK_MODEL_ID = $Model
if ($Profile) {
    $env:AWS_PROFILE = $Profile
}

Write-Host "Starting AWS Bedrock demo on http://127.0.0.1:$Port"
Write-Host "Region: $Region"
Write-Host "Model:  $Model"
& $pythonPath -m uvicorn app:app --host 127.0.0.1 --port $Port
