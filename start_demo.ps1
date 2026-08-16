param(
    [ValidateSet("qwen3-vl:2b-instruct", "qwen3-vl:4b-instruct")]
    [string]$Model = "qwen3-vl:4b-instruct"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python virtual environment not found. Run: python -m venv .venv"
}

$ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
$ollamaPath = if ($ollamaCommand) {
    $ollamaCommand.Source
} else {
    Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
}
if (-not (Test-Path -LiteralPath $ollamaPath)) {
    throw "Ollama is not installed. Install it from https://ollama.com/download/windows"
}

# Use only the RX 6600M discrete GPU. Small sequential frame segments reduce
# sustained Vulkan pressure while keeping the local demo responsive.
Get-Process -Name "ollama", "ollama app", "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500
Remove-Item Env:OLLAMA_LLM_LIBRARY -ErrorAction SilentlyContinue
Remove-Item Env:OLLAMA_KV_CACHE_TYPE -ErrorAction SilentlyContinue
$env:OLLAMA_VULKAN = "1"
$env:GGML_VK_VISIBLE_DEVICES = "0"
$env:OLLAMA_CONTEXT_LENGTH = "8192"
$env:OLLAMA_FLASH_ATTENTION = "false"
$env:OLLAMA_GPU_OVERHEAD = "1073741824"
$env:OLLAMA_KEEP_ALIVE = "10m"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_NUM_PARALLEL = "1"

Write-Host "Starting Ollama with Vulkan on RX 6600M (Vulkan0)..."
Start-Process -FilePath $ollamaPath -ArgumentList "serve" -WindowStyle Hidden
$ready = $false
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    Start-Sleep -Milliseconds 500
    try {
        Invoke-RestMethod "http://127.0.0.1:11434/api/version" -TimeoutSec 2 | Out-Null
        $ready = $true
        break
    } catch {
        # Wait briefly while the local model service starts.
    }
}
if (-not $ready) {
    throw "Ollama did not start with Vulkan on http://127.0.0.1:11434"
}

$models = (& $ollamaPath list | Out-String)
if ($LASTEXITCODE -ne 0 -or $models -notmatch [regex]::Escape($Model)) {
    throw "Required model is missing. Run: ollama pull $Model"
}

$env:MODEL_NAME = $Model
& $pythonPath -m uvicorn app:app --host 127.0.0.1 --port 8000
