$ErrorActionPreference = "Stop"

python -m ruff check --isolated --select E4,E7,E9,F app.py video_analysis scripts tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m compileall -q app.py video_analysis scripts tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$html = Get-Content -LiteralPath "static/index.html" -Raw
$scriptMatch = [regex]::Match($html, "(?s)<script>(.*?)</script>")
if (-not $scriptMatch.Success) {
    throw "No inline JavaScript block found in static/index.html."
}
$scriptMatch.Groups[1].Value | node --check -
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
