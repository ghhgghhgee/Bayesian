param([string]$Python = "python", [string]$Tag = "leader_20261021")
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Invoke-CheckedPython {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
# Keep source files unchanged until all run pairs and comparisons finish.
foreach ($algorithm in @('np1', 'np0')) {
    $runPrefix = "results/${Tag}_${algorithm}"
    Invoke-CheckedPython main.py run --config "configs/${algorithm}_validation.json" --out "${runPrefix}_a" --data-dir "data/${Tag}_a"
    Invoke-CheckedPython main.py run --config "configs/${algorithm}_validation.json" --out "${runPrefix}_b" --data-dir "data/${Tag}_b"
    Invoke-CheckedPython main.py plot "${runPrefix}_a"
    Invoke-CheckedPython main.py compare "${runPrefix}_a" "${runPrefix}_b" --report "${runPrefix}_reproducibility.json"
    # The existing audit checks data, metrics, histories and stop conditions
    # for all configured entries; its optional pair-difference table is NP0-only.
    Invoke-CheckedPython scripts/audit_np0_outputs.py "${runPrefix}_a" "${runPrefix}_b" --out "${runPrefix}_numerical_audit.json"
}
