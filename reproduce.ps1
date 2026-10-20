param(
    [string]$Python = "python",
    [string]$Config = "configs/smoke.json",
    [string]$Tag = "replay"
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
function Invoke-CheckedPython {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
Invoke-CheckedPython -m unittest discover -s tests -v
Invoke-CheckedPython main.py run --config $Config --out "results/${Tag}_a"
Invoke-CheckedPython main.py run --config $Config --out "results/${Tag}_b"
Invoke-CheckedPython main.py compare "results/${Tag}_a" "results/${Tag}_b" --report "results/${Tag}_reproducibility.json"
Invoke-CheckedPython main.py plot "results/${Tag}_a"
