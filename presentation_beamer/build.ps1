$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    for ($pass = 1; $pass -le 2; $pass++) {
        & xelatex -interaction=nonstopmode -halt-on-error -jobname=robot_control main.tex
        if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX compilation failed.' }
    }
} finally {
    Pop-Location
}
