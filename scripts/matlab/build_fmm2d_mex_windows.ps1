param(
    [string] $MatlabRoot = "C:\Program Files\MATLAB\R2026a",
    [string] $MsysRoot = "C:\msys64",
    [string] $Fmm2dRoot = "external\chunkie-matlab\chunkie\fmm2d"
)

$ErrorActionPreference = "Stop"

function Convert-ToMsysPath([string] $Path) {
    $resolved = (Resolve-Path $Path).Path
    if ($resolved -match "^([A-Za-z]):\\(.*)$") {
        return "/" + $matches[1].ToLower() + "/" + ($matches[2] -replace "\\", "/")
    }
    return $resolved -replace "\\", "/"
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$fmmRoot = (Resolve-Path (Join-Path $repo $Fmm2dRoot)).Path
$mingwRoot = Join-Path $MsysRoot "mingw64"
$bash = Join-Path $MsysRoot "usr\bin\bash.exe"
$matlab = Join-Path $MatlabRoot "bin\matlab.exe"
$mexopts = Join-Path $MatlabRoot "bin\win64\mexopts\mingw64.xml"

foreach ($path in @($fmmRoot, $mingwRoot, $bash, $matlab, $mexopts)) {
    if (-not (Test-Path $path)) {
        throw "Required path not found: $path"
    }
}

$env:MW_MINGW64_LOC = $mingwRoot
$env:Path = (Join-Path $mingwRoot "bin") + ";" + $env:Path

& $matlab -batch "setenv('MW_MINGW64_LOC','$($mingwRoot.Replace('\','\\'))'); mex -setup:'$($mexopts.Replace('\','\\'))' C"
if ($LASTEXITCODE -ne 0) {
    throw "MATLAB mex MinGW setup failed."
}

$fmmRootMsys = Convert-ToMsysPath $fmmRoot
$buildCommand = "cd '$fmmRootMsys' && export MW_MINGW64_LOC=/mingw64 && export PATH=/mingw64/bin:/usr/bin:`$PATH && make -B matlab OMP=OFF"

& $bash -lc $buildCommand
if ($LASTEXITCODE -ne 0) {
    throw "FMM2D MATLAB MEX build failed."
}

& $matlab -batch "setenv('PATH',['$((Join-Path $mingwRoot 'bin').Replace('\','\\'));' getenv('PATH')]); addpath('$((Join-Path $fmmRoot 'matlab').Replace('\','\\'))'); assert(exist(['fmm2d.' mexext],'file') == 3); disp(which('fmm2d'))"
if ($LASTEXITCODE -ne 0) {
    throw "FMM2D MATLAB MEX verification failed."
}
