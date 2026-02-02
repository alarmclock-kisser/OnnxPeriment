param(
    [string]$CudaRoot,
    [string[]]$OutputDirs,
    [switch]$TestLoad
)

$ErrorActionPreference = "Stop"

function Resolve-CudaRoot {
    param([string]$CudaRoot)

    if (-not [string]::IsNullOrWhiteSpace($CudaRoot) -and (Test-Path $CudaRoot)) {
        return (Resolve-Path $CudaRoot).Path
    }

    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc -and (Test-Path $nvcc.Path)) {
        $bin = Split-Path $nvcc.Path -Parent
        return (Split-Path $bin -Parent)
    }

    $default = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    if (Test-Path $default) {
        return $default
    }

    return $null
}

function Get-DefaultOutputDirs {
    $root = Split-Path $PSScriptRoot -Parent
    $candidates = Get-ChildItem -Path $root -Recurse -Directory -Filter "native" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*\\runtimes\\win-x64\\native" }

    if ($candidates.Count -gt 0) {
        return $candidates.FullName
    }

    return @(
        Join-Path $root "OnnxPeriment.Forms\bin\Debug\net10.0-windows\runtimes\win-x64\native",
        Join-Path $root "OnnxPeriment.Runtime\bin\Debug\net10.0\runtimes\win-x64\native"
    )
}

$cudaRootResolved = Resolve-CudaRoot -CudaRoot $CudaRoot
if (-not $cudaRootResolved) {
    Write-Error "CUDA root not found. Provide -CudaRoot or ensure nvcc is in PATH."
}

$cudaBin = Join-Path $cudaRootResolved "bin"
if (-not (Test-Path $cudaBin)) {
    Write-Error "CUDA bin folder not found at $cudaBin."
}

if (-not $OutputDirs -or $OutputDirs.Length -eq 0) {
    $OutputDirs = Get-DefaultOutputDirs
}

$OutputDirs = $OutputDirs | ForEach-Object { if ($_ -and $_.Trim().Length -gt 0) { $_ } } | Select-Object -Unique

foreach ($dir in $OutputDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

$cudaDlls = @(
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cufft64_12.dll",
    "curand64_10.dll",
    "cusolver64_11.dll",
    "cusparse64_12.dll"
)

$missing = @()

foreach ($dll in $cudaDlls) {
    $src = Join-Path $cudaBin $dll
    if (-not (Test-Path $src)) {
        $missing += $dll
        continue
    }

    foreach ($dir in $OutputDirs) {
        Copy-Item -Path $src -Destination (Join-Path $dir $dll) -Force
    }
}

if ($missing.Count -gt 0) {
    Write-Warning "Missing CUDA DLLs in $cudaBin: $($missing -join ', ')"
}

$cuDnnDlls = @(
    "cudnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll"
)

$cuDnnFound = @()
foreach ($dll in $cuDnnDlls) {
    $src = Join-Path $cudaBin $dll
    if (Test-Path $src) {
        $cuDnnFound += $dll
        foreach ($dir in $OutputDirs) {
            Copy-Item -Path $src -Destination (Join-Path $dir $dll) -Force
        }
    }
}

if ($cuDnnFound.Count -eq 0) {
    Write-Warning "cuDNN DLLs not found in $cudaBin. Install cuDNN for CUDA 12.x and ensure DLLs are on PATH or copied here."
}

Write-Host "CUDA root: $cudaRootResolved"
Write-Host "Output directories:" 
$OutputDirs | ForEach-Object { Write-Host "  - $_" }

if ($TestLoad) {
    $code = @"
using System;
using System.Runtime.InteropServices;

public static class DllTester
{
    public static bool TryLoad(string path)
    {
        IntPtr handle;
        if (NativeLibrary.TryLoad(path, out handle))
        {
            NativeLibrary.Free(handle);
            return true;
        }
        return false;
    }
}
"@

    Add-Type -TypeDefinition $code -Language CSharp

    foreach ($dir in $OutputDirs) {
        Write-Host "Testing DLL loads in $dir"
        foreach ($dll in $cudaDlls + $cuDnnFound) {
            $path = Join-Path $dir $dll
            if (Test-Path $path) {
                $ok = [DllTester]::TryLoad($path)
                Write-Host "  $dll => $ok"
            }
        }
    }
}
