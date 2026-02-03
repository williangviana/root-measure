# Root Measure — One-line installer for Windows
# Usage: powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/williangviana/root-measure/stable/install/install.ps1 | iex"

$ErrorActionPreference = "Stop"

$AppName = "Root Measure"
$Repo = "williangviana/root-measure"
$WorkDir = "$env:TEMP\root-measure-install"
$InstallDir = "$env:LOCALAPPDATA\Root Measure"

Write-Host ""
Write-Host "============================================"
Write-Host "  Root Measure - Installer"
Write-Host "============================================"
Write-Host ""

# --- 1. Ensure Python 3 is installed ---
$py = $null
foreach ($cmd in @("python", "python3")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3") {
            $py = $cmd
            break
        }
    } catch {}
}

if (-not $py) {
    Write-Host "[1/6] Python not found. Installing Python..."
    $pyInstaller = "$env:TEMP\python-installer.exe"
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe" -OutFile $pyInstaller
    Start-Process -Wait -FilePath $pyInstaller -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_pip=1"
    Remove-Item $pyInstaller -Force
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $py = "python"
}

$pyVersion = & $py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "[1/6] Python $pyVersion OK"

# --- 2. Download project from GitHub ---
Write-Host "[2/6] Downloading Root Measure..."
if (Test-Path $WorkDir) { Remove-Item $WorkDir -Recurse -Force }
New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null
$zipPath = "$env:TEMP\root-measure.zip"
Invoke-WebRequest -Uri "https://github.com/$Repo/archive/refs/heads/stable.zip" -OutFile $zipPath
Expand-Archive -Path $zipPath -DestinationPath $WorkDir -Force
Remove-Item $zipPath -Force
# Move contents out of the nested folder
$nested = Get-ChildItem $WorkDir -Directory | Select-Object -First 1
Get-ChildItem $nested.FullName | Move-Item -Destination $WorkDir -Force
Remove-Item $nested.FullName -Recurse -Force
Write-Host "[2/6] Downloaded OK"

# --- 3. Create virtual environment ---
Set-Location $WorkDir
& $py -m venv .venv
& .venv\Scripts\Activate.ps1
Write-Host "[3/6] Virtual environment OK"

# --- 4. Install dependencies ---
Write-Host "[4/6] Installing dependencies..."
& pip install --upgrade pip -q
& pip install -r install/requirements.txt -q
& pip install cx_Freeze -q
Write-Host "[4/6] Dependencies OK"

# --- 5. Build executable ---
Write-Host "[5/6] Building app..."
& python install/setup.py build_exe

$builtDir = Get-ChildItem "build" -Directory | Where-Object { $_.Name -match "exe" } | Select-Object -First 1
if (-not $builtDir) {
    Write-Host "ERROR: Build failed."
    exit 1
}
Write-Host "[5/6] Build OK"

# --- 6. Install to local app directory ---
if (Test-Path $InstallDir) { Remove-Item $InstallDir -Recurse -Force }
Move-Item $builtDir.FullName $InstallDir

# Create desktop shortcut
$exePath = Get-ChildItem $InstallDir -Filter "RootMeasure.exe" -Recurse | Select-Object -First 1
if ($exePath) {
    $desktop = [Environment]::GetFolderPath("Desktop")
    $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut("$desktop\Root Measure.lnk")
    $shortcut.TargetPath = $exePath.FullName
    $shortcut.WorkingDirectory = $InstallDir
    $shortcut.Description = $AppName
    $shortcut.Save()
}
Write-Host "[6/6] Installed OK"

# Clean up
Set-Location $env:USERPROFILE
Remove-Item $WorkDir -Recurse -Force

Write-Host ""
Write-Host "============================================"
Write-Host "  Installation complete!"
Write-Host ""
Write-Host "  A shortcut has been added to your Desktop."
Write-Host "  You can also find the app at:"
Write-Host "  $InstallDir"
Write-Host "============================================"
