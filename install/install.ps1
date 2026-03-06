# Root Measure — One-line installer for Windows
# Usage: powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/williangviana/root-measure/stable/install/install.ps1 | iex"

$ErrorActionPreference = "Stop"

$AppName = "Root Measure"
$Repo = "williangviana/root-measure"
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
    Write-Host "[1/4] Python not found. Installing Python..."
    $pyInstaller = "$env:TEMP\python-installer.exe"
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe" -OutFile $pyInstaller
    Start-Process -Wait -FilePath $pyInstaller -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_pip=1"
    Remove-Item $pyInstaller -Force
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $py = "python"
}

$pyVersion = & $py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "[1/4] Python $pyVersion OK"

# --- 2. Download project from GitHub ---
Write-Host "[2/4] Downloading Root Measure..."
if (Test-Path $InstallDir) { Remove-Item $InstallDir -Recurse -Force }
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
$zipPath = "$env:TEMP\root-measure.zip"
Invoke-WebRequest -Uri "https://github.com/$Repo/archive/refs/heads/stable.zip" -OutFile $zipPath
Expand-Archive -Path $zipPath -DestinationPath $env:TEMP\root-measure-tmp -Force
# Move contents out of the nested folder into install dir
$nested = Get-ChildItem "$env:TEMP\root-measure-tmp" -Directory | Select-Object -First 1
Get-ChildItem $nested.FullName | Move-Item -Destination $InstallDir -Force
Remove-Item "$env:TEMP\root-measure-tmp" -Recurse -Force
Remove-Item $zipPath -Force
Write-Host "[2/4] Downloaded OK"

# --- 3. Create virtual environment and install dependencies ---
Set-Location $InstallDir
& $py -m venv .venv
Write-Host "[3/4] Virtual environment OK"
Write-Host "[3/4] Installing dependencies (this may take a few minutes)..."
& .venv\Scripts\python.exe -m pip install -r install\requirements.txt -q
Write-Host "[3/4] Dependencies OK"

# --- 4. Create desktop shortcut ---
$pythonw = "$InstallDir\.venv\Scripts\pythonw.exe"
$appScript = "$InstallDir\gui\app.py"

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut("$desktop\Root Measure.lnk")
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = "`"$appScript`""
$shortcut.WorkingDirectory = $InstallDir
$shortcut.Description = $AppName
# Use icon if available
$icoPath = "$InstallDir\icon\icon.ico"
if (Test-Path $icoPath) {
    $shortcut.IconLocation = $icoPath
}
$shortcut.Save()
Write-Host "[4/4] Shortcut created"

Write-Host ""
Write-Host "============================================"
Write-Host "  Installation complete!"
Write-Host ""
Write-Host "  A shortcut has been added to your Desktop."
Write-Host "  You can also find the app at:"
Write-Host "  $InstallDir"
Write-Host "============================================"
