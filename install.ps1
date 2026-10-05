# Installs (or updates) Meeting Transcriptions on Windows from the latest GitHub release.
#   irm https://raw.githubusercontent.com/karlis-balcers/meeting-transcriptions/main/install.ps1 | iex
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$repo = 'karlis-balcers/meeting-transcriptions'
$installDir = Join-Path $env:LOCALAPPDATA 'Programs\MeetingTranscriptions'

Write-Host 'Looking up the latest release...'
$release = Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest" -Headers @{ 'User-Agent' = 'meeting-transcriptions-installer' }
$asset = $release.assets | Where-Object { $_.name -like '*windows*.zip' } | Select-Object -First 1
if (-not $asset) { throw "No Windows build in release $($release.tag_name)." }

$zip = Join-Path $env:TEMP $asset.name
Write-Host "Downloading $($asset.name) ($($release.tag_name))..."
Invoke-WebRequest $asset.browser_download_url -OutFile $zip -UseBasicParsing

# Close a running copy so its files can be replaced.
Get-Process MeetingTranscriptions, meeting-engine -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500

if (Test-Path $installDir) { Remove-Item $installDir -Recurse -Force }
New-Item -ItemType Directory -Path $installDir | Out-Null
Expand-Archive $zip -DestinationPath $installDir -Force
Remove-Item $zip

$exe = Join-Path $installDir 'MeetingTranscriptions.exe'
$shell = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath('Programs'), [Environment]::GetFolderPath('Desktop'))) {
    $link = $shell.CreateShortcut((Join-Path $dir 'Meeting Transcriptions.lnk'))
    $link.TargetPath = $exe
    $link.WorkingDirectory = $installDir
    $link.Save()
}

Write-Host "Installed $($release.tag_name) to $installDir"
Write-Host 'Shortcuts added to the Start menu and Desktop. Starting it now...'
Start-Process $exe
