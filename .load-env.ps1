# Load .env file variables into PowerShell environment
# Usage: . .\.load-env.ps1

$envFile = ".env"

if (-Not (Test-Path $envFile)) {
    Write-Host "Error: $envFile not found in current directory" -ForegroundColor Red
    return
}

Get-Content $envFile | ForEach-Object {
    # Skip empty lines and comments
    if ($_ -match '^\s*$' -or $_ -match '^\s*#') {
        return
    }
    
    # Parse KEY=VALUE
    if ($_ -match '^([^=]+)=(.*)$') {
        $key = $matches[1].Trim()
        $value = $matches[2].Trim()
        
        # Remove quotes if present
        $value = $value -replace '^["\']|["\']$', ''
        
        # Set environment variable
        [Environment]::SetEnvironmentVariable($key, $value, "Process")
        Write-Host "✓ $key" -ForegroundColor Green
    }
}

Write-Host "`n.env variables loaded!" -ForegroundColor Cyan
