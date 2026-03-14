# Загрузка переменных окружения из .env файла
# Использование: .\scripts\load-env.ps1 && dvc push

$envFile = Join-Path $PSScriptRoot "..\.env"

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        # Пропускаем комментарии и пустые строки
        if ($_ -match '^\s*#' -or $_ -match '^\s*$') {
            return
        }
        # Разбираем KEY=VALUE
        if ($_ -match '^([^=]+)=(.*)$') {
            $key = $matches[1]
            $value = $matches[2]
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
            Write-Host "Loaded: $key"
        }
    }
    Write-Host "`nEnvironment variables loaded from .env"
} else {
    Write-Error ".env file not found at $envFile"
    exit 1
}
