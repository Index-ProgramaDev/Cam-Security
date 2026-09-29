# Script de Inicialização Rápida para o Cam-Security
Write-Host "Iniciando configuração do ambiente Cam-Security..." -ForegroundColor Green

# Verifica se o Python está instalado
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "Python não encontrado! Instale o Python (versão 3.9 a 3.11 recomendada) antes de continuar." -ForegroundColor Red
    exit 1
}

# Cria o ambiente virtual se não existir
if (-not (Test-Path "venv")) {
    Write-Host "Criando ambiente virtual (venv)..." -ForegroundColor Cyan
    python -m venv venv
} else {
    Write-Host "Ambiente virtual já existe. Pulando criação." -ForegroundColor Yellow
}

# Ativa o ambiente virtual e instala as dependências
Write-Host "Instalando dependências do requirements.txt..." -ForegroundColor Cyan
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# Cria diretórios de storage que possam estar faltando
$dirs = @("storage", "storage/events", "storage/evidences", "storage/face_tracks", "storage/periodic_snapshots", "storage/snapshots", "storage/temp")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
}

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "Tudo pronto! Para rodar o projeto, use:" -ForegroundColor White
Write-Host "1. Ative o venv: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "2. Rode: python main.py" -ForegroundColor Yellow
Write-Host "=======================================================" -ForegroundColor Green
