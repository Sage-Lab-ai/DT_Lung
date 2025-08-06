@echo off
SETLOCAL

echo Starting Streamlit Docker App...

IF NOT EXIST "Output" (
    echo [INFO] 'Output' folder not found. Creating it...
    mkdir "Output"
)

docker run -p 8501:8501 ^
  -v "%cd%\Output:/app/Output" ^
  sagelabuhn/dt_lung:latest

ENDLOCAL
