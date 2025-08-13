#!/bin/bash

echo "Starting Streamlit Docker App..."

if [ ! -d "Output" ]; then
  echo "[INFO] 'Output' folder not found. Creating it..."
  mkdir -p "Output"
fi

docker run -p 8501:8501 \
  -v "$(pwd)/Output:/app/Output" \
  sagelabuhn/dt_lung:latest
