#!/bin/bash

echo "Starting Streamlit Docker App..."

if [ ! -d "Output" ]; then
  echo "[INFO] 'Output' folder not found. Creating it..."
  mkdir -p "Output"
fi

docker run -it \
  -v "$(pwd)/Output:/app/Output" \
  sagelabuhn/dt_lung:latest python main.py
