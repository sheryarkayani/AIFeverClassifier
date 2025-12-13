#!/bin/bash
set -e

# Create venv if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Install dependencies
./venv/bin/pip install -r requirements.txt

# Train model if not exists
if [ ! -f "fever_model.pkl" ]; then
    echo "Model not found. Training model..."
    ./venv/bin/python src/model.py
fi

# Run the application
./venv/bin/python -m uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
