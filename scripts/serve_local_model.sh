#!/bin/bash
# Helper script to serve a local model via vLLM

MODEL_NAME=${1:-"Qwen/Qwen2.5-1.5B-Instruct"}
PORT=${2:-8001}

echo "Starting vLLM serving $MODEL_NAME on port $PORT..."
vllm serve "$MODEL_NAME" --port "$PORT"
