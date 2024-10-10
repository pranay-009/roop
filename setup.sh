#!/bin/bash

# Download the ONNX model
pip install -r requirements.txt
wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -O inswapper_128.onnx

# Create a directory for models if it doesn't exist
mkdir -p models

# Move the downloaded ONNX model to the models directory
mv inswapper_128.onnx ./models

# Download the ONNX Runtime GPU wheel file
wget -O onnxruntime_gpu-1.16.3-cp310-cp310-linux_x86_64.whl "https://huggingface.co/MarioG/ff_components/resolve/main/onnxruntime_gpu-1.16.3-cp310-cp310-linux_x86_64.whl?download=true"

# Uninstall the existing onnxruntime package quietly
pip uninstall -y -q onnxruntime

# Install the new ONNX Runtime GPU wheel file
pip install ./onnxruntime_gpu-1.16.3-cp310-cp310-linux_x86_64.whl

# Print completion message
echo -e '\033[1;32mDone!\033[0m'

