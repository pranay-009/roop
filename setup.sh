#!/bin/bash

# Download the ONNX model
pip install -r requirements.txt
pip install gfpgan

wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -O inswapper_128.onnx
mkdir models
mv inswapper_128.onnx ./models
wget -O /content/onnxruntime_gpu-1.16.3-cp310-cp310-linux_x86_64.whl "https://huggingface.co/MarioG/ff_components/resolve/main/onnxruntime_gpu-1.16.3-cp310-cp310-linux_x86_64.whl?download=true"
pip uninstall -y -q onnxruntime
pip install /content/onnxruntime_gpu-1.16.3-cp310-cp310-linux_x86_64.whl

pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
pip install huggingface_hub==0.20.2