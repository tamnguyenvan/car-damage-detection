# Car Damage Detection
![cover](./assets/cover.gif)

A production-grade, containerized FastAPI microservice designed to serve YOLO object detection inference. This repository is structured for seamless local development and Docker-based container deployments.

## Table of Contents
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Development](#development)
- [Training](#training)
- [API Documentation](docs/API.md)

## Key Features

- **High-Performance Framework**: Built with **FastAPI** and served via **Uvicorn** for asynchronous request handling.
- **Optimized Lifespan Management**: Model weights are loaded once at startup and cleared gracefully from memory on shutdown.
- **Robust Observability**: Structured JSON error responses and system-level logging are implemented for predictability and production monitoring.
- **Secure Containerization**: Configured with a multi-layered Docker setup that runs as a non-privileged system user (`appuser`).
- **Production-Ready Endpoints**: Includes structured health probes (`/health`) and OpenAPI specification-compliant output payloads.


---

## Getting Started

The recommended way to run this microservice is via Docker. We provide configurations for both CPU and GPU deployment.

### 1. Clone the Repository

```bash
git clone https://github.com/tamnvcc/car-damage-detection.git
cd car-damage-detection
```

### 2. Model Weights Setup
Place your trained YOLO model weight file (e.g., `best.pt` or `best.onnx`) inside the `models/` directory:
```bash
mkdir -p models
cp /path/to/your/trained/weights.pt models/best.pt
```

### 3. Run with Docker (CPU)

**Build the Docker Image**
Execute the build command in the root directory where the `Dockerfile` resides:
```bash
docker build -t car-damage-detection:1.0.0 .
```

**Run the Container**
Start the container while mounting the host `models` directory. This allows you to update or swap the weights on the host without rebuilding the container.

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name car-damage-detector \
  car-damage-detection:1.0.0
```

### 4. Run with Docker (GPU)

For deployment on a GPU-enabled server, use the dedicated GPU Dockerfile (`Dockerfile.gpu`). Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your host system.

**Build the GPU Image**
```bash
docker build -t car-damage-detection:1.0.0-gpu -f Dockerfile.gpu .
```

**Run the GPU Container**
Start the container and attach all available GPUs by adding the `--gpus all` flag:
```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name car-damage-detector-gpu \
  car-damage-detection:1.0.0-gpu
```

---

## Development

For local testing or development without Docker:

```bash
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install pytorch
# CPU
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu

# GPU
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128


# Install other dependencies
pip install -r requirements.txt

# Run the API locally
export MODEL_PATH="./models/best.pt"
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
The API documentation will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)


Quick test
```bash
# cURL
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/image.jpg"

# or run python script
python test.py /path/to/image.jpg
```

---

## Training

We provide a complete object-detection training and fine-tuning pipeline for Ultralytics YOLO models on the CarDD dataset. We highly recommend running a fully automated, serverless training job using **Modal** for access to powerful GPUs (e.g., H100/A100). Alternatively, you can train the model interactively using **Google Colab**.

For complete setup instructions, training scripts, and dataset preparation, please refer to the [Training Guide](notebooks/README.md).

---

## API Documentation

For detailed information on the available endpoints, expected payloads, and error handling, please refer to the [API Documentation](docs/API.md).
