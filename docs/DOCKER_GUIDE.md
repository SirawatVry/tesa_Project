# Docker Setup Guide

Run the pipeline in isolated containers.

---

## 🐳 Docker Configuration

### Available Options
- **Dockerfile**: CPU version (Python 3.10)
- **Dockerfile.gpu**: GPU version (CUDA 11.8)

### File Organization
```
.
├── Dockerfile              # CPU image
├── Dockerfile.gpu          # GPU image
├── docker-compose.yml      # CPU orchestration
├── docker-compose.gpu.yml  # GPU orchestration
└── .dockerignore          # Excluded files
```

---

## 🚀 Quick Start

### CPU Version
```bash
# Build
docker build -t drone-detection:latest .

# Run
docker run --rm -v $(pwd)/outputs:/app/outputs drone-detection:latest
```

### GPU Version
```bash
# Build
docker build -f Dockerfile.gpu -t drone-detection:gpu .

# Run (requires NVIDIA runtime)
docker run --rm --gpus all \
  -v $(pwd)/outputs:/app/outputs \
  drone-detection:gpu
```

---

## 🧬 Docker Compose

### CPU
```bash
# Start
docker-compose up

# Stop
docker-compose down

# View logs
docker-compose logs -f
```

### GPU
```bash
# Start
docker-compose -f docker-compose.gpu.yml up

# Stop
docker-compose -f docker-compose.gpu.yml down
```

---

## 🔧 Volume Mounts

Default volumes:
```yaml
inputs:
  - P3_VIDEO.mp4 (read-only)

outputs:
  - outputs/

configs:
  - configs/ (read-only)

models:
  - models/ (read-only)

data:
  - data/ (read-only)
```

---

## 💻 Interactive Mode

### Access Container Shell
```bash
docker-compose run --rm drone-detection /bin/bash
```

### Run Commands Inside Container
```bash
docker-compose exec drone-detection python src/problem_3_pipeline.py
```

---

## 🎯 GPU Support

### Prerequisites
- NVIDIA GPU
- NVIDIA Docker Runtime
- CUDA 11.8+
- cuDNN 8.x

### Verify GPU
```bash
docker run --rm --gpus all drone-detection:gpu \
  python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

---

## 📊 Image Size

| Version | Size |
|---------|------|
| CPU | ~2.5-3 GB |
| GPU | ~4-5 GB |

---

## 🧹 Cleanup

```bash
# Remove containers
docker-compose down

# Remove images
docker rmi drone-detection:latest
docker rmi drone-detection:gpu

# Remove all
docker system prune -a
```
```

### 2. Run with GPU
```bash
# Direct Docker command
docker run --rm --gpus all \
  -v $(pwd)/outputs:/app/outputs \
  tesa-drone-detection:gpu

# Using Docker Compose
docker-compose -f docker-compose.gpu.yml up
```

### 3. Verify GPU is Available
```bash
docker run --rm --gpus all tesa-drone-detection:gpu \
  python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

---

## 📁 Volume Mounts

Default mounts:
```
P3_VIDEO.mp4    → Input video
outputs/        → Output directory
configs/        → Configuration files
models/         → Pre-trained models
data/           → Processed data
```

### Custom Mount Example
```bash
docker run -it \
  -v $(pwd)/P3_VIDEO.mp4:/app/P3_VIDEO.mp4:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/custom_models:/app/custom_models:ro \
  tesa-drone-detection:latest
```

---

## 🔧 Interactive Mode

Run container with shell access:

### With Docker Compose
```bash
# CPU
docker-compose run --rm tesa-drone-detection /bin/bash

# GPU
docker-compose -f docker-compose.gpu.yml run --rm tesa-drone-detection-gpu /bin/bash
```

### Direct Docker
```bash
docker run -it --rm \
  -v $(pwd):/app \
  tesa-drone-detection:latest /bin/bash
```

---

## 📊 Monitoring

### View Container Logs
```bash
# Real-time logs
docker-compose logs -f

# GPU usage
docker stats
```

### Access Running Container
```bash
docker exec -it <container_id> /bin/bash

# Or with docker-compose
docker-compose exec tesa-drone-detection /bin/bash
```

---

## 🛑 Cleanup

```bash
# Stop and remove container
docker-compose down

# Remove image
docker rmi tesa-drone-detection:latest

# Remove all Docker artifacts
docker system prune -a
```

---

## 📝 File Size

### CPU Image
```
~2.5-3 GB (with all dependencies)
```

### GPU Image
```
~4-5 GB (with CUDA, cuDNN, PyTorch)
```

---

## 🐛 Troubleshooting

### Issue: GPU not detected
```bash
# Check NVIDIA runtime
docker run --rm --gpus all ubuntu nvidia-smi

# Check Docker daemon config
docker info | grep nvidia
```

### Issue: Out of memory
```bash
# Limit GPU memory
docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  tesa-drone-detection:gpu
```

### Issue: Permission denied on outputs
```bash
# Fix permissions
chmod -R 777 outputs/

# Or use docker-compose with user mapping
# (See docker-compose.yml for details)
```

---

## 📖 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Docker Documentation](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

