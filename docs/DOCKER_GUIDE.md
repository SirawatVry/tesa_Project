# Docker Guide - TESA Problem 2

## 📦 Docker Setup

โปรเจคนี้รองรับการรัน Docker ในสองรูปแบบ:
- **CPU**: Dockerfile (เร็วน้อยแต่เบา)
- **GPU**: Dockerfile.gpu (เร็วขึ้น แต่ต้องมี CUDA)

---

## 🚀 Quick Start (CPU)

### 1. Build Image
```bash
docker build -t tesa-drone-detection:latest .
```

### 2. Run Container
```bash
# แบบอย่างสำหรับ:
docker run --rm -v $(pwd)/outputs:/app/outputs tesa-drone-detection:latest

# Windows PowerShell:
docker run --rm -v ${PWD}\outputs:C:\app\outputs tesa-drone-detection:latest

# Windows CMD:
docker run --rm -v %CD%\outputs:C:\app\outputs tesa-drone-detection:latest
```

### 3. Using Docker Compose
```bash
# Build and run
docker-compose up

# Build only
docker-compose build

# Run container
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop container
docker-compose down
```

---

## 🎮 GPU Support

### Prerequisites
- NVIDIA GPU
- NVIDIA Docker Runtime
- CUDA Toolkit 11.8+
- cuDNN 8.x

### 1. Build GPU Image
```bash
docker build -f Dockerfile.gpu -t tesa-drone-detection:gpu .
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

