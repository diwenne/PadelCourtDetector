# Use offical Python slim image with stable support
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python packages optimizing for CPU
# Standard Fly.io nodes do NOT have GPUs; running on CPU is completely fine for single frames
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    numpy \
    opencv-python-headless \
    scipy \
    python-multipart

# Copy weights inside container (relative to context)
COPY exps/ /app/exps/

# Copy necessary code files
COPY app.py predictor.py tracknet.py postprocess.py ./

EXPOSE 8000

# Start Uvicorn running the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
