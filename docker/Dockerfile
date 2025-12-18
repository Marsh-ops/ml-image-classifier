# Use a stable slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for Pillow and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY ../requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ../src/ ./src/

# Copy trained model weights
COPY ../model.pth ./model.pth

# Expose API port
EXPOSE 8000

# Set entrypoint to run FastAPI via uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]