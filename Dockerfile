# Use a stable and compatible Python base image
FROM python:3.11-slim-bookworm

# Disable Python bytecode and buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required for PyQt5 GUI
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libgl1 \
    libsm6 \
    libice6 \
    libfontconfig1 \
    libfreetype6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose optional port
EXPOSE 8000

# Default startup command
CMD ["python3", "main.py"]
