FROM python:3.11-bullseye

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file used for dependencies
COPY requirements.txt .

# Install build dependencies and essential libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    g++ \
    ffmpeg \
    ninja-build \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    linux-headers-amd64 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Environment variables for the application
ENV PORT=8080

# Command to run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app