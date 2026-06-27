# Use a lightweight official Python runtime base image
FROM python:3.11-slim

# Prevent Python from writing pyc files to disc and enable live stream logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DAMAGE_MODEL_PATH="/app/models/car_damage_segformer" \
    PARTS_MODEL_PATH="/app/models/car_parts_yolo26_seg.pt" \
    DAMAGE_MIN_AREA="16" \
    DAMAGE_CONFIDENCE_THRESHOLD="0.30" \
    DAMAGE_ROI_ENABLED="true" \
    DAMAGE_ROI_PADDING_RATIO="0.08" \
    DAMAGE_ROI_MIN_PADDING="32" \
    PART_COVERAGE_THRESHOLD="0.50" \
    PORT=8000

WORKDIR /app

# Install system dependencies required by OpenCV and native standard libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install package dependencies in a single layer
COPY requirements.txt /app/
RUN pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create separate directory path for model assets
RUN mkdir -p /app/models && \
    useradd -m -u 10001 appuser && \
    mkdir -p /home/appuser/.config /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser

# Create a non-privileged user to run application processes (Security Best Practice)
RUN chown -R appuser:appuser /app

ENV MPLCONFIGDIR="/tmp/matplotlib-cache" \
    YOLO_CONFIG_DIR="/tmp"


# Copy application files
COPY --chown=appuser:appuser app/ /app/app/

# COPY THE MODEL DIRECTLY INTO THE IMAGE
COPY --chown=appuser:appuser models/ /app/models/

# Switch context to the non-privileged user
USER appuser

EXPOSE 8000

# Start the application using Uvicorn
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
