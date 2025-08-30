# --- Base Stage ---
FROM python:3.11-slim AS base

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    build-essential \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Set updated JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Copy everything
COPY . .

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# DVC Setup
RUN dvc init --no-scm
COPY .dvc/config .dvc/config
RUN dvc pull
RUN dvc repro

EXPOSE 8000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Test Stage ---
FROM base as test
CMD ["python", "test_api.py"]
