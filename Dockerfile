# Use official Python runtime as base image
FROM python:3.11-slim

# Set environment variables for Java
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies including Java
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    build-essential \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Copy your project code into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set MLflow tracking URI
ENV MLFLOW_TRACKING_URI=http://your-mlflow-uri:5000

# Set up DVC
RUN dvc init --no-scm
RUN dvc remote add -d myremote <your-remote-url>  # replace with actual remote if not committed
COPY .dvc/config .dvc/config  # in case remote is preconfigured in repo

# Pull raw data only
RUN dvc pull

# Run pipeline to generate outputs
RUN dvc repro

# Expose port (adjust if your API runs on a different port)
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Test Stage ---
FROM base as test

CMD ["python", "test_api.py"]
