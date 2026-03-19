FROM python:3.11-slim

# Prevent Python from writing .pyc files and force stdout/stderr flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System packages often needed for scientific/python builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better layer caching
COPY pyproject.toml ./
COPY README.md ./

# Copy source needed for editable/local install
COPY src ./src
COPY main.py ./

# Install the package
RUN pip install --upgrade pip && pip install .

# Copy the rest of the project
COPY config ./config
COPY examples ./examples
COPY tests ./tests
COPY .env.example ./.env.example

# Expose FastAPI port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "waam_rag.api.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]