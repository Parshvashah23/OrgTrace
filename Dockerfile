FROM python:3.11-slim
WORKDIR /app

# Install system dependencies if needed for packages like scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-generate corpus at build time
RUN python generator.py --seed 42

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
