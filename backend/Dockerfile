# Use a lightweight base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer flushing issues
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first to leverage layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and app code
COPY models/ /app/models/
COPY app.py /app/app.py

# Expose port
EXPOSE 8000

# Command to run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
