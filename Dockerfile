# 1. Use an official Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file into the container
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code
COPY . .

# 7. Create necessary directories
RUN mkdir -p data logs chroma_db

# 8. Expose the port that Streamlit runs on
EXPOSE 8501

# 9. Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ANONYMIZED_TELEMETRY=False
ENV PYTHONPATH=/app

# 10. Command to run the application
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
