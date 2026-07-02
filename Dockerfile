# 1. Use an official Python base image
FROM python:3.10-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application
COPY . .

# 6. Create necessary directories with correct permissions
RUN mkdir -p data logs chroma_db && chmod -R 777 data logs chroma_db

# 7. Expose the port HuggingFace expects (7860)
EXPOSE 7860

# 8. Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ANONYMIZED_TELEMETRY=False

# 9. Run streamlit on port 7860
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
