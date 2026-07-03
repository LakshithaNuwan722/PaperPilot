# 1. Official Python image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install essential system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Create a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# 5. Copy requirements and install (as user)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 6. Copy the rest of the app with correct ownership
COPY --chown=user . .

# 7. Create necessary directories in user space
RUN mkdir -p /app/data /app/logs /app/chroma_db

# 8. Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ANONYMIZED_TELEMETRY=False

# 9. Run streamlit on port 7860
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
