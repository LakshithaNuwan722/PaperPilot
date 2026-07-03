# 1. Use official Python image
FROM python:3.10-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# 3. Create a non-root user (Hugging Face standard)
RUN useradd -m -u 1000 user

# 4. Set the working directory to the user's home folder
WORKDIR $HOME/app

# 5. Install system dependencies (as root)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 6. Copy and install requirements (as root, but for the user)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the application code
COPY . .

# 8. Create data/log folders inside the app directory
# These will now have the correct permissions
RUN mkdir -p data logs chroma_db && \
    chown -R user:user $HOME/app

# 9. Switch to the non-root user
USER user

# 10. Expose the port Hugging Face expects
EXPOSE 7860

# 11. Run the application
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
