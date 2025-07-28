# Dockerfile

# Use an official Python runtime as a parent image.
FROM python:3.11-slim-buster

# Set the working directory in the container to /app.
WORKDIR /app

# Set environment variables for Hugging Face Hub cache.
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/hub

# Create the directory for Hugging Face cache.
RUN mkdir -p /app/.cache/huggingface/hub

# Copy requirements first (for better layer caching).
COPY requirements.txt .

# Install dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app.
COPY . .

# --- Nomic API Key Placeholder ---
# NOTE: Do NOT hardcode API keys in Dockerfile for security reasons.
# You will pass it as a runtime argument:
# docker run -e NOMIC_API_KEY=your_key_here <image_name>

# Expose port 8000.
EXPOSE 8000

# Start the FastAPI server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
