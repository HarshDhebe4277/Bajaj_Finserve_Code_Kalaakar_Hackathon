# Dockerfile

# Use an official Python runtime as a parent image.
FROM python:3.11-slim-buster

# Set the working directory in the container to /app.
WORKDIR /app

# Set environment variables for Hugging Face Hub cache.
# This ensures models/cache are written to a writable directory inside our app.
ENV HF_HOME /app/.cache/huggingface
ENV TRANSFORMERS_CACHE /app/.cache/huggingface/hub/

# Create the directory for Hugging Face cache.
RUN mkdir -p /app/.cache/huggingface/hub/ # Removed chmod -R 777 as it's not strictly needed for this path

# Copy the requirements.txt file into the container at /app.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# --- Removed: NLTK download lines as unstructured is no longer used for email parsing ---
# --- Removed: Sentence-Transformers pre-cache as we're using Nomic ---

# Copy the rest of your application code into the container at /app.
COPY . .

# Expose port 8000.
EXPOSE 8000

# Define the command to run your application when the container starts.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]